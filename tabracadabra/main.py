#!/usr/bin/env python3
import os
import threading
import time
import base64
import io
import asyncio
import json
import re
from typing import Optional, List, Dict, Any

import Quartz
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import pdb

# deps for screenshot + image encoding
import mss
from PIL import Image, ImageDraw

# gum imports
from gum import gum as GumClass
from gum.db_utils import get_related_observations

load_dotenv()

# ------------- Config -------------
MODEL = os.getenv("MODEL", "gpt-4.1")  # change as needed
client = OpenAI()  # uses OPENAI_API_KEY env var

# Save settings
SAVE_DIR = Path(os.getenv("SAVE_DIR", "./"))
SAVE_FORMAT = os.getenv("SAVE_FORMAT", "png").lower()  # png or jpg
SAVE_QUALITY = int(os.getenv("SAVE_QUALITY", "92"))  # used for jpg
ANNOTATE_CURSOR = os.getenv("ANNOTATE_CURSOR", "1") not in ("0", "false", "False")
USER_NAME = os.getenv("USER_NAME")

JOINER = "\u2060"  # WORD JOINER (plays nice with Backspace)

HOLD_THRESHOLD = float(os.getenv("HOLD_THRESHOLD", "1.0"))  # seconds for Tab hold activation

with open("tab_prompt.txt", "r") as f:
    TAB_PROMPT = f.read().format(USER_NAME=USER_NAME)

# ------------- Screenshot helpers (mss + Quartz) -------------
def _get_cursor_point():
    ev = Quartz.CGEventCreate(None)
    loc = Quartz.CGEventGetLocation(ev)
    return int(loc.x), int(loc.y)

def _find_monitor_for_point(monitors, x, y):
    for mon in monitors[1:]:
        left, top = mon["left"], mon["top"]
        right = left + mon["width"]
        bottom = top + mon["height"]
        if left <= x < right and top <= y < bottom:
            return mon
    return monitors[1] if len(monitors) > 1 else monitors[0]

def _annotate_with_cursor(img: Image.Image, mon: dict, mx: int, my: int) -> Image.Image:
    # Calculate cursor position relative to this monitor capture
    cx = mx - mon["left"]
    cy = my - mon["top"]
    cx = max(0, min(cx, img.width - 1))
    cy = max(0, min(cy, img.height - 1))

    # Estimate DPI (pixels per inch)
    width_mm = mon.get("width_mm")
    height_mm = mon.get("height_mm")
    if width_mm and height_mm:
        dpi_x = mon["width"] / (width_mm / 25.4)
        dpi_y = mon["height"] / (height_mm / 25.4)
        dpi = (dpi_x + dpi_y) / 2
    else:
        dpi = 96  # fallback guess

    # Scale dot radius so it's about 12 px at 96 DPI and scales proportionally
    base_size_at_96dpi = 12
    dot_r = int((dpi / 96) * base_size_at_96dpi)
    outline_w = max(2, dot_r // 4)

    draw = ImageDraw.Draw(img, "RGBA")

    # White border
    draw.ellipse(
        (cx - dot_r - outline_w, cy - dot_r - outline_w,
         cx + dot_r + outline_w, cy + dot_r + outline_w),
        fill=(255, 255, 255, 255)
    )

    # Red center
    draw.ellipse(
        (cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
        fill=(255, 0, 0, 255)
    )

    return img

def _save_image(img: Image.Image) -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = "jpg" if SAVE_FORMAT in ("jpg", "jpeg") else "png"
    path = SAVE_DIR / f"screenshot_{ts}.{ext}"
    if ext == "jpg":
        img_to_save = img.convert("RGB")
        img_to_save.save(path, format="JPEG", quality=SAVE_QUALITY, optimize=True, progressive=True)
    else:
        img.save(path, format="PNG", optimize=True)
    return str(path)

def capture_active_monitor_as_data_url(max_width=1600, jpeg_quality=85, annotate_cursor=ANNOTATE_CURSOR):
    with mss.mss() as sct:
        mx, my = _get_cursor_point()
        mon = _find_monitor_for_point(sct.monitors, mx, my)
        raw = sct.grab(mon)  # BGRA
        img = Image.frombytes("RGB", raw.size, raw.rgb)

        if annotate_cursor:
            img = _annotate_with_cursor(img, mon, mx, my)

        out_img = img
        if out_img.width > max_width:
            ratio = max_width / out_img.width
            out_img = out_img.resize((max_width, int(out_img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"
        return data_url, None

def build_messages():
    data_url, saved_path = capture_active_monitor_as_data_url()
    print(f"[Saved annotated screenshot] {saved_path}")
    return [
        {"role": "system", "content": "You are a concise, helpful autocomplete system. When context feels underspecified or time-bound, call the get_user_context tool before answering."},
        {"role": "user", "content": [{"type": "input_image", "image_url": data_url}, {"type": "input_text", "text": TAB_PROMPT}]},
    ]

# ------------- Keycodes -------------
KC_TAB = 48       # 0x30
KC_BACKSPACE = 51 # 0x33

# ------------- Tagging for our own events -------------
OUR_EVENT_TAG = 0xC0DEFEED

# ------------- State -------------
state = {
    "session_active": False,      # streaming session is running
    "inserted_len": 0,            # how many chars we've injected into the app
    "keep_contents": False,       # commit vs erase on release
    "stream_thread": None,
    "cancel_event": None,         # cancel the spinner/stream
    "spinner_count": 0,           # placeholder + glyph
    "last_char_space": False,
    "content_started": False,     # True after first model token typed

    # Tab-hold detection
    "tab_down": False,
    "tab_down_t0": 0.0,
    "activated": False,           # became a "hold" (>= threshold) and started streaming
    "activation_timer": None,
    "first_piece_event": None,
    "spinner_thread": None,
    "spinner_active": False,
}

# Global I/O lock
io_lock = threading.RLock()

# Keep a global reference to the tap so we can re-enable on timeout
tap_ref = None

# ------------- Event posting helpers -------------
def _post_event(ev):
    Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, OUR_EVENT_TAG)
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)

# ------------- Typing helpers (CRITICAL FIX) -------------
def _keyboard_text_insert(s: str):
    """
    Insert the entire string in ONE keyDown with a Unicode payload,
    followed by a keyUp with NO Unicode payload. This avoids apps
    seeing 'two spaces' or duplicating characters on keyUp.
    """
    if not s:
        return
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
    with io_lock:
        down = Quartz.CGEventCreateKeyboardEvent(src, 0, True)
        Quartz.CGEventKeyboardSetUnicodeString(down, len(s), s)
        _post_event(down)

        up = Quartz.CGEventCreateKeyboardEvent(src, 0, False)
        _post_event(up)

def type_text(s: str):
    _keyboard_text_insert(s)

def press_backspace(times: int):
    if times <= 0:
        return
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
    with io_lock:
        for _ in range(times):
            down = Quartz.CGEventCreateKeyboardEvent(src, KC_BACKSPACE, True)
            _post_event(down)
            up = Quartz.CGEventCreateKeyboardEvent(src, KC_BACKSPACE, False)
            _post_event(up)

# ---- Normalization to eliminate autocorrect triggers & glue ----
_ZW_CHARS = ("\u200B", "\u2060")  # ZWSP + WORD JOINER
def _normalize_piece(piece: str) -> str:
    if not piece:
        return piece
    # Remove zero-widths outright
    for zw in _ZW_CHARS:
        piece = piece.replace(zw, "")
    # Convert non-breaking space to regular space
    piece = piece.replace("\u00A0", " ")
    # Collapse runs of 2+ ASCII spaces inside the chunk
    piece = re.sub(r" {2,}", " ", piece)
    return piece

# Boundary-safe typing to avoid double-space → period
def safe_type_piece(piece: str):
    if not piece:
        return
    piece = _normalize_piece(piece)
    # Drop a single leading space if we already ended with a space
    if state.get("last_char_space") and piece.startswith(" "):
        piece = piece[1:]
    if not piece:
        return
    type_text(piece)
    state["inserted_len"] += len(piece)
    state["last_char_space"] = (piece[-1] == " ")

# ------------- Loading animation (spinner) -------------
def loading_spinner(first_piece_event: threading.Event, cancel_event: threading.Event):
    """
    Shows JOINER + spinner glyph until first token or cancel.
    Cleanup is handled elsewhere.
    """
    try:
        FRAMES = ["◐", "◓", "◑", "◒"]
        INTERVAL = 0.12

        # Seed with placeholder + first frame
        type_text(JOINER + FRAMES[0])
        state["inserted_len"] += 2
        state["spinner_count"] = 2
        state["last_char_space"] = False

        idx = 0

        def wait_with_checks(seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if first_piece_event.is_set() or cancel_event.is_set():
                    return True
                time.sleep(0.02)
            return False

        while not first_piece_event.is_set() and not cancel_event.is_set():
            if wait_with_checks(INTERVAL):
                break

            # swap just the spinner glyph (keep placeholder)
            press_backspace(1)
            state["inserted_len"] -= 1
            # Reflect that only the JOINER remains on screen right now
            state["spinner_count"] = max(0, state["spinner_count"] - 1)

            # Bail if content/cleanup started during the swap to avoid typing after cleanup
            if first_piece_event.is_set() or cancel_event.is_set():
                break

            # Re-type next glyph
            next_frame = FRAMES[(idx + 1) % len(FRAMES)]
            type_text(next_frame)
            state["inserted_len"] += 1
            # JOINER + glyph are present again
            state["spinner_count"] += 1

            idx = (idx + 1) % len(FRAMES)
    finally:
        state["spinner_active"] = False

# ------------- GUM init + tool definition -------------
gum_instance: Optional[GumClass] = None

TOOLS: List[Dict[str, Any]] = [{
    "name": "get_user_context",
    "type": "function",
    "description": (
        "Retrieve contextual info for a user query."
        "Use liberally. Time window is only needed if the user mentions a specific time period."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Optional lexical query (e.g., 'what I was doing in vs code'). "
                    "Leave empty for broad context. Note that is is a BM25 index, it's NOT semantic search."
                ),
            }
        },
        "required": [],
    }
}]

async def _init_gum_async() -> GumClass:
    gi = GumClass(USER_NAME, None)  # model=None
    await gi.connect_db()
    return gi

def ensure_gum_initialized():
    global gum_instance
    if gum_instance is None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            gum_instance = loop.run_until_complete(_init_gum_async())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

async def _get_user_context_async(query: Optional[str] = "", start_hh_mm_ago: Optional[str] = None, end_hh_mm_ago: Optional[str] = None) -> str:
    from datetime import datetime, timezone, timedelta
    gi = gum_instance
    if gi is None:
        raise RuntimeError("gum_instance not initialized")
    now = datetime.now(timezone.utc)
    start_time = end_time = None
    if start_hh_mm_ago:
        hh, mm = map(int, start_hh_mm_ago.split(":"))
        start_time = now - timedelta(hours=hh, minutes=mm)
    if end_hh_mm_ago:
        hh, mm = map(int, end_hh_mm_ago.split(":"))
        end_time = now - timedelta(hours=hh, minutes=mm)
    results = await gi.query(query or "", start_time=start_time, end_time=end_time, limit=3)
    if not results:
        return "No relevant context found for the given query and time window."
    parts = []
    async with gi._session() as session:
        for proposition, score in results:
            txt = f"• {proposition.text}"
            if getattr(proposition, "reasoning", None): txt += f"\n  Reasoning: {proposition.reasoning}"
            if getattr(proposition, "confidence", None): txt += f"\n  Confidence: {proposition.confidence}"
            txt += f"\n  Relevance Score: {score:.2f}"
            observations = await get_related_observations(session, proposition.id, limit=1)
            if observations:
                txt += "\n  Supporting Observations:"
                for obs in observations:
                    txt += f"\n    - [{obs.observer_name}] {obs.content}"
            parts.append(txt)
    return "\n\n".join(parts)

def get_user_context_tool_call(args_json: str) -> str:
    ensure_gum_initialized()
    try:
        args = json.loads(args_json or "{}")
    except Exception:
        args = {}
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_get_user_context_async(args.get("query", "") or "",
                                                              args.get("start_hh_mm_ago"),
                                                              args.get("end_hh_mm_ago")))
    finally:
        loop.close()
        asyncio.set_event_loop(None)

# ------------- Streaming workers (with tool loop) -------------
def _cleanup_spinner_if_present():
    sc = state.get("spinner_count", 0)
    if sc > 0:
        press_backspace(sc)
        state["inserted_len"] = max(0, state["inserted_len"] - sc)
        state["spinner_count"] = 0
        state["last_char_space"] = False

def stream_worker(cancel_event: threading.Event, first_piece_event: threading.Event):
    """
    1) Resolve tools.
    2) Stream and type content.
    Spinner is started on Tab keydown; we only clear it when first content arrives.
    """
    messages = build_messages()

    # ---- Phase 1: resolve tools ----
    resolved_messages = list(messages)
    pre = client.responses.create(
        model=MODEL,
        input=resolved_messages,
        tool_choice="required",
        tools=TOOLS,
    )

    resolved_messages += pre.output

    for item in pre.output:
        if item.type == "function_call":
            if item.name == "get_user_context":
                user_context = get_user_context_tool_call(item.arguments)
                resolved_messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": user_context
                })

    instructions = "Respond with a helpful continuation for the textbox in the image."
    resolved_messages.append({"role": "user", "content": [{"type": "input_text", "text": instructions}]})

    # ---- Phase 2: stream ----
    first_piece_seen_local = False
    stream = client.responses.create(model=MODEL, input=resolved_messages, stream=True)
    try:
        for chunk in stream:
            if cancel_event.is_set():
                break

            piece = getattr(chunk, "delta", None)
            if not piece:
                continue

            if not first_piece_seen_local:
                first_piece_seen_local = True
                # Stop spinner and wait so no interleaving
                first_piece_event.set()
                # Cleanup spinner before typing first content
                _cleanup_spinner_if_present()
                state["content_started"] = True

            safe_type_piece(piece)
    finally:
        first_piece_event.set()

# ------------- Tab-based control -------------
def _start_spinner(cancel_event: threading.Event, first_piece_event: threading.Event):
    state["spinner_active"] = True
    t = threading.Thread(target=loading_spinner, args=(first_piece_event, cancel_event), daemon=True)
    t.start()
    state["spinner_thread"] = t

def _start_stream():
    # assumes spinner already running
    state["session_active"] = True
    t = threading.Thread(target=stream_worker, args=(state["cancel_event"], state["first_piece_event"]), daemon=True)
    state["stream_thread"] = t
    t.start()

def _activation_timer_body(start_t: float, cancel_event: threading.Event):
    # Wait until threshold; if still held, activate
    while True:
        if cancel_event.is_set() or not state["tab_down"]:
            return
        if time.monotonic() - start_t >= HOLD_THRESHOLD:
            if not state["activated"]:
                state["activated"] = True
                _start_stream()
            return
        time.sleep(0.01)

def handle_tab_down():
    if state["tab_down"]:
        return
    state["tab_down"] = True
    state["tab_down_t0"] = time.monotonic()
    state["activated"] = False
    state["keep_contents"] = False
    state["inserted_len"] = 0
    state["spinner_count"] = 0
    state["last_char_space"] = False
    state["content_started"] = False

    cancel = threading.Event()
    state["cancel_event"] = cancel
    first_piece_event = threading.Event()
    state["first_piece_event"] = first_piece_event

    # Start spinner immediately on press
    _start_spinner(cancel, first_piece_event)

    # Arm activation timer (pass local cancel_event to avoid races)
    at = threading.Thread(target=_activation_timer_body, args=(state["tab_down_t0"], cancel), daemon=True)
    state["activation_timer"] = at
    at.start()

def _synthesize_tab_keypress():
    # Generate a real Tab hardware press, not a Unicode \t (more compatible)
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStatePrivate)
    down = Quartz.CGEventCreateKeyboardEvent(src, KC_TAB, True)
    _post_event(down)
    up = Quartz.CGEventCreateKeyboardEvent(src, KC_TAB, False)
    _post_event(up)

def _stop_stream(join: bool = True):
    if state["cancel_event"]:
        state["cancel_event"].set()
    if join and state["stream_thread"] and state["stream_thread"].is_alive():
        state["stream_thread"].join(timeout=0.5)
    state["stream_thread"] = None
    # leave cancel_event clearing to spinner cleanup path

def _stop_spinner_and_cleanup():
    # signal stop
    if state["cancel_event"] and not state["cancel_event"].is_set():
        state["cancel_event"].set()
    # join spinner thread if present
    t = state.get("spinner_thread")
    if t and t.is_alive():
        t.join(timeout=0.5)
    # erase any spinner chars left
    _cleanup_spinner_if_present()
    # clear refs
    state["spinner_thread"] = None

def handle_tab_up():
    if not state["tab_down"]:
        return

    state["tab_down"] = False
    was_activated = state["activated"]

    if not was_activated:
        # Quick tap: stop spinner first, then synthesize a real Tab
        _stop_spinner_and_cleanup()
        _synthesize_tab_keypress()
    else:
        # Streaming: commit and stop stream
        state["keep_contents"] = True
        _stop_stream(join=False)
        if not state["content_started"]:
            _stop_spinner_and_cleanup()

    # reset transient flags AFTER cleanup to avoid races
    state["activation_timer"] = None
    state["first_piece_event"] = None
    state["content_started"] = False
    state["inserted_len"] = 0
    state["last_char_space"] = False
    state["cancel_event"] = None

# ------------- Event Tap Callback -------------
def callback(proxy, event_type, event, refcon):
    global tap_ref

    # If the tap times out, macOS disables it. Re-enable.
    if event_type == Quartz.kCGEventTapDisabledByTimeout:
        if tap_ref is not None:
            Quartz.CGEventTapEnable(tap_ref, True)
        return event

    # Only key events matter now
    if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
        tag = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData)
        is_ours = (tag == OUR_EVENT_TAG)

        # Always pass through our own synthetic events
        if is_ours:
            return event

        keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)

        # Intercept ALL physical Tab events (including autorepeat)
        if keycode == KC_TAB:
            if event_type == Quartz.kCGEventKeyDown:
                autorepeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
                # Run handle_tab_down only on the first non-autorepeat, but always swallow
                if not autorepeat and not state["tab_down"]:
                    handle_tab_down()
                return None  # swallow hardware Tab down (and repeats)
            else:  # KeyUp
                handle_tab_up()
                return None  # swallow hardware Tab up

        # Let other keys pass through normally
        return event

    # Ignore other event types
    return event

# ------------- Main -------------
def main():
    global tap_ref

    try:
        ensure_gum_initialized()
        print("GUM initialized.")
    except Exception as e:
        print(f"Warning: GUM failed to initialize now. Will retry on first use. Error: {e}")

    mask = (Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) |
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp))
    tap_ref = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        mask,
        callback,
        None
    )
    if not tap_ref:
        raise RuntimeError("Failed to create event tap. Check Accessibility permissions.")

    src = Quartz.CFMachPortCreateRunLoopSource(None, tap_ref, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap_ref, True)

    print("Hold Tab to stream. Quick tap inserts a normal Tab. Ctrl+C to quit.")
    Quartz.CFRunLoopRun()

if __name__ == "__main__":
    main()
