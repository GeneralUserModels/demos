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
MODEL = os.getenv("MODEL", "gpt-4o-mini")  # change as needed
client = OpenAI()  # uses OPENAI_API_KEY env var

# Save settings
SAVE_DIR = Path(os.getenv("SAVE_DIR", "./"))
SAVE_FORMAT = os.getenv("SAVE_FORMAT", "png").lower()  # png or jpg
SAVE_QUALITY = int(os.getenv("SAVE_QUALITY", "92"))  # used for jpg
ANNOTATE_CURSOR = os.getenv("ANNOTATE_CURSOR", "1") not in ("0", "false", "False")
USER_NAME = os.getenv("USER_NAME")

JOINER = "\u2060"  # WORD JOINER (plays nice with Backspace)

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
    cx = mx - mon["left"]; cy = my - mon["top"]
    cx = max(0, min(cx, img.width - 1)); cy = max(0, min(cy, img.height - 1))
    draw = ImageDraw.Draw(img, "RGBA")
    halo_r = max(12, img.width // 150); ring_r = max(8, img.width // 220)
    ring_w = max(3, img.width // 700); arm = max(14, img.width // 180)
    draw.ellipse((cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r), fill=(255, 255, 255, 90))
    draw.ellipse((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r), outline=(255, 0, 0, 255), width=ring_w)
    draw.line((cx - arm, cy, cx + arm, cy), fill=(255, 0, 0, 255), width=ring_w)
    draw.line((cx, cy - arm, cx, cy + arm), fill=(255, 0, 0, 255), width=ring_w)
    dot_r = max(2, ring_w // 2)
    draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=(255, 0, 0, 255))
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
        {"role": "user", "content": [{"type": "input_text", "text": TAB_PROMPT}, {"type": "input_image", "image_url": data_url}]},
    ]

# ------------- Keycodes / Flags -------------
KC_TAB = 48       # 0x30
KC_BACKSPACE = 51 # 0x33
FN_FLAG = getattr(Quartz, "kCGEventFlagMaskSecondaryFn", 0x00080000)

# ------------- Tagging for our own events -------------
OUR_EVENT_TAG = 0xC0DEFEED

# ------------- State -------------
state = {
    "fn_down": False,
    "session_active": False,
    "inserted_len": 0,
    "keep_contents": False,
    "stream_thread": None,
    "cancel_event": None,
    "spinner_count": 0,          # placeholder + glyph
    "last_char_space": False,    # for boundary de-dupe
    "content_started": False,    # becomes True at first token (after spinner cleanup)
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
        # Send the whole chunk at once
        Quartz.CGEventKeyboardSetUnicodeString(down, len(s), s)
        _post_event(down)

        up = Quartz.CGEventCreateKeyboardEvent(src, 0, False)
        # DO NOT set a Unicode payload on keyUp
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
        type_text(FRAMES[(idx + 1) % len(FRAMES)])
        state["inserted_len"] += 1
        idx = (idx + 1) % len(FRAMES)

# ------------- GUM init + tool definition -------------
gum_instance: Optional[GumClass] = None

TOOLS: List[Dict[str, Any]] = [{
    "name": "get_user_context",
    "type": "function",
    "description": (
        "Retrieve contextual info for a user query within a relative time window. "
        "Use liberally."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Optional lexical query (e.g., 'what I was doing in vs code'). "
                    "Leave empty for broad context."
                ),
            },
            "start_hh_mm_ago": {
                "type": "string",
                "description": "Optional lower bound as 'HH:MM' ago from now (e.g., '01:00').",
            },
            "end_hh_mm_ago": {
                "type": "string",
                "description": "Optional upper bound as 'HH:MM' ago from now (e.g., '00:10').",
            },
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

def stream_worker(cancel_event: threading.Event):
    """
    1) Resolve tools.
    2) Stream and type content.
    Cleanup of spinner happens either:
      - just before the first content is typed, or
      - on commit/cancel if no content ever started.
    """
    messages = build_messages()

    # Start loading spinner
    first_piece_event = threading.Event()
    spinner = threading.Thread(target=loading_spinner, args=(first_piece_event, cancel_event), daemon=True)
    spinner.start()

    # ---- Phase 1: resolve tools ----
    resolved_messages = list(messages)
    # max_tool_rounds = 4 # disabled for now
 
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

    # ---- Phase 2: stream ----
    first_piece_seen_local = False
    stream =client.responses.create(
        model=MODEL, 
        instructions=f"Respond only with the completion using the user's context. Remember to NOT repeat what is already on the screen, and to mirror the same style as {USER_NAME}'s past writing.",
        input=resolved_messages, 
        stream=True, 
    )
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
                spinner.join()
                # Cleanup spinner before typing first content
                _cleanup_spinner_if_present()
                state["content_started"] = True

            safe_type_piece(piece)
    finally:
        first_piece_event.set()

# ------------- Stream Session Control -------------
def handle_fn_down():
    if state["session_active"]:
        return
    state["session_active"] = True
    state["keep_contents"] = False
    state["inserted_len"] = 0
    state["spinner_count"] = 0
    state["last_char_space"] = False
    state["content_started"] = False

    cancel = threading.Event()
    state["cancel_event"] = cancel

    t = threading.Thread(target=stream_worker, args=(cancel,), daemon=True)
    state["stream_thread"] = t
    t.start()

def stop_stream(join: bool = True):
    if state["cancel_event"]:
        state["cancel_event"].set()
    if join and state["stream_thread"] and state["stream_thread"].is_alive():
        state["stream_thread"].join(timeout=0.5)
    state["stream_thread"] = None
    state["cancel_event"] = None

def handle_fn_up():
    if not state["session_active"]:
        return
    # Early release = cancel & erase everything inserted
    stop_stream(join=True)
    if not state["keep_contents"] and state["inserted_len"] > 0:
        press_backspace(state["inserted_len"])
    # Reset state
    state["session_active"] = False
    state["inserted_len"] = 0
    state["keep_contents"] = False
    state["spinner_count"] = 0
    state["last_char_space"] = False
    state["content_started"] = False

# ------------- Event Tap Callback -------------
def callback(proxy, event_type, event, refcon):
    global tap_ref

    # If the tap times out, macOS disables it. Re-enable.
    if event_type == Quartz.kCGEventTapDisabledByTimeout:
        if tap_ref is not None:
            Quartz.CGEventTapEnable(tap_ref, True)
        return event

    # Track Fn changes
    if event_type == Quartz.kCGEventFlagsChanged:
        flags = Quartz.CGEventGetFlags(event)
        fn_now = bool(flags & FN_FLAG)
        if fn_now and not state["fn_down"]:
            state["fn_down"] = True
            handle_fn_down()
        elif not fn_now and state["fn_down"]:
            state["fn_down"] = False
            handle_fn_up()
        return event

    # Swallow real key events while Fn is down; let our own tagged events pass.
    if event_type in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp):
        tag = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData)
        is_ours = (tag == OUR_EVENT_TAG)

        if state["fn_down"] and not is_ours:
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                autorepeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)

                # Fn+Tab → commit
                if keycode == KC_TAB and not autorepeat:
                    if not state.get("content_started", False):
                        _cleanup_spinner_if_present()  # ensure nothing remains
                        state["inserted_len"] = 0
                    state["keep_contents"] = True
                    stop_stream(join=False)
                    return None
            # Swallow all other hardware key events while Fn held
            return None

        return event

    return event

# ------------- Main -------------
def main():
    global tap_ref

    try:
        ensure_gum_initialized()
        print("GUM initialized.")
    except Exception as e:
        print(f"Warning: GUM failed to initialize now. Will retry on first use. Error: {e}")

    mask = (Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged) |
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) |
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

    print("Hold Fn to stream. Fn+Tab commits. Release Fn to cancel and erase. Ctrl+C to quit.")
    Quartz.CFRunLoopRun()

if __name__ == "__main__":
    main()
