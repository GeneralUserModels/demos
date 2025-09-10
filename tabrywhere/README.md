# Tabrywhere

[!WARNING]
Heads up- this project is a massive hack that relies on PyObjC and keyboard event tapping (capturing system-wide keystrokes and injecting synthetic text) plus a bunch of other things. It's definitely a proof of concept and will probably break often.

## What is this?

Tabrywhere is a system-wide AI autocomplete that lets you summon intelligent text completion anywhere on your Mac by holding the Fn key. Think of it as GitHub Copilot, but for literally any text field - emails, messages, documents, terminal, you name it.

## How it works

1. **Hold Fn** - Activates the system and takes a screenshot of your current screen
2. **Use GUM** - Uses the General User Model to retrieve context from the user
3. **Completions** - Streams helpful text completions directly into whatever app you're using
4. **Fn + Tab** - Commits the suggestion and keeps it
5. **Release Fn** - Cancels and erases everything if you don't like it

## Setup

1. Install with `uv` (or pip):
   ```bash
   uv sync
   ```

2. Set up your environment:
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY and USER_NAME
   ```

3. **Important**: Give the Terminal permissions in System Preferences → Security & Privacy → Accessibility

4. Run it:
   ```bash
   uv run python main.py
   ```

## Why this exists

Sometimes you just want AI autocomplete everywhere, not just in your code editor.