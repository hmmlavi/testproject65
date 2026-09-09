"""GOJO terminal chat — Phase 1, first runnable step.

This is a developer front-end to prove the brain works end to end
(real AI, GOJO personality) before we build the desktop window.

Run:
    python -m gojo.cli_chat

Inside chat:
    /help   show commands
    /clear  forget the current conversation
    /exit   quit
"""
from __future__ import annotations

import sys

from .ai.base import ChatMessage
from .ai.prompts import GOJO_SYSTEM_PROMPT
from .ai.router import build_router
from .config import load_settings

BANNER = r"""
  ____  ___ ____    _
 / ___|| _ / ___|  / \   GOJO  -  personal AI assistant
 \___ \| _ \___ \ / _ \   Phase 1: brain + text chat
 ____) | _ |__) |/ ___ \  type /help for commands
|_____/|___|____//_/   \_\
"""

HELP = """Commands:
  /help   this help
  /clear  reset the current conversation
  /exit   quit (or Ctrl+C)
"""


def main() -> int:
    settings = load_settings()

    try:
        brain = build_router(settings)
    except (RuntimeError, ValueError) as exc:
        print(f"Could not start GOJO's brain:\n\n{exc}\n")
        return 1

    print(BANNER)
    if brain.name == "mock":
        print("  !! MOCK MODE — offline wiring test only, NOT a real AI !!\n")
    else:
        print(f"  brain: {brain.name} ({settings.gemini_model})\n")

    history: list[ChatMessage] = []

    while True:
        try:
            user_text = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(bai boss — next time same time.)")
            return 0

        if not user_text:
            continue

        cmd = user_text.lower()
        if cmd in {"/exit", "/quit", "exit"}:
            print("bai boss.")
            return 0
        if cmd == "/help":
            print(HELP)
            continue
        if cmd == "/clear":
            history.clear()
            print("(conversation reset.)")
            continue

        history.append(ChatMessage(role="user", text=user_text))

        try:
            reply = brain.chat(history, GOJO_SYSTEM_PROMPT)
        except Exception as exc:  # network errors, auth errors, etc.
            print(f"  !! GOJO hit an error: {exc}")
            print("  (your message was NOT added to the conversation — try again)")
            history.pop()
            continue

        history.append(ChatMessage(role="model", text=reply))
        print(f"gojo ▸ {reply}\n")

    return 0  # unreachable, keeps type checkers happy


if __name__ == "__main__":
    sys.exit(main())
