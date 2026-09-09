"""UI package.

web/  : static front-end (HTML/CSS/JS) loaded inside the pywebview window
api.py: the JS <-> Python bridge — every UI capability goes through it
"""
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
