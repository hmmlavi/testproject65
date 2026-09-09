"""TTS providers (Phase 3).

base.py        — TTSProvider interface + TTSResult + TTSError
fish_audio.py  — Fish Audio cloud provider (primary; free tier model string)
windows_tts.py — local Windows SAPI fallback (free, offline, no download)
normalize.py   — speakable-text layer (TTS only, transcript untouched)
router.py      — provider chain with automatic fallback (never crashes)
"""
