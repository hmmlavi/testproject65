"""Voice package (Phase 3).

recorder.py  — microphone capture (press-to-talk), CPU-friendly
stt.py       — faster-whisper speech-to-text (local, auto language)
pipeline.py  — orchestrates: record -> STT -> shared brain path -> TTS -> play
playback.py  — audio playback (Windows: winsound) + temp file cleanup
tts/         — TTS providers: base (interface), fish_audio, windows_tts,
               normalize (speech text layer), router (fallback chain)
wakeword.py  — WakeWordEngine interface (Phase 4; NOT active in Phase 3)

Privacy default: the microphone is only ever open while the user is
press-to-talking. No continuous listening in Phase 3.
"""
