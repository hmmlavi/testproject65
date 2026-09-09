"""Checks for the TTS layer: provider interface, router + fallback,
Fish Audio request format, normalization, playback cleanup.

Uses fake transports/fakes — no network, no audio, no API key needed.
The Fish Audio request shape is asserted against the CURRENT documented
API (verified 2026-09): POST /v1/tts, `model` in the HTTP HEADER,
`text` + `reference_id` in the JSON body, raw MP3 bytes back.

Run:  python -m tests.test_tts
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from gojo.voice.playback import AudioPlayer
from gojo.voice.tts.base import TTSError, TTSProvider, TTSResult
from gojo.voice.tts.fish_audio import FishAudioTTS
from gojo.voice.tts.normalize import normalize_for_speech
from gojo.voice.tts.router import TTSRouter
from gojo.voice.tts.windows_tts import WindowsTTS


class FakeProvider(TTSProvider):
    def __init__(self, name: str, available: bool = True, fail: bool = False) -> None:
        self.name = name
        self._available = available
        self._fail = fail
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, out_dir: Path) -> Path:
        self.calls += 1
        if self._fail:
            raise TTSError(f"{self.name} exploded", provider=self.name)
        path = out_dir / f"fake-{self.name}.wav"
        path.write_bytes(b"RIFF-fake-audio")
        return path


# ---------------------------------------------------------------------
# router / fallback
# ---------------------------------------------------------------------


def test_router_uses_first_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish, windows = FakeProvider("fish"), FakeProvider("windows")
        router = TTSRouter([fish, windows], order="auto")
        result = router.synthesize("hello", Path(tmp))
        assert result.provider == "fish" and not result.fell_back
        assert fish.calls == 1 and windows.calls == 0
        assert result.path.exists()
    print("PASS test_router_uses_first_available")


def test_router_falls_back_on_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish, windows = FakeProvider("fish", fail=True), FakeProvider("windows")
        router = TTSRouter([fish, windows], order="fish")
        result = router.synthesize("hello", Path(tmp))
        assert result.provider == "windows" and result.fell_back
        assert router.last_fallback_used is True
    print("PASS test_router_falls_back_on_failure")


def test_router_skips_unavailable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish, windows = FakeProvider("fish", available=False), FakeProvider("windows")
        router = TTSRouter([fish, windows], order="auto")
        result = router.synthesize("hello", Path(tmp))
        assert result.provider == "windows" and result.fell_back and fish.calls == 0
    print("PASS test_router_skips_unavailable")


def test_router_all_failed_raises_with_details() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish, windows = FakeProvider("fish", fail=True), FakeProvider("windows", fail=True)
        router = TTSRouter([fish, windows], order="auto")
        try:
            router.synthesize("hello", Path(tmp))
        except TTSError as exc:
            msg = str(exc)
            assert "fish" in msg and "windows" in msg, "error must list every attempt"
            return
        raise AssertionError("all-failed chain did not raise")
    print("PASS test_router_all_failed_raises_with_details")


def test_router_order_windows_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fish, windows = FakeProvider("fish"), FakeProvider("windows")
        router = TTSRouter([fish, windows], order="windows")
        assert router.chain == ["windows"]
        result = router.synthesize("hello", Path(tmp))
        assert result.provider == "windows" and not result.fell_back and fish.calls == 0
    print("PASS test_router_order_windows_only")


# ---------------------------------------------------------------------
# fish audio provider (request format verified against live docs)
# ---------------------------------------------------------------------


FAKE_MP3 = b"ID3-fake-mp3" + b"\x00" * 2048  # comfortably above the 1 KB sanity floor


def _make_fish(captured: dict, response: bytes = FAKE_MP3, status: int = 200) -> FishAudioTTS:
    def fake_post(url: str, headers: dict[str, str], data: bytes) -> tuple[int, bytes]:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(data)
        return status, response

    return FishAudioTTS(api_key="test-key", model_id="0a7bf4b832b2459eb012547b4e3643d3",
                        model="s2.1-pro-free", post=fake_post)


def test_fish_request_format() -> None:
    cap: dict[str, Any] = {}
    fish = _make_fish(cap)
    with tempfile.TemporaryDirectory() as tmp:
        out = fish.synthesize("hello boss", Path(tmp))
        written = out.read_bytes()
        suffix = out.suffix
    assert cap["url"] == "https://api.fish.audio/v1/tts"
    assert cap["headers"]["Authorization"] == "Bearer test-key"
    assert cap["headers"]["model"] == "s2.1-pro-free"  # model is a HEADER
    assert cap["body"]["text"] == "hello boss"
    assert cap["body"]["reference_id"] == "0a7bf4b832b2459eb012547b4e3643d3"
    assert cap["body"]["format"] == "mp3"
    assert suffix == ".mp3" and written == FAKE_MP3
    print("PASS test_fish_request_format")


def test_fish_requires_real_key() -> None:
    for bad_key in (None, "", "paste-your-key-here"):
        fish = FishAudioTTS(api_key=bad_key, model_id="x", model="s2.1-pro-free")
        assert fish.available is False, f"key {bad_key!r} must mark provider unavailable"
    print("PASS test_fish_requires_real_key")


def test_fish_http_error_raises_ttser() -> None:
    cap: dict[str, Any] = {}
    fish = _make_fish(cap, response=json.dumps({"detail": "Invalid API key"}).encode(), status=401)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fish.synthesize("hi", Path(tmp))
        except TTSError as exc:
            assert "401" in str(exc)
            return
        raise AssertionError("HTTP 401 did not raise TTSError")
    print("PASS test_fish_http_error_raises_ttser")


def test_fish_rate_limit_is_retryable() -> None:
    cap: dict[str, Any] = {}
    fish = _make_fish(cap, response=b"rate limited", status=429)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fish.synthesize("hi", Path(tmp))
        except TTSError as exc:
            assert exc.retryable is True
            return
        raise AssertionError("429 did not raise TTSError")
    print("PASS test_fish_rate_limit_is_retryable")


def test_fish_network_error_is_retryable() -> None:
    def boom(url: str, headers: dict[str, str], data: bytes) -> tuple[int, bytes]:
        raise OSError("no network")

    fish = FishAudioTTS(api_key="k", model_id="x", model="s2.1-pro-free", post=boom)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fish.synthesize("hi", Path(tmp))
        except TTSError as exc:
            assert exc.retryable is True
            return
        raise AssertionError("network error did not raise TTSError")
    print("PASS test_fish_network_error_is_retryable")


def test_fish_tiny_payload_rejected() -> None:
    """A near-empty response is a bad payload, not valid audio."""
    cap: dict[str, Any] = {}
    fish = _make_fish(cap, response=b"ok", status=200)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fish.synthesize("hi", Path(tmp))
        except TTSError:
            return
        raise AssertionError("tiny 200-response was accepted as audio")
    print("PASS test_fish_tiny_payload_rejected")


# ---------------------------------------------------------------------
# windows fallback provider
# ---------------------------------------------------------------------


def test_windows_provider_reports_platform() -> None:
    tts = WindowsTTS()
    import sys
    expected = sys.platform == "win32"
    assert tts.available is expected
    # on a non-Windows dev box it must degrade cleanly, never crash the router
    if not expected:
        with tempfile.TemporaryDirectory() as tmp:
            router = TTSRouter([WindowsTTS()], order="windows")
            try:
                router.synthesize("hi", Path(tmp))
            except TTSError:
                return
            raise AssertionError("unavailable windows TTS should raise TTSError via router")
    print("PASS test_windows_provider_reports_platform")


def test_windows_provider_synthesizes_on_windows() -> None:
    """Runs (and is verified) on the user's Windows PC; on other platforms
    it is skipped gracefully rather than faked."""
    import sys
    if sys.platform != "win32":
        print("PASS test_windows_provider_synthesizes_on_windows (skipped: not Windows here)")
        return
    tts = WindowsTTS()
    with tempfile.TemporaryDirectory() as tmp:
        out = tts.synthesize("hello boss", Path(tmp))
        assert out.exists() and out.stat().st_size > 1000 and out.suffix == ".wav"
    print("PASS test_windows_provider_synthesizes_on_windows")


# ---------------------------------------------------------------------
# normalization (speakable text — TTS only, transcript untouched)
# ---------------------------------------------------------------------


def test_normalize_preserves_technical_terms() -> None:
    text = "Chal boss, ab main VS Code mein Python script chala ke GitHub pe push karunga. API aur Android ke liye C++ module ready hai."
    out = normalize_for_speech(text)
    for term in ("VS Code", "Python", "GitHub", "API", "Android"):
        assert term in out, f"technical term lost: {term}"
    print("PASS test_normalize_preserves_technical_terms")


def test_normalize_expands_symbols() -> None:
    out = normalize_for_speech("C++ program mein ₹70,000 ka laptop, error: File not found ⚠️")
    assert "C++" not in out or "c plus plus" in out.lower(), out
    assert "₹" not in out
    assert "⚠" not in out
    print("PASS test_normalize_expands_symbols")


def test_normalize_strips_markdown() -> None:
    out = normalize_for_speech("**Done** boss, file `main.py` ready hai.\n- point 1\n- point 2")
    assert "**" not in out and "`" not in out
    assert "Done" in out and "main.py" in out
    print("PASS test_normalize_strips_markdown")


# ---------------------------------------------------------------------
# playback + temp cleanup
# ---------------------------------------------------------------------


def test_playback_cleans_up_temp_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        played: list[Path] = []
        player = AudioPlayer(play_fn=lambda p: played.append(p))
        audio = Path(tmp) / "audio"
        audio.mkdir()
        file = audio / "test.mp3"
        file.write_bytes(b"ID3")
        assert player.play(file) is True
        assert played == [file]
        assert not file.exists(), "played temp audio must be deleted afterwards"
    print("PASS test_playback_cleans_up_temp_files")


def test_playback_failure_still_cleans_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        def broken(p: Path) -> None:
            raise RuntimeError("speaker on fire")

        player = AudioPlayer(play_fn=broken)
        audio = Path(tmp) / "audio"
        audio.mkdir()
        file = audio / "test.wav"
        file.write_bytes(b"RIFF")
        assert player.play(file) is False
        assert not file.exists(), "cleanup must happen even when playback fails"
    print("PASS test_playback_failure_still_cleans_up")


def test_playback_session_sweep_removes_stale() -> None:
    """Old files from a previous/crashed session get swept at startup.
    Fresh files (created minutes ago) must be left alone."""
    import os
    import time

    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio"
        audio.mkdir()
        stale = audio / "old-session.mp3"
        stale.write_bytes(b"x")
        old = time.time() - 7200
        os.utime(stale, (old, old))  # 2 hours old
        fresh = audio / "still-in-use.mp3"
        fresh.write_bytes(b"y")
        swept = AudioPlayer.sweep_stale(audio)
        assert swept == 1 and not stale.exists()
        assert fresh.exists()  # fresh audio must survive the sweep
    print("PASS test_playback_session_sweep_removes_stale")


def main() -> int:
    tests = [
        test_router_uses_first_available,
        test_router_falls_back_on_failure,
        test_router_skips_unavailable,
        test_router_all_failed_raises_with_details,
        test_router_order_windows_only,
        test_fish_request_format,
        test_fish_requires_real_key,
        test_fish_http_error_raises_ttser,
        test_fish_rate_limit_is_retryable,
        test_fish_network_error_is_retryable,
        test_fish_tiny_payload_rejected,
        test_windows_provider_reports_platform,
        test_windows_provider_synthesizes_on_windows,
        test_normalize_preserves_technical_terms,
        test_normalize_expands_symbols,
        test_normalize_strips_markdown,
        test_playback_cleans_up_temp_files,
        test_playback_failure_still_cleans_up,
        test_playback_session_sweep_removes_stale,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
