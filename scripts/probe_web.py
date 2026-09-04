"""Playwright smoke probe for a local or deployed Auto Transcribe web app."""

from __future__ import annotations

import argparse
import io
import wave
from urllib.parse import urlparse

from playwright.sync_api import Route, expect, sync_playwright

MOCK_JOB_ID = "f" * 32


def silent_wav(seconds: int = 4, sample_rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * seconds * sample_rate)
    return buffer.getvalue()


def install_result_fixture(page) -> None:
    audio = silent_wav()
    base_path = f"/transcriptions/{MOCK_JOB_ID}"
    job = {
        "job_id": MOCK_JOB_ID,
        "state": "completed",
        "progress": 100,
        "source_name": "playwright-complex-demo.wav",
        "instruments": None,
        "generate_score": True,
        "created_at": "2026-09-04T00:00:00Z",
        "updated_at": "2026-09-04T00:00:01Z",
        "links": {
            "self": base_path,
            "audio": f"{base_path}/audio",
            "events": f"{base_path}/events",
            "piano_roll": f"{base_path}/piano-roll",
            "midi": f"{base_path}/midi",
            "score_pdf": f"{base_path}/score.pdf",
        },
        "result": {
            "audio_seconds": 4.0,
            "note_count": 3,
            "inference": {"seconds": 1.25, "estimated_gpu_cost_usd": 0.000278},
        },
    }
    roll = {
        "job_id": MOCK_JOB_ID,
        "duration": 4.0,
        "instruments": ["acoustic_piano", "drums"],
        "notes": [
            {"index": 1, "pitch": 60, "instrument": "acoustic_piano", "start": 0.2, "end": 1.1},
            {"index": 2, "pitch": 67, "instrument": "acoustic_piano", "start": 1.3, "end": 2.8},
            {"index": 3, "pitch": 36, "instrument": "drums", "start": 2.0, "end": 2.15},
        ],
    }

    def handle(route: Route) -> None:
        path = urlparse(route.request.url).path
        if path == base_path:
            route.fulfill(json=job)
            return
        if path == f"{base_path}/piano-roll":
            route.fulfill(json=roll)
            return
        if path == f"{base_path}/audio":
            headers = {
                "Accept-Ranges": "bytes",
                "Content-Type": "audio/wav",
            }
            requested_range = route.request.headers.get("range")
            if requested_range:
                start_text, end_text = requested_range.removeprefix("bytes=").split("-", 1)
                start = int(start_text or 0)
                end = min(int(end_text) if end_text else len(audio) - 1, len(audio) - 1)
                body = audio[start : end + 1]
                headers.update(
                    {
                        "Content-Range": f"bytes {start}-{end}/{len(audio)}",
                        "Content-Length": str(len(body)),
                    }
                )
                route.fulfill(status=206, headers=headers, body=body)
            else:
                headers["Content-Length"] = str(len(audio))
                route.fulfill(headers=headers, body=audio)
            return
        route.fulfill(status=404, body="not part of the browser fixture")

    page.route("**/transcriptions/**", handle)


def run_probe(base_url: str, *, headed: bool) -> None:
    base_url = base_url.rstrip("/")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base_url, wait_until="networkidle")
        expect(
            page.get_by_role("heading", name="Hear the structure inside the song.")
        ).to_be_visible()
        expect(page.get_by_role("link", name="How it works").first).to_be_visible()
        expect(page.get_by_text("shared $10 monthly compute pool")).to_be_visible()
        assert page.get_by_text("MusicXML").count() == 0

        page.get_by_role("link", name="How it works").first.click()
        expect(page).to_have_url(f"{base_url}/how-it-works")
        expect(
            page.get_by_role("heading", name="A recording becomes a musical timeline.")
        ).to_be_visible()
        expect(page.get_by_role("heading", name="Align notes to the beat grid")).to_be_visible()

        install_result_fixture(page)
        page.goto(f"{base_url}/jobs/{MOCK_JOB_ID}", wait_until="networkidle")
        expect(page.get_by_text("Transcription complete")).to_be_visible()
        expect(page.get_by_role("link", name="View score")).to_be_visible()
        expect(page.get_by_role("link", name="MIDI")).to_be_visible()
        assert page.get_by_text("MusicXML").count() == 0
        page.wait_for_function("document.querySelector('#source-audio').readyState >= 1")
        page.locator("#timeline").evaluate(
            "element => { element.value = '700'; element.dispatchEvent(new Event('input', { bubbles: true })); }"
        )
        page.wait_for_function("document.querySelector('#source-audio').currentTime > 2.5")
        current_time = page.locator("#source-audio").evaluate("element => element.currentTime")
        assert 2.5 < current_time < 3.2

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(f"{base_url}/how-it-works", wait_until="networkidle")
        expect(
            mobile.get_by_role("heading", name="A recording becomes a musical timeline.")
        ).to_be_visible()
        assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        browser.close()

    print(f"Playwright probe passed: {base_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    run_probe(args.base_url, headed=args.headed)


if __name__ == "__main__":
    main()
