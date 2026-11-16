"""Capture representative UI states with Playwright."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "docs" / "images"
SERVER_HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((SERVER_HOST, 0))
        return int(sock.getsockname()[1])


def _capture(page: Page, filename: str) -> Path:
    path = IMAGE_DIR / filename
    page.screenshot(path=str(path), full_page=True)
    return path


def _make_gif(paths: list[Path]) -> None:
    frames = [Image.open(path).convert("RGB").resize((800, 600)) for path in paths]
    frames[0].save(
        IMAGE_DIR / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    for frame in frames:
        frame.close()


def main() -> None:
    """Start the app, capture UI states, and shut the server down."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.main:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    captures: list[Path] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1000, "height": 700}, device_scale_factor=1)
            url = f"http://{SERVER_HOST}:{port}/"
            for _ in range(30):
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=1000)
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                raise RuntimeError("The application did not become ready.")

            captures.append(_capture(page, "ui_empty.png"))
            page.locator("#ticket").fill(
                "My laptop is broken and crashes when I open the finance app."
            )
            page.locator("#classify").click()
            page.locator("#result").wait_for(state="visible")
            captures.append(_capture(page, "ui_result_light.png"))
            page.locator("#theme").click()
            captures.append(_capture(page, "ui_result_dark.png"))
            page.locator("#ticket").fill("x")
            page.locator("#classify").click()
            captures.append(_capture(page, "ui_error.png"))
            browser.close()
        _make_gif(captures)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()

