import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

TARGET = Path("python_Scripts") / "yolov3.weights"
MIN_EXPECTED_BYTES = 200_000_000

URLS = [
    "https://data.pjreddie.com/files/yolov3.weights",
    "https://pjreddie.com/media/files/yolov3.weights",
]


def download(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, open(target, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> int:
    target = TARGET
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"Weights already present: {target} ({target.stat().st_size} bytes)")
        return 0

    temp = target.with_suffix(".weights.part")
    if temp.exists():
        temp.unlink()

    last_error = None
    for url in URLS:
        try:
            print(f"Trying {url}")
            download(url, temp)
            size = temp.stat().st_size
            if size < MIN_EXPECTED_BYTES:
                raise RuntimeError(f"Downloaded file too small ({size} bytes)")
            temp.replace(target)
            print(f"Downloaded YOLO weights to {target} ({size} bytes)")
            return 0
        except Exception as exc:
            last_error = exc
            print(f"Failed from {url}: {exc}")
            if temp.exists():
                temp.unlink()

    print(f"ERROR: could not download YOLO weights. Last error: {last_error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
