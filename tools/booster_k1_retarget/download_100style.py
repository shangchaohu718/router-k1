"""Resumable bounded-range download of the official 100STYLE BVH archive."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import argparse
import json
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1/downloads"
URL = "https://zenodo.org/records/8127870/files/100STYLE.zip?download=1"
SIZE = 1468515354
MD5 = "3cf627852fd8192024c04a8d0ef49583"
CHUNK = 8 * 1024 * 1024
PARTS = "100style_parts"


def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = ROOT / PARTS / f"{index:04d}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    for attempt in range(4):
        try:
            with requests.get(
                URL + f"&part={index}",
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(20, 60),
            ) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{SIZE}"
                ):
                    raise ValueError(
                        f"Unexpected range response: {response.status_code} {response.headers.get('Content-Range')}"
                    )
                with path.open("wb") as out:
                    received = 0
                    for block in response.iter_content(256 * 1024):
                        received += len(block)
                        if received > end - start + 1:
                            raise ValueError(
                                "Response exceeded requested range"
                            )
                        out.write(block)
            if path.stat().st_size != end - start + 1:
                raise ValueError("Incomplete range")
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    global URL, SIZE, MD5, PARTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mirror", choices=["zenodo", "official-drive"], default="zenodo"
    )
    args = parser.parse_args()
    if args.mirror == "official-drive":
        URL = "https://drive.usercontent.google.com/download?id=1-Wr0MkBEIk0SkByJaDXt0wUEZevYFBAk&export=download&confirm=t"
        SIZE = 1468168639
        MD5 = None  # Official author mirror differs in ZIP bytes; verify ZIP CRC below.
        PARTS = "100style_drive_parts"
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / PARTS).mkdir(exist_ok=True)
    target = ROOT / "100STYLE.zip"
    if target.exists():
        raise FileExistsError(target)
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, i) for i in range(count)]
        for done, future in enumerate(as_completed(jobs), 1):
            future.result()
            print(f"100STYLE ranges {done}/{count}", flush=True)
    digest = hashlib.md5()
    assembled = ROOT / "100STYLE.zip.assembling"
    with assembled.open("wb") as out:
        for i in range(count):
            with (ROOT / PARTS / f"{i:04d}.part").open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
    if assembled.stat().st_size != SIZE or (
        MD5 is not None and digest.hexdigest() != MD5
    ):
        raise ValueError("Official checksum mismatch; archive not published")
    with zipfile.ZipFile(assembled) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
    assembled.rename(target)
    (ROOT / "100STYLE.download.json").write_text(
        json.dumps(
            {
                "url": URL,
                "bytes": SIZE,
                "md5": digest.hexdigest(),
                "official_md5_verified": MD5 is not None,
                "zip_crc_verified": True,
                "mirror": args.mirror,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"COMPLETE {target} bytes={SIZE} md5={digest.hexdigest()}", flush=True
    )


if __name__ == "__main__":
    main()
