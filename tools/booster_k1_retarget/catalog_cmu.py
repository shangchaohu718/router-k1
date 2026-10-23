"""Cache official CMU descriptions and per-trial FPS, without motion downloads."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2] / "data/holomotion_k1_v1"
RAW = ROOT / "raw/cmu/all_asfamc/subjects"


def fetch(subject):
    cache = ROOT / "catalog/cmu_pages" / f"{subject}.html"
    url = f"http://mocap.cs.cmu.edu/search.php?subjectnumber={int(subject)}"
    if cache.exists():
        html = cache.read_text()
    else:
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        html = response.text
        if len(html) > 2_000_000 or ".amc" not in html:
            raise ValueError(f"Unexpected CMU page for {subject}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(html)
    result = {}
    for row in BeautifulSoup(html, "html.parser").find_all("tr"):
        links = [a.get("href", "") for a in row.find_all("a")]
        amc = next((v for v in links if v.endswith(".amc")), None)
        if not amc:
            continue
        cells = [
            c.get_text(" ", strip=True)
            for c in row.find_all("td", recursive=False)
        ]
        if len(cells) < 9:
            continue
        fps = float(cells[-2]) if cells[-2].isdigit() else None
        result[Path(amc).stem] = {
            "description": cells[2],
            "fps": fps,
            "source_url": url,
            "transport": "official_public_http_read_only",
        }
    return result


def main():
    subjects = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    records, errors = {}, {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fetch, s): s for s in subjects}
        for future in as_completed(jobs):
            subject = jobs[future]
            try:
                records.update(future.result())
            except Exception as exc:
                errors[subject] = str(exc)
            print(
                f"CMU metadata {len(records)} clips, failed subjects={len(errors)}",
                flush=True,
            )
    local = {p.stem for p in RAW.rglob("*.amc")}
    output = {
        "clips": {k: v for k, v in records.items() if k in local},
        "missing": sorted(local - records.keys()),
        "fetch_errors": errors,
        "note": "Missing/unknown FPS must not silently default to 120 Hz.",
    }
    (RAW / "cmu_metadata.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "catalog/cmu_motion_metadata.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(
        f"Complete: {len(output['clips'])} labelled, {len(output['missing'])} missing"
    )


if __name__ == "__main__":
    main()
