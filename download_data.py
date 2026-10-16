"""Download the original MovieLens 1M archive from GroupLens; retain its README."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import urllib.request
import urllib.error
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CHECKSUM_URL = URL + ".md5"


def download(url, target):
    """Keep TLS verification enabled; Windows can resolve missing intermediates natively."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    except urllib.error.URLError:
        if os.name != "nt":
            raise
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        command = f"$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri {quote(url)} -OutFile {quote(target)}"
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ml-1m.zip"
    if not archive.exists():
        download(URL, archive)
    content = archive.read_bytes()
    checksum_file = destination / "ml-1m.zip.md5"
    if not checksum_file.exists():
        download(CHECKSUM_URL, checksum_file)
    official_checksum_text = checksum_file.read_text(encoding="ascii").strip()
    computed_md5 = hashlib.md5(content).hexdigest()
    if computed_md5 not in official_checksum_text.lower():
        raise ValueError("Archive MD5 differs from the official GroupLens checksum")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            target = (destination / entry.filename).resolve()
            if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(destination):
                raise ValueError("Unsafe archive member")
        bundle.extractall(destination)
    required = ["ratings.dat", "movies.dat", "users.dat", "README"]
    for name in required:
        if not (destination / "ml-1m" / name).is_file():
            raise ValueError(f"Missing dataset file: {name}")
    manifest = {"source_url": URL, "checksum_url": CHECKSUM_URL,
                "official_checksum_text": official_checksum_text,
                "archive_sha256": hashlib.sha256(content).hexdigest(), "archive_md5": computed_md5,
                "ratings_sha256": hashlib.sha256((destination / "ml-1m/ratings.dat").read_bytes()).hexdigest()}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("Read data/ml-1m/README before using or redistributing the dataset.")


if __name__ == "__main__":
    main()
