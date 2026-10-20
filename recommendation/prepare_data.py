"""Download and verify the original GroupLens MovieLens-1M archive."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def main():
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    archive = root / "ml-1m.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Downloaded archive checksum mismatch")
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Existing archive checksum mismatch")
    with zipfile.ZipFile(archive) as handle:
        for name in ["README", "ratings.dat", "movies.dat", "users.dat"]:
            relative = f"ml-1m/{name}"
            target = root / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(handle.read(relative))
    print(f"DATA_READY {root / 'ml-1m'}")


if __name__ == "__main__":
    main()
