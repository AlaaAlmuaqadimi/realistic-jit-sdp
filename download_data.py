"""Download the two public datasets used in the study into data/.

    python download_data.py

ApacheJIT      : https://github.com/hosseinkshvrz/apachejit (Keshavarz & Nagappan, MSR 2022)
Kamei et al.   : https://github.com/WeiFoo/RevisitUnsupervised (replication package of
                 Fu & Menzies, FSE 2017; data from Kamei et al., TSE 2013)
The SHA-256 hashes printed at the end must match those reported in the paper.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from jitsdp import config as C  # noqa: E402
from jitsdp.data import file_sha256  # noqa: E402

KAMEI_RAW = "https://raw.githubusercontent.com/WeiFoo/RevisitUnsupervised/master/jit/input/{}.csv"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists  {dest.relative_to(C.ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"get     {url}")
    urllib.request.urlretrieve(url, dest)


def main():
    fetch(C.DATA_URL, C.DATA_FILE)
    for p in C.KAMEI_PROJECTS:
        fetch(KAMEI_RAW.format(p), C.KAMEI_DIR / f"{p}.csv")
    print("ApacheJIT sha256:", file_sha256([C.DATA_FILE]))
    print("Kamei     sha256:", file_sha256([C.KAMEI_DIR / f"{p}.csv" for p in C.KAMEI_PROJECTS]))


if __name__ == "__main__":
    main()
