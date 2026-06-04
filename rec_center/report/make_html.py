"""Convert Rec_Center_Final_Report.md to HTML using pandoc."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).parent
MD_FILE = REPORT_DIR / "Rec_Center_Final_Report.md"
OUT_FILE = REPORT_DIR / "Rec_Center_Final_Report.html"


def main() -> None:
    if not MD_FILE.exists():
        raise FileNotFoundError(MD_FILE)
    cmd = [
        "pandoc",
        str(MD_FILE),
        "-o",
        str(OUT_FILE),
        "--standalone",
        "--metadata",
        "title=Cal Poly Rec Center Usage Prediction — Final Report",
    ]
    subprocess.run(cmd, check=True)
    print("HTML written to:", OUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print("pandoc failed:", exc, file=sys.stderr)
        sys.exit(exc.returncode)
