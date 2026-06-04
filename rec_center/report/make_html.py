"""Convert Rec_Center_Final_Report.md to HTML using pandoc."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).parent
MD_FILE = REPORT_DIR / "Rec_Center_Final_Report.md"
OUT_FILE = REPORT_DIR / "Rec_Center_Final_Report.html"

AUTHORS = ("Tyler Roberts", "Matt Kennedy", "Peter Mazolewski")


def inject_authors(html: str) -> str:
    author_lines = "".join(f'  <p class="author">{name}</p>\n' for name in AUTHORS)
    author_css = """
    .author {
      margin: 0.15em 0;
      font-size: 0.95em;
      text-align: center;
    }
"""
    if author_css.strip() not in html:
        html = html.replace("</style>", author_css + "  </style>", 1)
    return html.replace("</h1>\n", f"</h1>\n{author_lines}", 1)


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
    html = OUT_FILE.read_text(encoding="utf-8")
    OUT_FILE.write_text(inject_authors(html), encoding="utf-8")
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
