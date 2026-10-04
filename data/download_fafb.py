"""Check manually placed FlyWire (FAFB) files in data/raw/.

This script does NOT download anything. It reports which expected files are present and
whether they contain the required columns. See data/README.md.

Usage: python data/download_fafb.py
"""

from __future__ import annotations

import sys

from flybrain.loading import (
    CONNECTIONS_STEM,
    ID_COL,
    NEURONS_STEM,
    POST_COL,
    PRE_COL,
    RAW_DIR,
    WEIGHT_COL,
    find_raw_file,
    read_table,
)

EXPECTED = {
    NEURONS_STEM: [ID_COL],
    CONNECTIONS_STEM: [PRE_COL, POST_COL, WEIGHT_COL],
}


def main() -> int:
    ok = True
    for stem, cols in EXPECTED.items():
        path = find_raw_file(RAW_DIR, stem)
        if path is None:
            print(f"MISSING  {RAW_DIR / stem}.(parquet|feather|csv|csv.gz)")
            ok = False
            continue
        header = read_table(path).head(0).columns
        missing = [c for c in cols if c not in header]
        if missing:
            print(f"BAD      {path.name}: missing columns {missing}")
            ok = False
        else:
            print(f"OK       {path.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
