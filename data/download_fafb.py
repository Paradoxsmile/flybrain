"""Check manually placed FlyWire (FAFB) files in data/raw/.

This script does NOT download anything. It reports which expected files are present and
whether they contain the required columns. See data/README.md.

Usage: python data/download_fafb.py
"""

from __future__ import annotations

import sys

from flybrain.loading import (
    CELL_TYPE_SRC_COL,
    CELL_TYPES_STEM,
    CONNECTIONS_STEM,
    EXTENSIONS,
    ID_COL,
    NEURONS_STEM,
    POST_COL,
    PRE_COL,
    RAW_DIR,
    WEIGHT_COL,
    find_raw_file,
    read_table,
)

# (stems, required columns, required file?)
EXPECTED = [
    (NEURONS_STEM, [ID_COL], True),
    (CONNECTIONS_STEM, [PRE_COL, POST_COL, WEIGHT_COL], True),
    (CELL_TYPES_STEM, [ID_COL, CELL_TYPE_SRC_COL], False),
]


def main() -> int:
    ok = True
    for stems, cols, required in EXPECTED:
        path = find_raw_file(RAW_DIR, stems)
        if path is None:
            names = " | ".join(stems)
            exts = "|".join(e.lstrip(".") for e in EXTENSIONS)
            if required:
                print(f"MISSING  {RAW_DIR}/({names}).({exts})")
                ok = False
            else:
                print(f"OPTIONAL {RAW_DIR}/({names}).({exts}) not found, no cell_type labels")
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
