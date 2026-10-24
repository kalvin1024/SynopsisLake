"""Turn the authors' range-query file into the dataset folder read by features.build_table (ml/SCHEMA.md).

    python ml/prepare_authors_range.py range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv data/ml/authors_range

Input: one query per line, minLon,maxLon,minLat,maxLat,exactCount (no header).
Output folder:
    queries-Ans.txt   the same five columns, one line per query; line number = qId
    queries-meta.csv  qId,scale,width,height; scale 1 = smallest query size
"""
import os
import shutil
import sys

import pandas as pd


def prepare(src, out):
    """Write queries-Ans.txt and queries-meta.csv for src into out. Returns the meta table."""
    q = pd.read_csv(src, header=None, names=["minLon", "maxLon", "minLat", "maxLat", "gt"])
    if q.isna().any().any():
        raise ValueError(f"{src}: missing values")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(src, os.path.join(out, "queries-Ans.txt"))

    width, height = q.maxLon - q.minLon, q.maxLat - q.minLat
    # The generator uses a few fixed query sizes; rank them (rounded to absorb float noise).
    size = width.round(6)
    scale = size.map({w: i + 1 for i, w in enumerate(sorted(size.unique()))})
    meta = pd.DataFrame({"qId": range(len(q)), "scale": scale, "width": width, "height": height})
    meta.to_csv(os.path.join(out, "queries-meta.csv"), index=False)
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    m = prepare(sys.argv[1], sys.argv[2])
    print(f"{len(m)} queries -> {sys.argv[2]}; queries per scale: {m.scale.value_counts().sort_index().to_dict()}")
