"""Build a second dataset from the 1M-point OSM sample, the way the authors' range data was made.

    python ml/prepare_osm_dataset.py SynopsisLake/osm21_pois_shuffle_sample.csv data/osm_global data/ml/osm_global
    python ml/prepare_osm_dataset.py australia_oceania_pois.csv data/oceania data/ml/oceania \
        --lon-cut 145 --lat-cut -30 --bbox 110 180 -60 25        # a regional extract (ml/extract_osm_pois.py)

The authors' data is four spatial tiles of one point set (files 0..3, tab separated lon, lat, timestamp; neighbouring
tiles overlap a little) and about 32,000 data-driven queries: centred on a random data point, one of four sizes that
are the extent divided by 84, 72, 60 and 48 per axis (12,696 / 10,243 / 5,722 / 3,370 queries), with the exact count
(`dataProcessing.BuildRangeQuery` / `BuildRangeAns`). This does the same for the OSM sample:

    <points_dir>/0.csv .. 3.csv      the tiles; the counts include the points repeated in the overlaps, as the raw scan does
    <dataset_dir>/queries-Ans.txt    minLon,maxLon,minLat,maxLat,exactCount (ml/SCHEMA.md); line number = qId
    <dataset_dir>/queries-meta.csv   qId,scale,width,height

Tiles: longitude cut at --lon-cut (default 10), latitude at --lat-cut (default 30), each tile reaching OVERLAP degrees
past the cut. --bbox minLon maxLon minLat maxLat first drops points outside it (for a region that wraps the
antimeridian, whose extent would otherwise be the whole globe and make every query huge).
Run the logger with `-dataDir <points_dir> -numFile 4 -srcRes 120,168,96,144 -queryFile <dataset_dir>/queries-Ans.txt`.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_authors_range import prepare  # noqa: E402

LON_CUT, LAT_CUT, OVERLAP = 10.0, 30.0, 3.0
DIVISORS = (84, 72, 60, 48)                       # query width and height = extent / divisor
COUNTS = (12696, 10243, 5722, 3370)               # queries per size in the authors' file
SEED = 0


def tiles(pts, lon_cut=LON_CUT, lat_cut=LAT_CUT):
    """Four overlapping tiles (SW, NW, SE, NE as in the authors' files: west first, then east)."""
    lon, lat = pts[:, 0], pts[:, 1]
    west, east = lon <= lon_cut + OVERLAP / 2, lon >= lon_cut - OVERLAP / 2
    south, north = lat <= lat_cut + OVERLAP / 2, lat >= lat_cut - OVERLAP / 2
    return [west & south, west & north, east & south, east & north]


def queries(pts, rng):
    sx = pts[np.argsort(pts[:, 0], kind="stable")]
    xs = sx[:, 0]
    ext_lon, ext_lat = pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()
    rows = []
    for k, n in zip(DIVISORS, COUNTS):
        w, h = ext_lon / k, ext_lat / k
        for cx, cy in pts[rng.integers(0, len(pts), n)]:
            x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
            lo, hi = np.searchsorted(xs, [x0, x1], side="left")[0], np.searchsorted(xs, x1, side="right")
            c = sx[lo:hi, 1]
            rows.append((x0, x1, y0, y1, float(((c >= y0) & (c <= y1)).sum())))
    return pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main(src, points_dir, dataset_dir, lon_cut=LON_CUT, lat_cut=LAT_CUT, bbox=None):
    pts = pd.read_csv(src, sep="\t", header=None).to_numpy()
    if bbox:
        x0, x1, y0, y1 = bbox
        keep = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        print(f"bbox {bbox}: kept {int(keep.sum()):,} of {len(pts):,} points")
        pts = pts[keep]
    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    masks = tiles(pts, lon_cut, lat_cut)
    for i, m in enumerate(masks):
        pd.DataFrame(pts[m]).to_csv(os.path.join(points_dir, f"{i}.csv"), sep="\t", header=False, index=False,
                                    float_format="%.7f", lineterminator="\n")
    allp = np.vstack([pts[m] for m in masks])         # what a scan of the four files sees (overlaps counted twice)
    print("tile sizes:", [int(m.sum()) for m in masks], "of", len(pts), "points; uncovered:",
          int((~np.logical_or.reduce(masks)).sum()))
    q = queries(allp[:, :2], np.random.default_rng(SEED))
    qfile = os.path.join(dataset_dir, "_queries.csv")
    q.to_csv(qfile, header=False, index=False)
    meta = prepare(qfile, dataset_dir)
    os.remove(qfile)
    print(f"{len(meta):,} queries -> {dataset_dir}; per scale {meta.scale.value_counts().sort_index().to_dict()}; "
          f"counts: median {int(q[4].median())}, with >= 10 points {int((q[4] >= 10).sum()):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("points_dir")
    ap.add_argument("dataset_dir")
    ap.add_argument("--lon-cut", type=float, default=LON_CUT)
    ap.add_argument("--lat-cut", type=float, default=LAT_CUT)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    a = ap.parse_args()
    main(a.src, a.points_dir, a.dataset_dir, a.lon_cut, a.lat_cut, a.bbox)
