"""Points of interest from an OpenStreetMap extract (.osm.pbf) as tab-separated `lon lat timestamp_ms`.

    python ml/extract_osm_pois.py ~/Downloads/geofabrik/australia-oceania-latest.osm.pbf australia_oceania_pois.csv

Needs `osmium` (brew install osmium-tool). A POI is a node tagged amenity, shop, tourism, leisure or name. The output
is the format of SynopsisLake/osm21_pois_shuffle_sample.csv and the input of ml/prepare_osm_dataset.py.
"""
import json
import os
import subprocess
import sys
import tempfile

TAGS = ["n/amenity", "n/shop", "n/tourism", "n/leisure", "n/name"]


def main(pbf, out):
    with tempfile.TemporaryDirectory() as tmp:
        poi, seq = os.path.join(tmp, "poi.osm.pbf"), os.path.join(tmp, "poi.geojsonseq")
        subprocess.run(["osmium", "tags-filter", pbf, *TAGS, "-o", poi], check=True)
        subprocess.run(["osmium", "export", poi, "-f", "geojsonseq", "--geometry-types=point", "-a", "timestamp",
                        "-o", seq], check=True)
        n = 0
        with open(seq) as f, open(out, "w") as w:
            for line in f:
                line = line.lstrip("\x1e").strip()
                if not line:
                    continue
                d = json.loads(line)
                lon, lat = d["geometry"]["coordinates"]
                w.write(f"{lon:.7f}\t{lat:.7f}\t{int(d['properties'].get('@timestamp', 0)) * 1000}\n")
                n += 1
    print(f"{n:,} points -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
