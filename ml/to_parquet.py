"""Convert the CSVs the Java side writes (approx.csv from ApproxQueryExp, gated.csv, plan.csv) to Parquet.

    python ml/to_parquet.py out/approx.csv            # writes out/approx.parquet
"""
import sys

import pandas as pd

for path in sys.argv[1:]:
    out = path.rsplit(".", 1)[0] + ".parquet"
    pd.read_csv(path).to_parquet(out, index=False)
    print("wrote", out)
