# ML error predictor: file formats

Two folders per dataset `<ds>` (for example `authors_range`). On Databricks they live in the
`synopsis_lake` volumes; locally the same layout is used under `out/ml/` and `data/ml/`.

| folder | Databricks | local | written by |
|---|---|---|---|
| run folder | `<volume>/output/ml/<ds>/` | `out/ml/<ds>/` | `sigspatial25.ErrorPredictionLogExp` |
| dataset folder | `<volume>/data/ml/<ds>/` | `data/ml/<ds>/` | `ml/prepare_authors_range.py` |

`features.build_table(run_dir, dataset_dir)` joins the two.

## Run folder (`ErrorPredictionLogExp -outDir`)

### `rows.csv`

One row per (method, merge rate, query). Header:

```
method,mergeRate,lonRes,latRes,qId,gt,est,relErr,predErr,betaAlpha,overlapSum,cellsTouched,srcSumEst
```

| column | meaning |
|---|---|
| `method` | `ARM` (the data-driven Align-Reshape-Merge histogram) |
| `mergeRate` | (lonRes x latRes) / (canonical lon x canonical lat); 1.0 at the canonical resolution |
| `lonRes`, `latRes` | buckets of the merged histogram; the first rate uses the canonical lon x lat, later rates are square |
| `qId` | 0-based line number of the query in `queries-Ans.txt` |
| `gt` | exact count (5th column of the query file) |
| `est` | `RangeQueryWithRatio` answer[0] on the merged histogram |
| `relErr` | \|est - gt\| / gt (the target) |
| `predErr` | answer[2]: SynopsisLake's Skewness-Align error estimate |
| `betaAlpha` | alpha (= beta) of the merged histogram's beta distribution |
| `overlapSum` | answer[3]: sum of query/cell overlap ratios (0 if the query misses the histogram) |
| `cellsTouched` | answer[4]: number of cells the query overlaps |
| `srcSumEst` | sum over the source histograms of their answer[0] (reading every source, no merge) |

The canonical resolution is the number of distinct source boundaries minus one per axis
(178 x 179 for sources 40, 56, 32, 48). Merge rates run from the canonical lon resolution down to
`-end` in steps of `-step` (defaults 42 and 8: 18 rates).

### `arm_<lonRes>.txt`, `src_<i>.txt`

Merged histogram per merge rate and the source histograms (`i` = source file id), comma separated:

```
line 1   merge rate (1.0 for sources)
line 2   lon boundaries (lonRes + 1 values)
line 3   lat boundaries (latRes + 1 values)
line 4+  one row of counts per lon bucket (latRes values each)
```

### `report/` (written by notebooks 02 and 04, or `train_v0_local.py`)

`v0_results.csv` (dataset, eps, score, auc, coverage, violation), `feature_importance.csv`
(feature, gain), `report.md`.

## Dataset folder (`prepare_authors_range.py`)

| file | contents |
|---|---|
| `queries-Ans.txt` | `minLon,maxLon,minLat,maxLat,exactCount`, no header; line number = `qId` |
| `queries-meta.csv` | `qId,scale,width,height`; `scale` ranks the distinct query widths, 1 = smallest |

## Logger flags

| flag | default | meaning |
|---|---|---|
| `-dataDir` | `../range_query/` | folder with `0.csv` ... `<numFile-1>.csv` (tab separated lon, lat, ...) |
| `-numFile` | 4 | number of source files |
| `-srcRes` | `40,56,32,48` | resolution of each source histogram |
| `-queryFile` | `../range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv` | queries with exact counts |
| `-outDir` | `../out/ml/authors_range/` | run folder |
| `-end`, `-step` | 42, 8 | last lon resolution and step |
| `-fixStride` | false | use `j * data[0].length + k` instead of `j * data.length + k` for the beta densities |
| `-reproduceAuthors` | false | reload the sources at every merge rate like `DataDrivenRangeQueryMergingRateExp`; the global bounds then widen by 1e-6 per reload, which reproduces `DataDrivenQueryAvgRelError.txt` (`ourLH`) exactly |

Local run (from the repo root, after `gunzip -k range_query/2.csv.gz range_query/3.csv.gz` and
`mvn -DskipTests package` in `SynopsisLake/`):

```
java -cp SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar:<commons-math3.jar> sigspatial25.ErrorPredictionLogExp \
  -dataDir range_query -queryFile range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv -outDir out/ml/authors_range
python ml/prepare_authors_range.py range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv data/ml/authors_range
python ml/train_v0_local.py out/ml/authors_range data/ml/authors_range
```
