# Learned error predictor: how to run it

SynopsisLake can answer a range count from a merged histogram (synopsis) and say how far to trust it. This folder
adds a trained LightGBM model that estimates that error, replacing the Skewness-Align score. A **gate** answers a
query from the synopsis when the estimated error is at or below a cutoff, and reads the raw data otherwise.

**Training happens in Databricks** (notebooks `ml/notebooks/`). The Java side only reads what training produces.
Formats: [SCHEMA.md](SCHEMA.md).

```
Databricks                              Java (SynopsisLake jar)
notebook 02 / ml/gate_artifacts.py  -->  model.txt, model.properties, cutoffs.csv
                                         |
ErrorPredictionLogExp  -> run folder (src_*.txt, arm_*.txt, rows.csv)
                                         |
ApproxQueryExp  (model + cutoffs + run folder + queries)  -> approx.csv: answer, error estimate, decision
```

## 1. Build

```
cd SynopsisLake && mvn -DskipTests package          # SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar
```

Every `java` command below needs the jar and commons-math3 on the classpath. From the repo root:

```
JAR=SynopsisLake/target/SynopsisLake-0.0.1-SNAPSHOT.jar
CM=$(find ~/.m2 -name "commons-math3-*.jar" | head -1)
CP=$JAR:$CM
```

## 2. What you need from training

Put these three files together (the names matter):

| File | From | |
|---|---|---|
| `model.txt` | `Booster.save_model` (notebook 02 logs it to MLflow) | any LightGBM regression model; its feature list is read from the file |
| `model.properties` | `ml/gate_artifacts.py` (notebook 02 logs it too) | `label.transform=log` if the model predicts `log(relErr)`, else `identity`; optional `gate.cutoff` |
| `cutoffs.csv` | `ml/gate_artifacts.py` | one cutoff per merge rate, per `eps` and `delta` |

For a model you already have, create the last two from the training data:

```
python ml/gate_artifacts.py out/ml/authors_range data/ml/authors_range model.txt out/gate --seed 0
```

Use the seed the model was trained with, so the calibration queries are not training queries. It also prints the
held-out coverage and violation, and writes `gate_report.csv` per merge rate.

## 3. Create the synopses (once per dataset)

The data files are `range_query/0.csv … 3.csv` (`gunzip -k range_query/2.csv.gz range_query/3.csv.gz` first).

```
java -cp $CP sigspatial25.ErrorPredictionLogExp \
  -dataDir range_query -queryFile range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv \
  -outDir out/ml/authors_range
```

This writes the synopses the next step reads (`src_<i>.txt`, `arm_<lonRes>.txt`, one per merge rate) and `rows.csv`.
Add `-modelPath model.txt` to also get a `learnedErr` column, and `-cutoffs cutoffs.csv -eps 0.10 -delta 0.05
[-rawFallback true]` to run the gate over every query and merge rate into `gated.csv`. `-rawFallback true` scans the
raw points for every rejected query, which is slow on the full dataset (about 4 ms per query here); leave it off to
only record the decisions.

## 4. Get approximate answers with estimated errors

```
java -cp $CP sigspatial25.ApproxQueryExp \
  -runDir out/ml/authors_range -queryFile queries.csv -outFile out/approx.csv \
  -modelPath model.txt -cutoffs cutoffs.csv -eps 0.10 -delta 0.05 -mergeRate 0.5
```

`queries.csv`: one query per line, `minLon,maxLon,minLat,maxLat[,exactCount]`. `-mergeRate` picks the nearest merged
synopsis. Without `-modelPath` the original Skewness-Align score is used (it needs its own `-cutoffs`).

`out/approx.csv` columns:

| column | meaning |
|---|---|
| `answer` | the count returned: the synopsis estimate, or the exact count if the gate sent it to raw data |
| `fromSynopsis` | `true` if the gate trusted the synopsis. `false` means read the raw data (use `-rawFallback true -dataDir range_query` to have the program do it) |
| `estRelError` | the model's estimated relative error of the synopsis answer; 0 for an exact raw answer; NaN for Skewness-Align |
| `errorScore`, `cutoff` | the score the gate compared with its cutoff |
| `gateNs`, `synNs`, `rawNs` | time to score, to answer from the synopsis, and to read raw data |
| `exact`, `relErr` | only when the query file has the exact count: the true error of `answer` |

Parquet: `python ml/to_parquet.py out/approx.csv` (needs pandas and pyarrow).

## 5. Choose the merge rate automatically

Instead of `-mergeRate`, let the planner pick the smallest synopsis that meets the accuracy target on queries with
exact counts:

```
java -cp $CP sigspatial25.ApproxQueryExp \
  -runDir out/ml/authors_range -queryFile queries.csv -outFile out/approx.csv \
  -modelPath model.txt -cutoffs cutoffs.csv -eps 0.10 -delta 0.05 \
  -plan true -planQueryFile plan_queries.csv -minCoverage 0.01
```

It evaluates every merge rate, writes `out/approx.csv.plan.csv` (coverage, violation and mean error per rate, and which
one was chosen) and answers `queries.csv` from the chosen rate. If no rate meets the target it prints that and exits
with status 2, or answers everything from the raw data with `-rawFallback true`.

Use planning queries that were **not** used to calibrate `cutoffs.csv`; otherwise the observed violation is
optimistic. `-minCoverage` is the share of queries a synopsis must answer for a rate to count (a synopsis that
answers almost nothing saves nothing).

## 6. Check the Java side matches the model

```
java -cp $CP sigspatial25.ErrorPredictionLogExp -dataDir range_query -queryFile … -outDir out/ml/authors_range -modelPath model.txt
python ml/check_java_parity.py out/ml/authors_range data/ml/authors_range model.txt
```

It compares the Java scores with `Booster.predict` and fails if more than 0.5% of rows differ or the rank correlation
is below 0.9999. Features differ from numpy in the last bit, so a model that splits on rounding noise sends about 0.1%
of rows down another path; rank and AUC do not change. Run it once for every new model.

## Using it from Java

```java
SynopsisStore store = SynopsisStore.load("out/ml/authors_range");
ErrorPredictor predictor = PredictorFactory.predictor("model.txt", null);          // null = Skewness-Align
CutoffTable cutoffs = PredictorFactory.cutoffs("cutoffs.csv", null, 0.10, 0.05, predictor);
double rate = store.nearestMergeRate(0.5);
GatedRangeQuery q = new GatedRangeQuery(store.context(rate, predictor.needsBeta()), predictor,
        cutoffs.forMergeRate(rate), null);                                         // or a RawRangeCounter
ApproxAnswer a = q.answer(minLon, maxLon, minLat, maxLat);                          // a.answer, a.estimatedRelError, a.fromSynopsis
```

Classes are in `SynopsisLake/src/main/java/predictor/`. A model that names a feature `FeatureRegistry` does not know is
rejected at load, with the name; add features with `FeatureRegistry.register` (its value must match the training code).

## Troubleshooting

| Message | Cause |
|---|---|
| `no cutoff for merge rate X and no default cutoff given` | `cutoffs.csv` has no row for that rate (or for the `-eps`/`-delta` asked for). Pass `-cutoff` or set `gate.cutoff` |
| `model uses features with no Java implementation: …` | the model uses a feature `FeatureRegistry` lacks |
| `this predictor needs the merged histogram's beta distribution` | only from a hand-built `SynopsisContext` with a null beta; `SynopsisStore.context(rate, predictor.needsBeta())` handles it |
| `only single-output regression models are supported` | the model is multiclass, linear-tree, or not a regression objective |
| `needs -cutoffs, -cutoff, or gate.cutoff` | no cutoff was given for the gate |
| every query goes to raw data | the cutoff is `-inf`: the calibration found no cutoff meeting `delta` at that merge rate |

## Run the tests

```
cd SynopsisLake && mvn test
```
