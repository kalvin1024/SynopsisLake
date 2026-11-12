# SynopsisLake
The is a temporary repository for SIGSPATIAL 2025 submission

## Run loading experiment
To run the loading experiment:

Please first compile the project:
	
 mvn package

Then run:

spark-submit --class SynopsisLake.SynopsisOverheadExp ./target/SynopsisLake-0.0.1-SNAPSHOT.jar -HDFSURI hdfs://REPLACE_WITH_YOUR_HDFS_URI -datasetPath osm21_pois_shuffle_sample.csv -expName loadNoSyno -synopsisFolder countHistogram -synopsisType 1 -synopsisResolution 128 -numFile 10 -numOfRecordPerFile 30000

Or open the source code to run SynopsisOverheadExp.java

The path to the file: 

/src/main/java/SynopsisLake/SynopsisOverheadExp.java

## Run query performance experiment
The data and query: https://drive.google.com/drive/folders/1QxHoaMPPadQNXg8pZBb7jPxzGhvLuJDF?usp=sharing

The path to the file: 

/src/main/java/sigspatial25/*

## Learned error predictor

A trained LightGBM model can replace the Skewness-Align score to decide whether to answer from a merged synopsis or the raw data. See [ml/README.md](ml/README.md).

`sigspatial25.ApproxQueryExp -mode exact -queryFile Q -dataDir D -numFile N` skips the synopsis and answers every query exactly from the raw data (`-mode approx` and `-mode gated` select the other paths). Details in [ml/README.md](ml/README.md#5b-exact-execution-mode).

The learned model can also choose which histogram boundaries to merge, in place of ARM's reshape score (`ErrorPredictionLogExp -mergePolicy`, trained by `ml/merge_policy.py`; `merge_policy.py select` is a per-dataset guardrail that picks ARM, a simple rule or the model on validation queries); see [ml/README.md](ml/README.md#7-learned-merge-policy-replaces-arms-reshape-score).
