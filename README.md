# SynopsisLake
The is a temporary repository for SIGSPATIAL 2025 submission

## Personal model training and ablation

Run `ml/notebooks/06_model_v1.py` from your personal Databricks Git folder.
It compares five source-feature variants across three seeds, reusing a compatible completed seed 0 run
and training seeds 1 and 2. It keeps team inputs read-only and saves each run in a unique personal directory.
See [the setup and ablation guide](ml/MODEL_ABLATION.md).

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
