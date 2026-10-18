package SynopsisLake;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import accumulator.BinaryFileSizeAccumulator;
import dataStructure.parquet.ParquetRow;
import dataStructure.parquet.ParquetRow2DCountHistogram;
import dataStructure.parquet.ParquetRow2DWavelets;
import dataStructure.parquet.ParquetRowGeometricHistogram;
import dataStructure.parquet.ParquetRowUniformSamples;
import loading.parquet.WriteBinaryFileToHDFS;

public class SynopsisOverheadExp {

	// Use pois dataset, to create 50 synopsis files for each type of synopsis
	int numFile = 50;
	int numOfRecordPerFile = 1000000;
//	int numFile = 10;
//	int numOfRecordPerFile = 30000;
	String regex = "\t";

	String HDFSURI = "";

	public SynopsisOverheadExp(String expName, String datasetPath, String HDFSURI, String synopsisFolder,
			int synopsisType, int synopsisResolution, boolean writeDataFile,
			int numFile, int numOfRecordPerFile) {
		this.numFile = numFile;
		this.numOfRecordPerFile = numOfRecordPerFile;
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		SparkConf sparkCond = new SparkConf().setAppName(expName);
		sparkCond.set("spark.driver.memory", "4g");
//		sparkCond.set("spark.driver.memory", "100g"); //TODO: add back in prod
//		sparkCond.set("spark.executor.memory", "100g");
		sparkCond.set("spark.memory.offHeap.enabled", "true");
		sparkCond.set("spark.memory.offHeap.size", "16g");
		sparkCond.set("spark.rpc.message.maxSize", "1024");
//		// Set Spark master to local if not already set
		if (!sparkCond.contains("spark.master"))
			sparkCond.setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(sparkCond);

		SparkSession spark = SparkSession.builder().config(sc.getConf()).getOrCreate();

		// Then, create an Accumulator of this type:
		BinaryFileSizeAccumulator<ParquetRow> sizeAccu = new BinaryFileSizeAccumulator<ParquetRow>();
		// Then, register it into spark context:
		sc.sc().register(sizeAccu, "sizeAccumulator");

		boolean hasSyno = true;
		int[] synopsisSizes = new int[2];
		synopsisSizes[0] = synopsisResolution;
		synopsisSizes[1] = synopsisResolution;
		
		double uniformSampleRate = 0;

		if (synopsisType == -1) {
			System.out.println("Not create synopses Files");
			hasSyno = false;
		} else if (synopsisType == 1) {
			// 2D count Hist
			System.out.println("Create Count Histogram Files");
		} else if (synopsisType == 3) {
			// 2D Wavelet
			System.out.println("Create Wavelets Files");
		} else if (synopsisType == 4) {
			// Geometric Hist
			System.out.println("Create Geometric Histogram Files");
		} else if (synopsisType == 6) {
			// Uniform Samples
			System.out.println("Create Uniform Samples Files");
			if (synopsisResolution == 128) {
				uniformSampleRate = 0.0006;
			} else if (synopsisResolution == 256) {
				uniformSampleRate = 0.001;
			} else if (synopsisResolution == 512) {
				uniformSampleRate = 0.005;
			}
		}

		ArrayList<ArrayList<String>> parallelLines = loadData(datasetPath);
		
		long start = System.nanoTime();
		JavaRDD<ArrayList<String>> lineRDD = sc.parallelize(parallelLines);

//		String outputFolder = HDFSURI + "/" + synopsisFolder;
		lineRDD.foreach(new WriteBinaryFileToHDFS(
				sizeAccu, hasSyno, HDFSURI, synopsisFolder, synopsisType, synopsisSizes,
				regex, uniformSampleRate, 0, writeDataFile));
		long end = System.nanoTime();
		long duration = end - start;
		
		if (hasSyno) {
			System.out.println("Create Synopses Parquet Files");
		} else {
			System.out.println("Create Parquet Files");
		}
		
		long startParquet = System.nanoTime();
		ArrayList<ParquetRow> list = sizeAccu.value();

		String parquetFilePath = HDFSURI + "/" + synopsisFolder + "/syno";
		writeParquetFile(list, spark, parquetFilePath, synopsisType);
		long endParquet = System.nanoTime();
		long durationParquet = endParquet - startParquet;
		
		System.out.println("Writing Parquet Files Finished");
		System.out.println("Create Parquet File Time = " + duration);
		System.out.println("Write Parquet File Time = " + durationParquet);
		System.out.println("Total time = " + (durationParquet + duration));
		
		sc.close();
	}
	
	private void writeParquetFile(ArrayList<ParquetRow> list, SparkSession spark, String parquetFilePath,
			int synopsisType) {
		
		switch (synopsisType) {
		case -1:
			Dataset<Row> df = spark.sqlContext().createDataFrame(list, ParquetRow.class);
			df.coalesce(1).write().parquet(parquetFilePath);
//			df.write().parquet(parquetFilePath);
			break;
		case 0:
			// 1D Count histogram
			break;
		case 1:
			// 2D Count histogram
			Dataset<Row> histDF = spark.sqlContext().createDataFrame(list, ParquetRow2DCountHistogram.class);
			histDF.coalesce(1).write().parquet(parquetFilePath);
//			histDF.write().parquet(parquetFilePath);
			break;
		case 3:
			// 2D Wavelet
			Dataset<Row> waveletDF = spark.sqlContext().createDataFrame(list, ParquetRow2DWavelets.class);
//			waveletDF.write().parquet(parquetFilePath);
			waveletDF.coalesce(1).write().parquet(parquetFilePath);
			break;
		case 4:
			// Geometric Histogram
			Dataset<Row> geohistDF = spark.sqlContext().createDataFrame(list, ParquetRowGeometricHistogram.class);
//			geohistDF.write().parquet(parquetFilePath);
			geohistDF.coalesce(1).write().parquet(parquetFilePath);
			break;
		case 6:
			// Uniform Samples
			Dataset<Row> usDF = spark.sqlContext().createDataFrame(list, ParquetRowUniformSamples.class);
			usDF.coalesce(1).write().parquet(parquetFilePath);
			break;
		}

	}

	private ArrayList<ArrayList<String>> loadData(String datasetPath) {
		System.out.println("Start loading files, numFile = " + numFile + ", numOfRecordPerFile = " + numOfRecordPerFile);
		ArrayList<ArrayList<String>> parallelLines = new ArrayList<ArrayList<String>>();

		File file = new File(datasetPath);

		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(file));
			ArrayList<String> lines = new ArrayList<String>();
			int fileId = 0;

			lines.add(String.valueOf(fileId));

			while (parallelLines.size() < numFile) {
				String line = reader.readLine();
				while (lines.size() < numOfRecordPerFile) {
					lines.add(line);
					line = reader.readLine();
				}
				if (lines.size() == numOfRecordPerFile) {
					parallelLines.add(lines);
					lines = new ArrayList<String>();
					fileId++;
					lines.add(String.valueOf(fileId));
					lines.add(line);
				}
			}

			reader.close();
			System.out.println("Load " + parallelLines.size() + " files.");
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return parallelLines;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String datasetPath = "/Users/xin_aurora/Downloads/Work/2019/"
				+ "UCR/Research/Spatial/sketches/exp/ICDE_loading/"
				+ "osm21_pois_shuffle_sample.csv";
//		int numFile = 100;
//		int numOfRecordPerFile = 1000000;
		int numFile = 10;
		int numOfRecordPerFile = 30000;

		String synopsisFolder = "/geoHist";
		int synopsisType = 4;
		int synopsisResolution = 128;

		String HDFSURI = "hdfs://localhost:9000/";
		String expName = "test";
		
		int writeDataFileInt = 0;
		boolean writeDataFile = false;
		
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-HDFSURI")) {
				HDFSURI = args[++i];
			} else if (args[i].equals("-datasetPath")) {
				datasetPath = args[++i];
			}  else if (args[i].equals("-expName")) {
				expName = args[++i];
			} else if (args[i].equals("-synopsisFolder")) {
				synopsisFolder = args[++i];
			} else if (args[i].equals("-synopsisType")) {
				synopsisType = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-synopsisResolution")) {
				synopsisResolution = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-numFile")) {
				numFile = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-numOfRecordPerFile")) {
				numOfRecordPerFile = Integer.parseInt(args[++i]);
			} else if (args[i].equals("-writeDataFileInt")) {
				writeDataFileInt = Integer.parseInt(args[++i]);
				if (writeDataFileInt == 0) {
					writeDataFile = true;
				} else {
					writeDataFile = false;
				}
			}
		}

		SynopsisOverheadExp loading = new SynopsisOverheadExp(expName, datasetPath, 
				HDFSURI, synopsisFolder, synopsisType, synopsisResolution, writeDataFile,
				numFile, numOfRecordPerFile);
	}

}
