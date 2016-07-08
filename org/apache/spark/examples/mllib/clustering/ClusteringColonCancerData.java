package org.apache.spark.examples.mllib.clustering;

import static java.nio.file.StandardOpenOption.CREATE;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

public abstract class ClusteringColonCancerData {
	
	protected void obtainClusters(){
		// Set application name
		String appName = "ClusteringExample";

		// Initialize Spark configuration & context
		SparkConf sparkConf = new SparkConf().setAppName(appName)
				.setMaster("local[1]").set("spark.executor.memory", "1g");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		String path = "hdfs://localhost:9000/user/konur/COLRECT.txt";
		
		// Read the data file and return it as RDD of strings
		JavaRDD<String> tempData = sc.textFile(path);		
		
		JavaRDD<Vector> data = tempData.map(mapFunction);
		data.cache();
		
		int numClusters = 5;
		int numIterations = 30;


		
		// Rely on concrete subclasses for this method.
		Tuple2<JavaRDD<Integer>,Vector[]> pair = dataCenters(data, numClusters, numIterations);
		JavaRDD<Integer> clusterIndexes = pair._1();
		Vector[] clusterCenters = pair._2();
		
		// Bring all data to driver node for displaying results.
		List<String> collectedTempData = tempData.collect();
		List<Integer> collectedClusterIndexes = clusterIndexes.collect();
		
		// Display the results
		displayResults(collectedTempData, collectedClusterIndexes, clusterCenters);
		sc.stop();
		sc.close();
	}

	@SuppressWarnings("serial")
	static Function<String, Vector> mapFunction = new Function<String, Vector>() {
		public Vector call(String s) {
			String[] sarray = s.split(" ");
			double[] values = new double[sarray.length - 1]; // Ignore 1st token, it is survival months and not needed here.
			for (int i = 0; i < sarray.length - 1; i++)
				values[i] = Double.parseDouble(sarray[i + 1]);
			return Vectors.dense(values);
		}
	};
	
	protected abstract Tuple2<JavaRDD<Integer>,Vector[]> dataCenters(JavaRDD<Vector> data, int numClusters,
			int numIterations);
	
	protected void displayResults(List<String> collectedTempData,
			List<Integer> collectedClusterIndexes, Vector[] clusterCenters) {
		
		// Total number of data points.
		System.out.println("\nTotal # patients: " + collectedClusterIndexes.size());

		// For each cluster center, this data structure will contain the corresponding survival months in an ArrayList<Integer>.
		// For each data point in the cluster, the corresponding survival months will be a distinct element in the ArrayList<Integer>. 
		Hashtable<Integer, ArrayList<Integer>> cl = new Hashtable<Integer, ArrayList<Integer>>();

		int j = 0;
		
		// This key to this data structure is the identifier for a cluster center, one of  0, 1, 2, ..., #clusters - 1. The
		// value is a data structure storing the number of times a unique pair of 'stage group' and 'regional nodes positive'
		// is encountered.
		Hashtable<Integer, Hashtable<java.util.Vector<Integer>, Integer>> clusteredPoints = 
				new Hashtable<Integer, Hashtable<java.util.Vector<Integer>, Integer>>();

		for (Integer i : collectedClusterIndexes) {
			// This ArrayList<Integer> stores individual survival months for each data point.
			ArrayList<Integer> srvMnths = cl.get(i);
			if (srvMnths == null) {
				srvMnths = new ArrayList<Integer>();
				cl.put(i, srvMnths);
			}
			
			// For a data point, get the corresponding survival months, 'stage group' and 'regional nodes positive'.
			String tempRow = collectedTempData.get(j++);
			StringTokenizer strTok = new StringTokenizer(tempRow);
			String survivalMonths = strTok.nextToken();
			String stage = strTok.nextToken();
			String regNodes = strTok.nextToken();

			srvMnths.add(Integer.parseInt(survivalMonths));
			
			// This data structure stores the number of times a unique pair of 'stage group' and 'regional nodes positive'
			// is encountered. The key is a vector with 2 elements: 'stage group' and 'regional nodes positive'.
			Hashtable<java.util.Vector<Integer>, Integer> dataPoints = clusteredPoints
					.get(i);
			
			if (dataPoints == null) {
				dataPoints = new Hashtable<java.util.Vector<Integer>, Integer>();
				clusteredPoints.put(i, dataPoints);
			}

			// Construct a vector consisting of a unique pair of 'stage group' and 'regional nodes positive'.
			java.util.Vector<Integer> pnt = new java.util.Vector<Integer>();
			pnt.add(Integer.parseInt(stage));
			pnt.add(Integer.parseInt(regNodes));
			
			// Have we encountered with that unique pair of 'stage group' and 'regional nodes positive' before?
			Integer numOccurences = dataPoints.get(pnt);
		
			// If answer is no, add it to in dataPoints.
			if (numOccurences == null) {
				dataPoints.put(pnt, 1);
			} 
			// If answer is yes, increment the # times we encountered with that particular pair of 'stage group' and 'regional nodes positive'.
			else {
				dataPoints.put(pnt, numOccurences + 1);
			}

		}

		// Display average survival months and # data points in each cluster.
		Enumeration<Integer> keys = cl.keys();
		while (keys.hasMoreElements()) {
			Integer i = keys.nextElement();
			System.out.println("\nCluster " + i);
			System.out.println("# points: " + cl.get(i).size());
			System.out.println("Average survival months: " + avg(cl.get(i)));
		}

		// For each cluster display distinct pair of 'stage group' and 'regional nodes positive' and how many times
		// they occurred.
		Enumeration<Integer> keysPoints = clusteredPoints.keys();
		while (keysPoints.hasMoreElements()) {
			Integer i = keysPoints.nextElement();
			System.out.println("\nCluster " + i + " points:");
			Hashtable<java.util.Vector<Integer>, Integer> dataPoints = clusteredPoints
					.get(i);
			Enumeration<java.util.Vector<Integer>> keyVectors = dataPoints
					.keys();
			while (keyVectors.hasMoreElements()) {
				java.util.Vector<Integer> pnt = keyVectors.nextElement();
				System.out.println("[ 'stage group': " + pnt.get(0) + ", 'regional nodes positive': " + pnt.get(1) + "]"
						+ "  repeated " + dataPoints.get(pnt) + " time(s). ]");
			}
		}

		// Generate a property file to be used by JCCKit for plotting
		// the clusters and corresponding data points.
		PropertyFileGenerator.generatePropertyFileForGraph(clusteredPoints, clusterCenters);
	}

	// Calculate the average survival months in a given cluster.
	private static double avg(ArrayList<Integer> in) {
		
		if (in == null || in.size() == 0) {
			return -1.;
		}

		double sum = 0.;
		for (Integer i : in) {
			sum += i;
		}
		return (sum / in.size());
	}
}

class PropertyFileGenerator {
	private static Path po = Paths
			.get("/Applications/spark-1.6.1-bin-hadoop2.6/konurExamplesEclipse/SparkExamples/data/clusterPoints.properties");
	private static String SPACE = " ";
	private static String NEWLINE = "\n";
	private static String[] Colors = { "0xFFC0CB", "0xFFFF00", "0x00FFFF",
			"0x00FF00", "0x0000FF", "0xF0E68C", "0xFFC0CB", "0xFFFF00",
			"0x00FFFF", "0x00FF00", "0x0000FF", "0xF0E68C" };

	protected static void generatePropertyFileForGraph(
			Hashtable<Integer, Hashtable<java.util.Vector<Integer>, Integer>> clusteredPoints,
			Vector[] clusterCenters) {

		try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(
				po, CREATE, java.nio.file.StandardOpenOption.TRUNCATE_EXISTING))) {

			int NUMCURVES = clusteredPoints.size();
			// Fixed section
			byte[] data = "plot/legendVisible = false\n".getBytes();
			out.write(data, 0, data.length);

			data = "plot/coordinateSystem/yAxis/axisLabel = Regional nodes positive\n"
					.getBytes();
			out.write(data, 0, data.length);

			data = "plot/coordinateSystem/xAxis/axisLabel = Stage group\n\n"
					.getBytes();
			out.write(data, 0, data.length);

			StringBuilder tmp = new StringBuilder("data/curves = ");
			for (int i = 1; i <= NUMCURVES; i++) {
				tmp.append("curve" + (NUMCURVES + 1 - i) + SPACE + "errors"
						+ (NUMCURVES + 1 - i) + SPACE);
			}

			tmp.append("curve" + (NUMCURVES + 1) + SPACE + "errors"
					+ (NUMCURVES + 1) + SPACE);

			tmp.append(NEWLINE).append(NEWLINE);
			data = tmp.toString().getBytes();
			out.write(data, 0, data.length);

			// Dynamic section
			Enumeration<Integer> keysPoints = clusteredPoints.keys();
			while (keysPoints.hasMoreElements()) {
				Integer i = keysPoints.nextElement();

				Hashtable<java.util.Vector<Integer>, Integer> dataPoints = clusteredPoints
						.get(i);
				data = generateFromCluster(new Integer(i + 1).toString(),
						dataPoints).getBytes();
				out.write(data, 0, data.length);

				data = generateZeros("data/errors" + (i + 1) + "/x = ", "",
						dataPoints.size()).toString().getBytes();
				out.write(data, 0, data.length);

				data = generateZeros("data/errors" + (i + 1) + "/y = ", "\n",
						dataPoints.size()).toString().getBytes();
				out.write(data, 0, data.length);

			}

			StringBuilder strBldr = new StringBuilder("data/curve"
					+ (NUMCURVES + 1) + "/x = ");
			for (int i = 0; i < clusterCenters.length; i++) {
				double[] clusterPoint = clusterCenters[i].toArray();
				strBldr.append(clusterPoint[0] + SPACE);
			}
			data = strBldr.append(NEWLINE).toString().getBytes();
			out.write(data, 0, data.length);

			strBldr = new StringBuilder("data/curve" + (NUMCURVES + 1)
					+ "/y = ");
			for (int i = 0; i < clusterCenters.length; i++) {
				double[] clusterPoint = clusterCenters[i].toArray();
				strBldr.append(clusterPoint[1] + SPACE);
			}
			data = strBldr.append(NEWLINE).append(NEWLINE).toString()
					.getBytes();
			out.write(data, 0, data.length);

			strBldr = new StringBuilder("data/errors" + (NUMCURVES + 1)
					+ "/x = ");
			for (int i = 0; i < clusterCenters.length; i++) {
				strBldr.append(0 + SPACE);
			}
			data = strBldr.append(NEWLINE).toString().getBytes();
			out.write(data, 0, data.length);

			strBldr = new StringBuilder("data/errors" + (NUMCURVES + 1)
					+ "/y = ");
			for (int i = 0; i < clusterCenters.length; i++) {
				strBldr.append(0 + SPACE);
			}
			data = strBldr.append(NEWLINE).append(NEWLINE).toString()
					.getBytes();
			out.write(data, 0, data.length);

			// Fixed section
			data = "background = 0xffffff\n".getBytes();
			out.write(data, 0, data.length);
			data = "".getBytes();
			out.write(data, 0, data.length);
			data = "defaultCoordinateSystem/ticLabelAttributes/fontSize = 0.03\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "defaultCoordinateSystem/axisLabelAttributes/fontSize = 0.04\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "defaultCoordinateSystem/axisLabelAttributes/fontStyle = bold\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "plot/coordinateSystem/xAxis/ = defaultCoordinateSystem/\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "plot/coordinateSystem/xAxis/minimum = 	0\n".getBytes();
			out.write(data, 0, data.length);
			data = "plot/coordinateSystem/yAxis/minimum = 	0\n".getBytes();
			out.write(data, 0, data.length);

			data = "plot/coordinateSystem/xAxis/maximum = 	80\n".getBytes();
			out.write(data, 0, data.length);
			data = "plot/coordinateSystem/yAxis/maximum = 	100\n".getBytes();
			out.write(data, 0, data.length);
			data = "plot/coordinateSystem/yAxis/ = defaultCoordinateSystem/\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "".getBytes();
			out.write(data, 0, data.length);
			data = "plot/initialHintForNextCurve/className = jcckit.plot.PositionHint\n"
					.getBytes();
			out.write(data, 0, data.length);
			data = "plot/initialHintForNextCurve/origin = 1 1\n".getBytes();
			out.write(data, 0, data.length);

			data = NEWLINE.getBytes();
			out.write(data, 0, data.length);

			// Dynamic section
			tmp = new StringBuilder("plot/curveFactory/definitions = ");
			for (int i = 1; i <= NUMCURVES + 1; i++) {
				tmp.append("cdef" + i + SPACE + "edef" + i + SPACE);
			}
			tmp.append(NEWLINE);
			data = tmp.toString().getBytes();
			out.write(data, 0, data.length);

			// Dynamic section
			tmp = new StringBuilder("");
			for (int i = 1; i <= NUMCURVES + 1; i++) {
				data = new String(
						"plot/curveFactory/cdef"
								+ i
								+ "/symbolFactory/className = jcckit.plot.ErrorBarFactory\n")
						.getBytes();
				out.write(data, 0, data.length);

				data = new String(
						"plot/curveFactory/cdef"
								+ i
								+ "/symbolFactory/symbolFactory/attributes/className = jcckit.graphic.ShapeAttributes\n")
						.getBytes();
				out.write(data, 0, data.length);

				if (i == NUMCURVES + 1) {
					data = new String(
							"plot/curveFactory/cdef"
									+ i
									+ "/symbolFactory/symbolFactory/className = jcckit.plot.CircleSymbolFactory\n")
							.getBytes();
					out.write(data, 0, data.length);

					data = new String(
							"plot/curveFactory/cdef"
									+ i
									+ "/symbolFactory/symbolFactory/attributes/lineThickness = 0.002\n")
							.getBytes();
					out.write(data, 0, data.length);
				} else {
					data = new String(
							"plot/curveFactory/cdef"
									+ i
									+ "/symbolFactory/symbolFactory/className = jcckit.plot.CircleSymbolFactory\n")
							.getBytes();
					out.write(data, 0, data.length);

					data = new String("plot/curveFactory/cdef" + i
							+ "/symbolFactory/symbolFactory/size = 0.005\n")
							.getBytes();
					out.write(data, 0, data.length);

					data = new String(
							"plot/curveFactory/cdef"
									+ i
									+ "/symbolFactory/symbolFactory/attributes/fillColor = "
									+ Colors[i - 1] + "\n").getBytes();
					out.write(data, 0, data.length);
				}

				data = new String("plot/curveFactory/cdef" + i
						+ "/withLine = false\n").getBytes();
				out.write(data, 0, data.length);

				data = NEWLINE.getBytes();
				out.write(data, 0, data.length);

				if (i == 1) {
					data = new String(
							"plot/curveFactory/edef"
									+ i
									+ "/symbolFactory/className = jcckit.plot.ErrorBarFactory\n")
							.getBytes();
					out.write(data, 0, data.length);

					data = new String(
							"plot/curveFactory/edef"
									+ i
									+ "/symbolFactory/attributes/className = jcckit.graphic.ShapeAttributes\n")
							.getBytes();
					out.write(data, 0, data.length);

					data = new String("plot/curveFactory/edef" + i
							+ "/softClipping = false\n").getBytes();
					out.write(data, 0, data.length);
				} else {
					data = new String("plot/curveFactory/edef" + i
							+ "/ = plot/curveFactory/edef1/\n").getBytes();
					out.write(data, 0, data.length);
				}
				data = new String("plot/curveFactory/edef" + i
						+ "/symbolFactory/attributes/fillColor = \n")
						.getBytes();
				out.write(data, 0, data.length);

				data = new String("plot/curveFactory/edef" + i
						+ "/symbolFactory/size = 0\n").getBytes();
				out.write(data, 0, data.length);

				data = new String("plot/curveFactory/edef" + i
						+ "/withLine = false\n").getBytes();
				out.write(data, 0, data.length);

				data = NEWLINE.getBytes();
				out.write(data, 0, data.length);

			}

		} catch (IOException exc) {
			System.err.println(exc);
		}

	}

	private static String generateFromCluster(String prefix,
			Hashtable<java.util.Vector<Integer>, Integer> dataPoints) {
		StringBuilder strBldr = new StringBuilder("data/curve" + prefix
				+ "/x = ");

		Enumeration<java.util.Vector<Integer>> keyVectors = dataPoints.keys();
		while (keyVectors.hasMoreElements()) {
			java.util.Vector<Integer> pnt = keyVectors.nextElement();
			strBldr.append(pnt.get(0)).append(SPACE);
		}

		strBldr.append(NEWLINE).append("data/curve" + prefix + "/y = ");

		keyVectors = dataPoints.keys();
		while (keyVectors.hasMoreElements()) {
			java.util.Vector<Integer> pnt = keyVectors.nextElement();
			strBldr.append(pnt.get(1)).append(SPACE);
		}

		strBldr.append(NEWLINE).append(NEWLINE);
		return strBldr.toString();
	}

	private static String generateZeros(String prefix, String suffix,
			int numZeros) {
		StringBuilder strBldr = new StringBuilder(prefix);
		for (int idx = 1; idx <= numZeros; ++idx) {
			strBldr.append(0);
			strBldr.append(SPACE);
		}
		strBldr.append(NEWLINE);
		strBldr.append(suffix);
		return strBldr.toString();
	}

}

