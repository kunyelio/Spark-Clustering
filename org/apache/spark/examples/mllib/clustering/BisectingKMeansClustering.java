package org.apache.spark.examples.mllib.clustering;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.BisectingKMeans;
import org.apache.spark.mllib.clustering.BisectingKMeansModel;
import org.apache.spark.mllib.linalg.Vector;

import scala.Tuple2;

public class BisectingKMeansClustering extends ClusteringColonCancerData {
	public static void main(String[] args) {
		BisectingKMeansClustering bisectingKMeans = new BisectingKMeansClustering();
		bisectingKMeans.obtainClusters();
	}
	
	public Tuple2<JavaRDD<Integer>,Vector[]> dataCenters(JavaRDD<Vector> data, int numClusters,
			int numIterations){
	    BisectingKMeans bkm = new BisectingKMeans().setK(numClusters).setMaxIterations(numIterations);
	    BisectingKMeansModel clusters = bkm.run(data);

		// Evaluate clustering by computing WCSS
		double WCSS = clusters.computeCost(data.rdd());
		System.out.println("WCSS = " + WCSS);
		
		// Display cluster centers
		Vector[] clusterCenters = clusters.clusterCenters();
		for (int i = 0; i < clusterCenters.length; i++) {
			Vector clusterCenter = clusterCenters[i];
			double[] centerPoint = clusterCenter.toArray();
			System.out.println("Cluster Center " + i + ": [ 'stage group': " + centerPoint[0] + 
					", 'regional nodes positive': " + centerPoint[1] + " ]");
		}

		// The size of clusterIndexes is # patients. For a given patient, clusterIndexes points to
		// the corresponding cluster, as one of 0, 1, 2, ..., #clusters - 1. 
		JavaRDD<Integer> clusterIndexes = clusters.predict(data);

		Tuple2<JavaRDD<Integer>,Vector[]> results = new Tuple2<JavaRDD<Integer>,Vector[]>(
				clusterIndexes, clusterCenters);
		return results;
	}
}
