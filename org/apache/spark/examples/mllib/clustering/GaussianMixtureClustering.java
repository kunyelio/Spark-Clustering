package org.apache.spark.examples.mllib.clustering;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian;

import scala.Tuple2;

public class GaussianMixtureClustering extends ClusteringColonCancerData {
	
	public static void main(String[] args) {
		GaussianMixtureClustering gaussianMixture = new GaussianMixtureClustering();
		gaussianMixture.obtainClusters();
	}
	
	public Tuple2<JavaRDD<Integer>, Vector[]> dataCenters(JavaRDD<Vector> data,
			int numClusters, int numIterations) {
		// Obtain model
		GaussianMixture gm = new GaussianMixture().setK(numClusters)
				.setMaxIterations(numIterations);
		GaussianMixtureModel clusters = gm.run(data.rdd());

		// Display cluster centers
		Vector[] clusterCenters = new Vector[clusters.gaussians().length];
		int i = 0;

		MultivariateGaussian[] gaussians = clusters.gaussians();
		for (MultivariateGaussian mg : gaussians) {
			Vector clusterCenter = mg.mu();
			clusterCenters[i] = clusterCenter;
			double[] centerPoint = clusterCenter.toArray();
			System.out.println("Cluster Center " + i + ": [ 'stage group': " + centerPoint[0] + 
					", 'regional nodes positive': " + centerPoint[1] + " ]");
			i++;
		}

		// The size of clusterIndexes is # patients. For a given patient, clusterIndexes gives
		// the corresponding cluster, as one of 0, 1, 2, ..., #clusters - 1. 
		JavaRDD<Integer> clusterIndexes = clusters.predict(data);

		Tuple2<JavaRDD<Integer>, Vector[]> results = new Tuple2<JavaRDD<Integer>, Vector[]>(
				clusterIndexes, clusterCenters);
		return results;

	}
}
