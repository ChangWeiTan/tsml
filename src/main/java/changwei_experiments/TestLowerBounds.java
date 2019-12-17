package changwei_experiments;

import experiments.data.DatasetLoading;
import timeseriesweka.classifiers.distance_based.FastEE.lowerBounds.LbKeogh;
import timeseriesweka.classifiers.distance_based.FastEE.utils.SequenceStatsCache;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.elastic_distance_measures.DTW;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Chang Wei
 */
public class TestLowerBounds {
    public static void main(String[] args) throws Exception {
        String trainPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TRAIN";
        String testPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TEST";
        Instances train = DatasetLoading.loadDataNullable(trainPath);
        Instances test = DatasetLoading.loadDataNullable(testPath);
        SequenceStatsCache trainCache = new SequenceStatsCache(train, test.numAttributes() - 1);
        SequenceStatsCache testCache = new SequenceStatsCache(test, test.numAttributes() - 1);

        // DTW
        int paramId = 0;
        double correct = 0;
        DTW1NN dtw1NN = new DTW1NN();
        dtw1NN.setParamsFromParamId(train, paramId);
//        int window = dtw1NN.getWindowSize(train.numAttributes() - 1);
        for (int i = 0; i < test.size(); i++) {
            Instance query = test.instance(i);
            double bsf = Double.POSITIVE_INFINITY;
            double pred = -1;
            for (int j = 0; j < train.size(); j++) {
                Instance reference = train.instance(j);
                double lbDist = dtw1NN.lowerBound(query, reference, i, j, bsf, testCache);
                double dtwDist = dtw1NN.distance(query, reference, bsf);
                System.out.println("LB: " + lbDist + ", DTW: " + dtwDist + ", BSF: " + bsf);
                if (lbDist < bsf) {
                    if (dtwDist < bsf) {
                        bsf = dtwDist;
                        pred = reference.classValue();
                    }
                }
            }
            if (query.classValue() == pred) {
                correct++;
            }
        }
        System.out.println("Acc: " + (correct / test.size()));
    }
}
