package changwei_experiments;

import experiments.data.DatasetLoading;
import timeseriesweka.classifiers.distance_based.FastEE.utils.SequenceStatsCache;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.DTW1NN;
import timeseriesweka.elastic_distance_measures.DTW;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Chang Wei
 */
public class CompareElasticDistances {
    public static void main(String[] args) throws Exception {
        String trainPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TRAIN";
        String testPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TEST";
        Instances train = DatasetLoading.loadDataNullable(trainPath);
        Instances test = DatasetLoading.loadDataNullable(testPath);
        SequenceStatsCache cache = new SequenceStatsCache(test, test.numAttributes() - 1);

        // DTW
        int paramId = 0;
        DTW1NN dtw1NN = new DTW1NN();
        dtw1NN.setParamsFromParamId(train, paramId);
        int window = dtw1NN.getWindowSize(train.numAttributes() - 1);
        double correctUEA = 0;
        double correctCW = 0;
        for (int i = 0; i < test.size(); i++) {
            Instance query = test.instance(i);
            double bsfUEA = Double.POSITIVE_INFINITY;
            double predUEA = -1;
            double bsfCW = Double.POSITIVE_INFINITY;
            double predCW = -1;
            for (int j = 0; j < train.size(); j++) {
                Instance reference = train.instance(j);

                double dtwUEA = dtw1NN.distance(query, reference, Double.POSITIVE_INFINITY);
                double dtwCW = DTW.distanceExt(query, reference, window).distance;

                if (dtwUEA < bsfUEA) {
                    bsfUEA = dtwUEA;
                    predUEA = reference.classValue();
                }

                if (dtwCW < bsfCW) {
                    bsfCW = dtwCW;
                    predCW = reference.classValue();
                }
                System.out.println("DTW distances: UEA=" + dtwUEA + ", CW=" + dtwCW);
            }
            if (query.classValue() == predUEA) {
                correctUEA++;
            }
            if (query.classValue() == predCW) {
                correctCW++;
            }
        }
        System.out.println("Acc UEA: " + (correctUEA / test.size()));
        System.out.println("Acc CW: " + (correctCW / test.size()));
    }
}
