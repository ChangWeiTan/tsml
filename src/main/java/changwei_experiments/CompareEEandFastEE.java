package changwei_experiments;

import experiments.data.DatasetLoading;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import timeseriesweka.classifiers.distance_based.FastEE.utils.SequenceStatsCache;
import timeseriesweka.classifiers.distance_based.FastElasticEnsemble;
import timeseriesweka.classifiers.distance_based.elastic_ensemble.Efficient1NN;
import weka.core.Instances;

/**
 * @author Chang Wei
 */
public class CompareEEandFastEE {
    public static void main(String[] args) throws Exception {
        String trainPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TRAIN";
        String testPath = "C:/Users/cwtan/workspace/Dataset/TSC_Problems/ArrowHead/ArrowHead_TEST";
        Instances train = DatasetLoading.loadDataNullable(trainPath);
        Instances test = DatasetLoading.loadDataNullable(testPath);
        SequenceStatsCache cache = new SequenceStatsCache(test, test.numAttributes() - 1);

//        ElasticEnsemble ee = new ElasticEnsemble();
//        ee.buildClassifier(train);

        FastElasticEnsemble fastee = new FastElasticEnsemble();
        fastee.buildClassifier(train);

//        System.out.println("Train Acc for EE: " + ee.getTrainAcc());
        System.out.println("Train Acc for FastEE: " + fastee.getTrainAcc());

        // check the params
//        Efficient1NN[] eeClassifiers = ee.getClassifiers();
        Efficient1NN[] fasteeClassifiers = fastee.getClassifiers();

//        int countCorrect = 0;
//        for (int c = 0; c < eeClassifiers.length; c++) {
//            System.out.println("Parameters: " + eeClassifiers[c].toString() +
//                    ", EE=" + eeClassifiers[c].getParamInformationString() +
//                    ", FastEE=" + fasteeClassifiers[c].getParamInformationString());
//        }

//        int correctEE = 0;
        int correctFastEE = 0;
        for (int i = 0; i < test.numInstances(); i++) {
            double actual = test.instance(i).classValue();
//            double predEE = ee.classifyInstance(test.instance(i));
            double predFastEE = fastee.classifyInstance(test.instance(i), i, cache);

//            if (actual == predEE) {
//                correctEE++;
//            }

            if (actual == predFastEE) {
                correctFastEE++;
            }
        }
//        System.out.println("Test Acc for EE: " + (double) correctEE / test.numInstances());
//        System.out.println("Test Acc for EE -- correct: " + correctEE + "/" + test.numInstances());
        System.out.println("Test Acc for FastEE: " + (double) correctFastEE / test.numInstances());
        System.out.println("Test Acc for FastEE -- correct: " + correctFastEE + "/" + test.numInstances());
    }
}
