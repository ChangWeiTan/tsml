/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package timeseriesweka.classifiers.hybrids;

import experiments.data.DatasetLoading;
import timeseriesweka.classifiers.distance_based.ElasticEnsemble;
import java.util.ArrayList;
import java.util.Random;
import timeseriesweka.classifiers.AbstractClassifierWithTrainingInfo;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransform;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import utilities.ClassifierTools;
import weka_uea.classifiers.ensembles.CAWPE;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import timeseriesweka.filters.ACF;
import timeseriesweka.filters.PowerSpectrum;
import weka.core.TechnicalInformationHandler;
/**
 * NOTE: consider this code experimental. This is a first pass and may not be final; it has been informally tested but awaiting rigurous testing before being signed off.
 * Also note that file writing/reading from file is not currently supported (will be added soon)
 
 @article{bagnall15cote,
  title={Time-Series Classification with {COTE}: The Collective of Transformation-Based Ensembles},
  author={A. Bagnall and J. Lines and J. Hills and A. Bostrom},
  journal={{IEEE} Transactions on Knowledge and Data Engineering},
  volume={27},
  issue={9},
  pages={2522--2535},
  year={2015}
}

 
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class FlatCote extends AbstractClassifierWithTrainingInfo implements TechnicalInformationHandler{

      
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "A. Bagnall and J. Lines and J. Hills and A. Bostrom");
        result.setValue(TechnicalInformation.Field.TITLE, "Time-Series Classification with COTE: The Collective of Transformation-Based Ensembles");
        result.setValue(TechnicalInformation.Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
        result.setValue(TechnicalInformation.Field.VOLUME, "27");
        result.setValue(TechnicalInformation.Field.NUMBER, "9");
        
        result.setValue(TechnicalInformation.Field.PAGES, "2522-2535");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return result;
    }    
    

    
    
    // Flat-COTE includes 35 constituent classifiers:
    //  -   11 from the Elastic Ensemble
    //  -   8 from the Shapelet Transform Ensemble
    //  -   8 from CAWPE (ACF transformed)
    //  -   8 from CAWPE (PS transformed)
    
    private enum EnsembleType{EE,ST,ACF,PS};
    private Instances train;
    
    
    private ElasticEnsemble ee;
    private CAWPE st;
    private CAWPE acf;
    private CAWPE ps;
    
//    private ShapeletTransform shapeletTransform;
    private double[][] cvAccs;
    private double cvSum;
    
    private double[] weightByClass;
    
    @Override
    public void buildClassifier(Instances train) throws Exception{
        long startTime=System.currentTimeMillis();
        this.train = train;
        
        ee = new ElasticEnsemble();
        ee.buildClassifier(train);
        
        //ShapeletTransform shapeletTransform = ShapeletTransformFactory.createTransform(train);
        ShapeletTransform shapeletTransform = ShapeletTransformTimingUtilities.createTransformWithTimeLimit(train, 24); // now defaults to max of 24 hours
        shapeletTransform.supressOutput();
        st = new CAWPE();
        st.setTransform(shapeletTransform);
        st.buildClassifier(train);
        
        acf = new CAWPE();
        acf.setTransform(new ACF());
        acf.buildClassifier(train);

        ps = new CAWPE();
        ps.setTransform(new PowerSpectrum());
        ps.buildClassifier(train);
        
        cvAccs = new double[4][];
        cvAccs[0] = ee.getCVAccs();
        cvAccs[1] = st.getIndividualCvAccs();
        cvAccs[2] = acf.getIndividualCvAccs();
        cvAccs[3] = ps.getIndividualCvAccs();
        
        cvSum = 0;
        for(int e = 0; e < cvAccs.length;e++){
            for(int c = 0; c < cvAccs[e].length; c++){
                cvSum+=cvAccs[e][c];
            }
        }
        trainResults.setBuildTime(System.currentTimeMillis()-startTime);

    }
    
    @Override
    public double[] distributionForInstance(Instance test) throws Exception{
        weightByClass = null;
        classifyInstance(test);
        double[] dists = new double[weightByClass.length];
        for(int c = 0; c < weightByClass.length; c++){
            dists[c] = weightByClass[c]/this.cvSum;
        }
        return dists;
    }
    
    @Override
    public double classifyInstance(Instance test) throws Exception{
        
        double[][] preds = new double[4][];
        
        preds[0] = this.ee.classifyInstanceByConstituents(test);
        preds[1] = this.st.classifyInstanceByConstituents(test);
        preds[2] = this.acf.classifyInstanceByConstituents(test);
        preds[3] = this.ps.classifyInstanceByConstituents(test);
        
        weightByClass = new double[train.numClasses()];
        ArrayList<Double> bsfClassVals = new ArrayList<>();
        double bsfWeight = -1;
        
        for(int e = 0; e < preds.length; e++){
            for(int c = 0; c < preds[e].length; c++){
                weightByClass[(int)preds[e][c]]+=cvAccs[e][c];
//                System.out.print(preds[e][c]+",");
                if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfWeight = weightByClass[(int)preds[e][c]];
                    bsfClassVals = new ArrayList<>();
                    bsfClassVals.add(preds[e][c]);
                }else if(weightByClass[(int)preds[e][c]] > bsfWeight){
                    bsfClassVals.add(preds[e][c]);
                }
            }
        }
        
        if(bsfClassVals.size()>1){
            return bsfClassVals.get(new Random().nextInt(bsfClassVals.size()));
        }        
        return bsfClassVals.get(0);
    }
    
    public static void main(String[] args) throws Exception{
        
        FlatCote fc = new FlatCote();
        Instances train = DatasetLoading.loadDataNullable("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TRAIN");
        Instances test = DatasetLoading.loadDataNullable("C:/users/sjx07ngu/dropbox/tsc problems/ItalyPowerDemand/ItalyPowerDemand_TEST");
        fc.buildClassifier(train);
        
        int correct = 0;
        for(int i = 0; i < test.numInstances(); i++){
            if(fc.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
        }
        System.out.println("Acc");
        System.out.println(correct+"/"+test.numInstances());
        System.out.println((double)correct/test.numInstances());
        
                
    }
    
}
