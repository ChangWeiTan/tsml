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
package timeseriesweka.classifiers.distance_based;
import java.io.FileReader;
import weka_uea.classifiers.kNN;

import weka.core.*;

import weka.core.EuclideanDistance;
import timeseriesweka.elastic_distance_measures.DTW;

/* This class is a specialisation of kNN that can only be used with the efficient DTW distance
 * 
 * The reason for specialising is this class has the option of searching for the optimal window length
 * through a grid search of values.
 * 
 * By default this class does a search. 
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible window size, then sets the 
 * proportion to the one with the largest accuracy. This will be slow. Speed it up by
 * 
 * 1. Set the max window size to consider by calling
 * setMaxWindowSize(double r) where r is on range 0..1, with 1 being a full warp.
 * 
 * 2. Set the increment size 
 * setIncrementSize(int s) where s is on range 1...trainSetSize 
 * 
 * This is a basic brute force implementation, not optimised! There are probably ways of 
 * incrementally doing this. It could further be speeded up by using PAA to reduce the dimensionality first.
 * 
 */

public class DTW_kNN extends kNN {
    private boolean optimiseWindow=false;
    private double windowSize=0.1;
    private double maxWindowSize=1;
    private int incrementSize=10;
    private Instances train;
    private int trainSize;
    private int bestWarp;
    DTW dtw=new DTW();

//	DTW_DistanceEfficient dtw=new DTW_DistanceEfficient();
    public DTW_kNN(){
            super();
            dtw.setR(windowSize);
            setDistanceFunction(dtw);
            super.setKNN(1);
    }

    public void optimiseWindow(boolean b){ optimiseWindow=b;}
    public void setMaxR(double r){ maxWindowSize=r;}


    public DTW_kNN(int k){
            super(k);
            dtw.setR(windowSize);
            optimiseWindow=true;
            setDistanceFunction(dtw);
    }
    public void buildClassifier(Instances d){
        dist.setInstances(d);
        train=d;
        trainSize=d.numInstances();
        if(optimiseWindow){
            double maxR=0;
            double maxAcc=0;
/*Set the maximum warping window: Not this is all a bit mixed up. 
The window size in the r value is range 0..1, but the increments should be set by the 
data*/
            int dataLength=train.numAttributes()-1;
            int max=(int)(dataLength*maxWindowSize);
//			System.out.println(" MAX ="+max+" increment size ="+incrementSize);
            for(double i=0;i<max;i+=incrementSize){
                //Set r for current value
                dtw.setR(i/(double)dataLength);
                double acc=crossValidateAccuracy();
//				System.out.println("\ti="+i+" r="+(i/(double)dataLength)+" Acc = "+acc);
                if(acc>maxAcc){
                    maxR=i/dataLength;
                    maxAcc=acc;
//					System.out.println(" Best so far ="+maxR +" Warps ="+i+" has Accuracy ="+maxAcc);
                }
            }
            bestWarp=(int)(maxR*dataLength);
            dtw.setR(maxR);
//			System.out.println(" Best R = "+maxR+" Best Warp ="+bestWarp+" Size = "+(maxR*dataLength));
        }
// Then just use the normal kNN with the DTW distance. 
        super.buildClassifier(d);
    }
//Could do this for BER instead	
    private double crossValidateAccuracy(){
        double a=0,d=0, minDist;
        int nearest=0;
        Instance inst;
        for(int i=0;i<trainSize;i++){
//Find nearest to element i
            nearest=0;
            minDist=Double.MAX_VALUE;
            inst=train.instance(i);
            for(int j=0;j<trainSize;j++){
                if(i!=j){
                    d=dtw.distance(inst,train.instance(j),minDist);
                    if(d<minDist){
                            nearest=j;
                            minDist=d;
                    }
                }
            }
            //Measure accuracy for nearest to element i			
            if(inst.classValue()==train.instance(nearest).classValue())
                    a++;
        }
        return a/(double)trainSize;
    }

}
