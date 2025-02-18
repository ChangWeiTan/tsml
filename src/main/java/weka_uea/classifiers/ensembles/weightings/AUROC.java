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
package weka_uea.classifiers.ensembles.weightings;

import weka_uea.classifiers.ensembles.EnsembleModule;

/**
 * Will call findMeanAuroc() on each module's results, therefore not necessary to call
 * it within HESCA/whatever ensemble
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AUROC extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public AUROC() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public AUROC(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(Math.pow(module.trainResults.findMeanAUROC(), power), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
    
}

