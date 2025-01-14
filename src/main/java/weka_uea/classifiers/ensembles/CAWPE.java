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
package weka_uea.classifiers.ensembles;

import experiments.CollateResults;
import experiments.Experiments;
import evaluation.MultipleClassifierEvaluation;
import weka_uea.classifiers.ensembles.weightings.TrainAcc;
import weka_uea.classifiers.ensembles.weightings.ModuleWeightingScheme;
import weka_uea.classifiers.ensembles.weightings.TrainAccByClass;
import weka_uea.classifiers.ensembles.voting.MajorityVote;
import weka_uea.classifiers.ensembles.voting.ModuleVotingScheme;

import java.io.File;
import java.util.Arrays;

import utilities.ClassifierTools;
import evaluation.evaluators.CrossValidationEvaluator;
import utilities.DebugPrinting;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import timeseriesweka.classifiers.SaveParameterInfo;
import utilities.StatisticalUtilities;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import weka_uea.classifiers.ensembles.EnsembleModule;
import weka_uea.classifiers.ensembles.voting.MajorityConfidence;
import timeseriesweka.filters.SAX;
import utilities.ErrorReport;
import static utilities.GenericTools.indexOfMax;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka_uea.classifiers.kNN;
import timeseriesweka.classifiers.TrainAccuracyEstimator;

/**
 * Can be constructed and will be ready for use from the default constructor like any other classifier.
 * Default settings are equivalent to the CAWPE in the paper.
 See exampleCAWPEUsage() for more detailed options on defining different component sets, ensemble schemes, and file handling


 For examples of file creation and results analysis for reproduction purposes, see
 buildCAWPEPaper_AllResultsForFigure3()


 CLASSIFICATION SETTINGS:
 Default setup is defined by setDefaultCAWPESettings(), i.e:
   Comps: SVML, MLP, NN, Logistic, C4.5
   Weight: TrainAcc(4) (train accuracies to the power 4)
   Vote: MajorityConfidence (summing probability distributions)

 For the original settings used in an older version of cote, call setOriginalCAWPESettings(), i.e:
   Comps: NN, SVML, SVMQ, C4.5, NB, bayesNet, RotF, RandF
   Weight: TrainAcc
   Vote: MajorityVote

 EXPERIMENTAL USAGE:
 By default will build/cv members normally, and perform no file reading/writing.
 To turn on file handling of any kind, call
          setResultsFileLocationParameters(...)
 1) Can build ensemble and classify from results files of its members, call
          setBuildIndividualsFromResultsFiles(true)
 2) If members built from scratch, can write the results files of the individuals with
          setWriteIndividualsTrainResultsFiles(true)
          and
          writeIndividualTestFiles(...) after testing is complete
 3) And can write the ensemble train/testing files with
         writeEnsembleTrainTestFiles(...) after testing is complete

 There are a bunch of little intricacies if you want to do stuff other than a bog standard run
 Best bet will be to email me for any specific usage questions.
 *
 * @author James Large (james.large@uea.ac.uk)
 *
 */

public class CAWPE extends AbstractClassifier implements SaveParameterInfo, DebugPrinting, TrainAccuracyEstimator, TechnicalInformationHandler {

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "J. Large, J. Lines and A. Bagnall");
        result.setValue(TechnicalInformation.Field.YEAR, "2017");
        result.setValue(TechnicalInformation.Field.TITLE, "The Heterogeneous Ensembles of Standard Classification Algorithms (HESCA): the Whole is Greater than the Sum of its Parts");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "arXiv");
        result.setValue(TechnicalInformation.Field.URL, "https://arxiv.org/abs/1710.09220");

        return result;
    }
    
    protected ModuleWeightingScheme weightingScheme = new TrainAcc(4);
    protected ModuleVotingScheme votingScheme = new MajorityConfidence();
    protected EnsembleModule[] modules;

    protected boolean setSeed = true;
    protected int seed = 0;

    protected SimpleBatchFilter transform;
    protected Instances trainInsts;

    //TrainAccuracyEstimate
    protected boolean writeEnsembleTrainingFile = false;

    protected boolean performEnsembleCV = true;
    protected CrossValidationEvaluator cv = null;
    protected ClassifierResults ensembleTrainResults = null;//data generated during buildclassifier if above = true
    protected ClassifierResults ensembleTestResults = null;//data generated during testing

    //saved after building so that it can be added to our test results, even if for some reason 
    //we're not building/writing train results
    protected long buildTime = -1; 
    
    //data info
    protected int numTrainInsts;
    protected int numAttributes;
    protected int numClasses;
    protected int testInstCounter = 0;
    protected int numTestInsts = -1;
    protected Instance prevTestInstance = null;

    //results file handling
    protected boolean readIndividualsResults = false;
    protected boolean writeIndividualsResults = false;

    protected boolean resultsFilesParametersInitialised;
    
    /**
     * An annoying comprimise to deal with base classfiers that dont produce dists 
     * while getting their train estimate. Off by default, shouldnt be turned on for 
     * mass-experiments, intended for cases where user knows that dists are missing
     * (for BOSS, in this case) but still just wants to get ensemble results anyway... 
     */
    protected boolean fillMissingDistsWithOneHotVectors; 
    
    /**
     * if readResultsFilesDirectories.length == 1, all classifier's results read from that one path
     * else, resultsPaths.length must equal classifiers.length, with each index aligning
     * to the path to read the classifier's results from.
     *
     * e.g to read 2 classifiers from one directory, and another 2 from 2 different directories:
     *
     *     Index |  Paths  | Classifier
     *     --------------------------
     *       0   |  pathA  |   c1
     *       1   |  pathA  |   c2
     *       2   |  pathB  |   c3
     *       3   |  pathC  |   c4
     *
     */
    protected String readResultsFilesDirectories[] = null;

    /**
     * if resultsWritePath is not set, will default to resultsPaths[0]
     * i.e, if only reading from one directory, will write back the chosen results
     * under the same directory. if reading from multiple directories but a particular
     * write path not set, will simply pick the first one given.
     */
    protected String writeResultsFilesDirectory = null;
    protected String ensembleIdentifier = "CAWPE";
    protected int resampleIdentifier;
    protected String datasetName;

    protected int numCVFolds = 10;

    public void setNumCVFolds(int i){
        numCVFolds = i;
    }

    public CAWPE() {
        this.ensembleIdentifier = "CAWPE";
        this.transform = null;
        this.setDefaultCAWPESettings();
    }

    public Classifier[] getClassifiers(){
        Classifier[] classifiers = new Classifier[modules.length];
        for (int i = 0; i < modules.length; i++)
            classifiers[i] = modules[i].getClassifier();
        return classifiers;
    }

    public void setClassifiersNamesForFileRead(String[] classifierNames) {
        setClassifiers(null,classifierNames,null);

    }

    /**
     * If building CAWPE from scratch, the minimum requirement for running is the
 classifiers array, the others could be left null.
     *
     * If building CAWPE from the results files of individuals, the minimum requirement for
 running is the classifierNames list.
     *
     * @param classifiers array of classifiers to use
     * @param classifierNames if null, will use the classifiers' class names by default
     * @param classifierParameters  if null, parameters of each classifier empty by default
     */
    public void setClassifiers(Classifier[] classifiers, String[] classifierNames, String[] classifierParameters) {
        if (classifiers == null) {
            classifiers = new Classifier[classifierNames.length];
            for (int i = 0; i < classifiers.length; i++)
                classifiers[i] = null;
        }

        if (classifierNames == null) {
            classifierNames = new String[classifiers.length];
            for (int i = 0; i < classifiers.length; i++)
                classifierNames[i] = classifiers[i].getClass().getSimpleName();
        }

        if (classifierParameters == null) {
            classifierParameters = new String[classifiers.length];
            for (int i = 0; i < classifiers.length; i++)
                classifierParameters[i] = "";
        }

        this.modules = new EnsembleModule[classifiers.length];
        for (int m = 0; m < modules.length; m++)
            modules[m] = new EnsembleModule(classifierNames[m], classifiers[m], classifierParameters[m]);
    }

    /**
     * Comps: NN, SVML, SVMQ, C4.5, NB, BN, RotF, RandF
     * Weight: TrainAcc
     * Vote: MajorityVote
     *
     * As used originally in ST_HESCA, COTE.
     */
    public final void setOriginalCAWPESettings(){
        this.weightingScheme = new TrainAcc();
        this.votingScheme = new MajorityVote();

        Classifier[] classifiers = new Classifier[8];
        String[] classifierNames = new String[8];

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[0] = k;
        classifierNames[0] = "NN";

        classifiers[1] = new NaiveBayes();
        classifierNames[1] = "NB";

        classifiers[2] = new J48();
        classifierNames[2] = "C45";

        SMO svml = new SMO();
        svml.turnChecksOff();
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        svml.setKernel(kl);
        if(setSeed)
            svml.setRandomSeed(seed);
        classifiers[3] = svml;
        classifierNames[3] = "SVML";

        SMO svmq =new SMO();
//Assumes no missing, all real valued and a discrete class variable
        svmq.turnChecksOff();
        PolyKernel kq = new PolyKernel();
        kq.setExponent(2);
        svmq.setKernel(kq);
        if(setSeed)
           svmq.setRandomSeed(seed);
        classifiers[4] =svmq;
        classifierNames[4] = "SVMQ";

        RandomForest r=new RandomForest();
        r.setNumTrees(500);
        if(setSeed)
           r.setSeed(seed);
        classifiers[5] = r;
        classifierNames[5] = "RandF";


        RotationForest rf=new RotationForest();
        rf.setNumIterations(50);
        if(setSeed)
           rf.setSeed(seed);
        classifiers[6] = rf;
        classifierNames[6] = "RotF";

        classifiers[7] = new BayesNet();
        classifierNames[7] = "bayesNet";

        setClassifiers(classifiers, classifierNames, null);
    }

    /**
     * Uses the 'basic UCI' set up:
     * Comps: SVML, MLP, NN, Logistic, C4.5
     * Weight: TrainAcc(4) (train accuracies to the power 4)
     * Vote: MajorityConfidence (summing probability distributions)
     */
    public final void setDefaultCAWPESettings(){
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();

        Classifier[] classifiers = new Classifier[5];
        String[] classifierNames = new String[5];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        smo.setKernel(kl);
        if (setSeed)
            smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVML";

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[1] = k;
        classifierNames[1] = "NN";

        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";

        classifiers[3] = new Logistic();
        classifierNames[3] = "Logistic";

        classifiers[4] = new MultilayerPerceptron();
        classifierNames[4] = "MLP";

        setClassifiers(classifiers, classifierNames, null);
    }

    /**
     * Uses the 'basic UCI' set up:
     * Comps: SVML, MLP, NN, Logistic, C4.5
     * Weight: TrainAcc(4) (train accuracies to the power 4)
     * Vote: MajorityConfidence (summing probability distributions)
     */
    public final void setDefaultCAWPESettings_NoLogistic(){
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();

        Classifier[] classifiers = new Classifier[4];
        String[] classifierNames = new String[4];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(1);
        smo.setKernel(kl);
        if (setSeed)
            smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVML";

        kNN k=new kNN(100);
        k.setCrossValidate(true);
        k.normalise(false);
        k.setDistanceFunction(new EuclideanDistance());
        classifiers[1] = k;
        classifierNames[1] = "NN";

        classifiers[2] = new J48();
        classifierNames[2] = "C4.5";

        classifiers[3] = new MultilayerPerceptron();
        classifierNames[3] = "MLP";

        setClassifiers(classifiers, classifierNames, null);
    }


    public final void setAdvancedCAWPESettings(){
        this.weightingScheme = new TrainAcc(4);
        this.votingScheme = new MajorityConfidence();

        Classifier[] classifiers = new Classifier[3];
        String[] classifierNames = new String[3];

        SMO smo = new SMO();
        smo.turnChecksOff();
        smo.setBuildLogisticModels(true);
        PolyKernel kl = new PolyKernel();
        kl.setExponent(2);
        smo.setKernel(kl);
        if (setSeed)
            smo.setRandomSeed(seed);
        classifiers[0] = smo;
        classifierNames[0] = "SVMQ";
        RandomForest rf= new RandomForest();
        rf.setNumTrees(500);
        classifiers[1] = rf;
        classifierNames[1] = "RandF";
        RotationForest rotf=new RotationForest();
        rotf.setNumIterations(200);
        classifiers[2] = rotf;
        classifierNames[2] = "RotF";

        setClassifiers(classifiers, classifierNames, null);
    }



    public void setPerformCV(boolean b) {
        performEnsembleCV = b;
    }

    public void setRandSeed(int seed){
        this.setSeed = true;
        this.seed = seed;
    }

    public static int findNumFolds(Instances train){
        return 10;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**CAWPE TRAIN**");
        
        long startTime = System.nanoTime();

        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("CAWPE.buildClassifier: more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }
        
        //transform data if specified
        if(this.transform==null){
            this.trainInsts = data;
//            this.trainInsts = new Instances(data);
        }else{
           transform.setInputFormat(data);
           this.trainInsts = Filter.useFilter(data,transform);
        }
          
        //init
        this.numTrainInsts = trainInsts.numInstances();
        this.numClasses = trainInsts.numClasses();
        this.numAttributes = trainInsts.numAttributes();

        //set up modules
        initialiseModules();
        
        //if modules' results are being read in from file, ignore the i/o overhead 
        //of loading the results, we'll sum the actual buildtimes of each module as 
        //reported in the files
        if (readIndividualsResults)
            startTime = System.nanoTime();
        
        //set up ensemble
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);

        buildTime = System.nanoTime() - startTime;
        if (readIndividualsResults) {
            //we need to sum the modules' reported build time as well as the weight
            //and voting definition time
            for (EnsembleModule module : modules)
                buildTime += module.testResults.getPredictionTimeInNanos(testInstCounter);
        }
        
        if(this.performEnsembleCV) {
            ensembleTrainResults = doEnsembleCV(data); //combine modules to find overall ensemble trainpreds

            if (writeEnsembleTrainingFile)
//                writeResultsFile(ensembleIdentifier, getParameters(), ensembleTrainResults, "train");
                writeEnsembleTrainAccuracyEstimateResultsFile();
        }
        else {
            ensembleTrainResults = new ClassifierResults();
            ensembleTrainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        }
        
        //HACK FOR CAWPE_EXTENSION PAPER: 
        //since experiments expects us to make a train results object 
        //and for us to record our build time, going to record it here instead of 
        //editting experiments to record the buildtime at that level
        
        //buildTime does not include the ensemble's cv in any case, only the work required to be ready for testing
        //time unit has been set in doEnsembleCV(data);
        ensembleTrainResults.turnOffZeroTimingsErrors();
        ensembleTrainResults.setBuildTime(buildTime);
        ensembleTrainResults.turnOnZeroTimingsErrors();

        this.testInstCounter = 0; //prep for start of testing
    }

    protected void initialiseModules() throws Exception {
        //prep cv
        if (willNeedToDoCV()) {
            //int numFolds = setNumberOfFolds(train); //through TrainAccuracyEstimate interface

            cv = new CrossValidationEvaluator();
            if (setSeed)
                cv.setSeed(seed);
            cv.setNumFolds(numCVFolds);
            cv.buildFolds(trainInsts);
        }

        //currently will only have file reading ON or OFF (not load some files, train the rest)
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load CAWPE modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        }
        else
            trainModules();

        
        for (int m = 0; m < modules.length; m++) { 
            //in case train results didnt have probability distributions, hack for old hive cote results tony todo clean
            modules[m].trainResults.setNumClasses(trainInsts.numClasses());
            
            if (fillMissingDistsWithOneHotVectors)
                modules[m].trainResults.populateMissingDists();
                        
            modules[m].trainResults.findAllStatsOnce();
        }
    }

    protected boolean willNeedToDoCV() {
        //if we'll need ensemble cv, we want the cv fold info
        if (performEnsembleCV)
            return true;

        //or if any of the modules dont have cv data already
        for (EnsembleModule m : modules)
            if (!(m.getClassifier() instanceof TrainAccuracyEstimator))
                return true;

        return false;
    }

    protected void trainModules() throws Exception {

        for (EnsembleModule module : modules) {
            if (module.getClassifier() instanceof TrainAccuracyEstimator) {
                module.getClassifier().buildClassifier(trainInsts);

                //these train results should also include the buildtime
                module.trainResults = ((TrainAccuracyEstimator)module.getClassifier()).getTrainResults();
                module.trainResults.finaliseResults();
                
                if (writeIndividualsResults) { //if we're doing trainFold# file writing
                    String params = module.getParameters();
                    if (module.getClassifier() instanceof SaveParameterInfo)
                        params = ((SaveParameterInfo)module.getClassifier()).getParameters();
                    writeResultsFile(module.getModuleName(), params, module.trainResults, "train"); //write results out
                    printlnDebug(module.getModuleName() + " writing train file data gotten through TrainAccuracyEstimate...");
                }
            }
            else {
                printlnDebug(module.getModuleName() + " performing cv...");
                module.trainResults = cv.crossValidateWithStats(module.getClassifier(), trainInsts);
                module.trainResults.finaliseResults();
                
                //assumption: classifiers that maintain a classifierResults object, which may be the same object that module.trainResults refers to,
                //and which this subsequent building of the final classifier would tamper with, would have been handled as an instanceof TrainAccuracyEstimate above
                long startTime = System.nanoTime();
                module.getClassifier().buildClassifier(trainInsts);
                module.trainResults.setBuildTime(System.nanoTime() - startTime);
                module.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);

                if (writeIndividualsResults) { //if we're doing trainFold# file writing
                    writeResultsFile(module.getModuleName(), module.getParameters(), module.trainResults, "train"); //write results out
                    printlnDebug(module.getModuleName() + " writing train file with full preds from scratch...");
                }
            }
        }
    }

    protected void loadModules() throws Exception {
        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little
        ErrorReport errors = new ErrorReport("Errors while loading modules from file. Directories given: " + Arrays.toString(readResultsFilesDirectories));

        //for each module
        for(int m = 0; m < this.modules.length; m++){
            String readResultsFilesDirectory = readResultsFilesDirectories.length == 1 ? readResultsFilesDirectories[0] : readResultsFilesDirectories[m];

            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false;

            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "train");
            if (moduleTrainResultsFile != null) {
                printlnDebug(modules[m].getModuleName() + " train loading... " + moduleTrainResultsFile.getAbsolutePath());

                modules[m].trainResults = new ClassifierResults(moduleTrainResultsFile.getAbsolutePath());
                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "test");
            if (moduleTestResultsFile != null) {
                //of course these results not actually used at all during training,
                //only loaded for future use when classifying with ensemble
                printlnDebug(modules[m].getModuleName() + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                modules[m].testResults = new ClassifierResults(moduleTestResultsFile.getAbsolutePath());

                numTestInsts = modules[m].testResults.numInstances();
                testResultsLoaded = true;
            }

            if (!trainResultsLoaded)
                errors.log("\nTRAIN results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            else if (needIndividualTrainPreds() && modules[m].trainResults.getProbabilityDistributions().isEmpty())
                errors.log("\nNo pred/distribution for instance data found in TRAIN results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");

            if (!testResultsLoaded)
                errors.log("\nTEST results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "' not found. ");
            else if (modules[m].testResults.numInstances()==0)
                errors.log("\nNo prediction data found in TEST results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + resampleIdentifier + "'. ");
        }

        errors.throwIfErrors();
    }

    protected boolean needIndividualTrainPreds() {
        return performEnsembleCV || weightingScheme.needTrainPreds || votingScheme.needTrainPreds;
    }

    protected File findResultsFile(String readResultsFilesDirectory, String classifierName, String trainOrTest) {
        File file = new File(readResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"/"+trainOrTest+"Fold"+resampleIdentifier+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else
            return file;
    }

    //hack for handling train accuracy estimate. experiments is giving us the full path and filename
    //to write to, instead of just the folder and expecting us to fill in the +classifierName+"/Predictions/"+datasetName+filename;
    //when doing the interface overhaul, sort this stuff out.
    protected void writeEnsembleTrainAccuracyEstimateResultsFile() throws Exception {
        ensembleTrainResults.writeFullResultsToFile(writeResultsFilesDirectory);
    }
    
    protected void writeResultsFile(String classifierName, String parameters, ClassifierResults results, String trainOrTest) throws Exception {
        String fullPath = writeResultsFilesDirectory+classifierName+"/Predictions/"+datasetName;
        new File(fullPath).mkdirs();
        fullPath += "/" + trainOrTest + "Fold" + seed + ".csv";
        
        results.setClassifierName(classifierName);
        results.setDatasetName(datasetName);
        results.setFoldID(seed);
        results.setSplit(trainOrTest);
        
        results.setParas(parameters);
        results.writeFullResultsToFile(fullPath);
    }

    /**
     * must be called (this or the directory ARRAY overload) in order to build ensemble from results files or to write individual's
     * results files
     *
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String individualResultsFilesDirectory, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;

        this.readResultsFilesDirectories = new String [] {individualResultsFilesDirectory};
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }

    /**
     * must be called (this or the single directory string overload) in order to build ensemble from results files or to write individual's
     * results files
     *
     * exitOnFilesNotFound defines whether the ensemble will simply throw exception/exit if results files
     * arnt found, or will try to carry on (e.g train the classifiers normally)
     */
    public void setResultsFileLocationParameters(String[] individualResultsFilesDirectories, String datasetName, int resampleIdentifier) {
        resultsFilesParametersInitialised = true;

        this.readResultsFilesDirectories = individualResultsFilesDirectories;
        this.datasetName = datasetName;
        this.resampleIdentifier = resampleIdentifier;
    }

    /**
     * if writing results of individuals/ensemble, but want to define a specific folder to write to as opposed to defaulting to the (only or first)
     * reading location
     */
    public void setResultsFileWritingLocation(String writeResultsFilesDirectory) {
        this.writeResultsFilesDirectory = writeResultsFilesDirectory;
    }

    public void setBuildIndividualsFromResultsFiles(boolean b) {
        readIndividualsResults = b;
        if (b)
            writeIndividualsResults = false;
    }

    public void setWriteIndividualsTrainResultsFiles(boolean b) {
        writeIndividualsResults = b;
        if (b)
            readIndividualsResults = false;
    }

    protected ClassifierResults doEnsembleCV(Instances data) throws Exception {
        double[] accPerFold = new double[cv.getNumFolds()]; //for variance

        double actual, pred, correct = 0;
        double[] dist;

        ClassifierResults trainResults = new ClassifierResults(data.numClasses());
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        
        //for each train inst
        for (int fold = 0; fold < cv.getNumFolds(); fold++) {
            for (int i = 0; i < cv.getFoldIndices().get(fold).size(); i++) {
                int instIndex = cv.getFoldIndices().get(fold).get(i);

                long startTime = System.nanoTime();
                dist = votingScheme.distributionForTrainInstance(modules, instIndex);
                long predTime = System.nanoTime()- startTime; //time for ensemble to form vote
                for (EnsembleModule module : modules) //                 +time for each member's predictions
                    predTime += module.trainResults.getPredictionTime(instIndex);
                
                pred = utilities.GenericTools.indexOfMax(dist);
                actual = data.instance(instIndex).classValue();
                
                trainResults.turnOffZeroTimingsErrors();
                trainResults.addPrediction(actual, dist, pred, predTime, "");
                trainResults.turnOnZeroTimingsErrors();
                
                if(pred==actual) {
                    correct++;
                    accPerFold[fold]++;
                }
            }

            accPerFold[fold] /= cv.getFoldIndices().get(fold).size();
        }

        double acc = correct/numTrainInsts;
        double stddevOverFolds = StatisticalUtilities.standardDeviation(accPerFold, false, acc);

        trainResults.setClassifierName(ensembleIdentifier);
        if (datasetName == null || datasetName.equals(""))
            datasetName = data.relationName();
        trainResults.setDatasetName(datasetName);
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.setParas(getParameters());
        
        trainResults.stddev = stddevOverFolds;
        trainResults.finaliseResults();
        
        return trainResults;
    }

    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     *
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted
     */
    public void finaliseIndividualModuleTestResults(double[] testSetClassVals) throws Exception {
        for (EnsembleModule module : modules)
            module.testResults.finaliseResults(testSetClassVals); //converts arraylists to double[]s and preps for writing
    }

    /**
     * If building individuals from scratch, i.e not read results from files, call this
     * after testing is complete to build each module's testResults (accessible by module.testResults)
     *
     * This will be done internally anyway if writeIndividualTestFiles(...) is called, this method
     * is made public only so that results can be accessed from memory during the same run if wanted
     */
    public void finaliseEnsembleTestResults(double[] testSetClassVals) throws Exception {
        this.ensembleTestResults.finaliseResults(testSetClassVals);
    }

    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother,
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception
     */
    public void writeIndividualTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!writeIndividualsResults || !resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeIndividualTestFiles(), must have called setResultsFileLocationParameters(...) and setWriteIndividualsResultsFiles()");
            else
                return; //do nothing
        }

        finaliseIndividualModuleTestResults(testSetClassVals);

        for (EnsembleModule module : modules)
            writeResultsFile(module.getModuleName(), module.getParameters(), module.testResults, "test");
    }

    /**
     * @param throwExceptionOnFileParamsNotSetProperly added to make experimental code smoother,
     *  i.e if false, can leave the call to writeIndividualTestFiles(...) in even if building from file, and this
     *  function will just do nothing. else if actually intending to write test results files, pass true
     *  for exceptions to be thrown in case of genuine missing parameter settings
     * @throws Exception
     */
    public void writeEnsembleTrainTestFiles(double[] testSetClassVals, boolean throwExceptionOnFileParamsNotSetProperly) throws Exception {
        if (!resultsFilesParametersInitialised) {
            if (throwExceptionOnFileParamsNotSetProperly)
                throw new Exception("to call writeEnsembleTrainTestFiles(), must have called setResultsFileLocationParameters(...)");
            else
                return; //do nothing
        }

        if (ensembleTrainResults != null) //performed cv
            writeResultsFile(ensembleIdentifier, getParameters(), ensembleTrainResults, "train");

        this.ensembleTestResults.finaliseResults(testSetClassVals);
        writeResultsFile(ensembleIdentifier, getParameters(), ensembleTestResults, "test");
    }

    public EnsembleModule[] getModules() {
        return modules;
    }

    public CrossValidationEvaluator getCrossValidator() {
        return cv;
    }

    public String[] getClassifierNames() {
        String[] classifierNames = new String[modules.length];
        for (int m = 0; m < modules.length; m++)
            classifierNames[m] = modules[m].getModuleName();
        return classifierNames;
    }

    @Override
    public double[] getTrainPreds() {
        return ensembleTrainResults.getPredClassValsAsArray();
    }

    @Override
    public double getTrainAcc() {
        return ensembleTrainResults.getAcc();
    }

    public String getEnsembleIdentifier() {
        return ensembleIdentifier;
    }

    public void setEnsembleIdentifier(String ensembleIdentifier) {
        this.ensembleIdentifier = ensembleIdentifier;
    }

    public boolean isFillMissingDistsWithOneHotVectors() {
        return fillMissingDistsWithOneHotVectors;
    }

    public void setFillMissingDistsWithOneHotVectors(boolean fillMissingDistsWithOneHotVectors) {
        this.fillMissingDistsWithOneHotVectors = fillMissingDistsWithOneHotVectors;
    }
    
    public double[][] getPosteriorIndividualWeights() {
        double[][] weights = new double[modules.length][];
        for (int m = 0; m < modules.length; ++m)
            weights[m] = modules[m].posteriorWeights;

        return weights;
    }

    public ModuleVotingScheme getVotingScheme() {
        return votingScheme;
    }

    public void setVotingScheme(ModuleVotingScheme votingScheme) {
        this.votingScheme = votingScheme;
    }

    public ModuleWeightingScheme getWeightingScheme() {
        return weightingScheme;
    }

    public void setWeightingScheme(ModuleWeightingScheme weightingScheme) {
        this.weightingScheme = weightingScheme;
    }

    public double[] getIndividualCvAccs() {
        double [] accs = new double[modules.length];
        for (int i = 0; i < modules.length; i++)
            accs[i] = modules[i].trainResults.getAcc();

        return accs;
    }

//    public double[] getPriorIndividualWeights() {
//        return priorIndividualWeights;
//    }
//
//    public void setPriorIndividualWeights(double[] priorWeights) {
//        this.priorIndividualWeights = priorWeights;
//    }
//
//    private void setDefaultPriorWeights() {
//        priorIndividualWeights = new double[classifierNames.length];
//        for (int i = 0; i < priorIndividualWeights.length; i++)
//            priorIndividualWeights[i] = 1;
//    }


    public double[][] getIndividualCvPredictions() {
        double [][] preds = new double[modules.length][];
        for (int i = 0; i < modules.length; i++)
            preds[i] = modules[i].trainResults.getPredClassValsAsArray();
        return preds;
    }

    public SimpleBatchFilter getTransform(){
        return this.transform;
    }

    public void setTransform(SimpleBatchFilter transform){
        this.transform = transform;
    }

    @Override //TrainAccuracyEstimate
    public void writeTrainEstimatesToFile(String path) {
        performEnsembleCV=true;
        writeEnsembleTrainingFile=true;
        
        setResultsFileWritingLocation(path);
    }
    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        performEnsembleCV=setCV;
    }

    @Override
    public boolean findsTrainAccuracyEstimate(){ return performEnsembleCV;}

    @Override
    public ClassifierResults getTrainResults(){
        return ensembleTrainResults;
    }

    public ClassifierResults getTestResults(){
        return ensembleTestResults;
    }

    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();

//        if (ensembleTrainResults != null) //cv performed
//            out.append("BuildTime,").append(ensembleTrainResults.buildTime).append(",Trainacc,").append(ensembleTrainResults.acc).append(",");
//        else
//            out.append("BuildTime,").append("-1").append(",Trainacc,").append("-1").append(",");

        out.append(weightingScheme.toString()).append(",").append(votingScheme.toString()).append(",");

        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].getModuleName()).append("(").append(modules[m].priorWeight);
            for (int j = 0; j < modules[m].posteriorWeights.length; ++j)
                out.append("/").append(modules[m].posteriorWeights[j]);
            out.append("),");
        }

        return out.toString();
    }

//    public void readParameters(String paramLine) {
//        String[] classifiers = paramLine.split(",");
//
//        String[] classifierNames = new String[classifiers.length];
//        double[] priorWeights = new double[classifiers.length];
//        double[] postWeights = new double[classifiers.length];
//
//        for (int i = 0; i < classifiers.length; ++i) {
//            String[] parts = classifiers[i].split("(");
//            classifierNames[i] = parts[0];
//            String[] weights = parts[1].split("/");
//            priorWeights[i] = Integer.parseInt(weights[0]);
//            for (int j = 1; j < weights.length; ++j)
//                postWeights[j-1] = Integer.parseInt(weights[j]);
//        }
//
//    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        if (ensembleTestResults == null || (testInstCounter == 0 && prevTestInstance == null)) {//definitely the first call, not e.g the first inst being classified for the second time
            printlnDebug("\n**TEST**");

            ensembleTestResults = new ClassifierResults(numClasses);
            ensembleTestResults.setTimeUnit(TimeUnit.NANOSECONDS);
            ensembleTestResults.setBuildTime(buildTime);
        }

        if (readIndividualsResults && testInstCounter >= numTestInsts) //if no test files loaded, numTestInsts == -1
            throw new Exception("Received more test instances than expected, when loading test results files, found " + numTestInsts + " test cases");

        double[] dist;
        long startTime = System.nanoTime();
        long predTime;
        if (readIndividualsResults) { //have results loaded from file
            dist = votingScheme.distributionForTestInstance(modules, testInstCounter);
            predTime = System.nanoTime() - startTime; //time for ensemble to form vote
            for (EnsembleModule module : modules) //            +time for each member's predictions
                predTime += module.testResults.getPredictionTime(testInstCounter);
        }
        else {//need to classify them normally
            dist = votingScheme.distributionForInstance(modules, ins);
            predTime = System.nanoTime() - startTime;
        }
        
        ensembleTestResults.turnOffZeroTimingsErrors();
        ensembleTestResults.addPrediction(dist, indexOfMax(dist), predTime, "");
        ensembleTestResults.turnOnZeroTimingsErrors();
        
        if (prevTestInstance != instance)
            ++testInstCounter;
        prevTestInstance = instance;

        return dist;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        return utilities.GenericTools.indexOfMax(dist);
    }

    /**
     * @return the predictions of each individual module, i.e [0] = first module's vote, [1] = second...
     */
    public double[] classifyInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        double[] predsByClassifier = new double[modules.length];

        for(int i=0;i<modules.length;i++)
            predsByClassifier[i] = modules[i].getClassifier().classifyInstance(ins);

        return predsByClassifier;
    }

    /**
     * @return the distributions of each individual module, i.e [0] = first module's dist, [1] = second...
     */
    public double[][] distributionForInstanceByConstituents(Instance instance) throws Exception{
        Instance ins = instance;
        if(this.transform!=null){
            Instances rawContainer = new Instances(instance.dataset(),0);
            rawContainer.add(instance);
            transform.setInputFormat(rawContainer);
            Instances converted = Filter.useFilter(rawContainer,transform);
            ins = converted.instance(0);
        }

        double[][] distsByClassifier = new double[this.modules.length][];

        for(int i=0;i<modules.length;i++){
            distsByClassifier[i] = modules[i].getClassifier().distributionForInstance(ins);
        }

        return distsByClassifier;
    }

    /**
     * classifiers/cnames are optional, leave null for default classifiers
     *
     * todo: is a bodge job from previous code, before 'the big refactor'
     *
     * clean it up at some point and use the up to date methods
     */
    public static void buildAndWriteFullIndividualTrainTestResults(Instances defaultTrainPartition, Instances defaultTestPartition,
            String resultOutputDir, String datasetIdentifier, String ensembleIdentifier, int resampleIdentifier,
            Classifier[] classifiers, String[] cNames,
            SimpleBatchFilter transform, boolean setSeed, boolean resample, boolean writeEnsembleResults) throws Exception{


        Instances train = new Instances(defaultTrainPartition);
        Instances test = new Instances(defaultTestPartition);
        if(resample && resampleIdentifier >0){
            Instances[] temp = InstanceTools.resampleTrainAndTestInstances(train, test, resampleIdentifier);
            train = temp[0];
            test = temp[1];
        }

        CAWPE h = new CAWPE();
        if(classifiers != null)
            h.setClassifiers(classifiers, cNames, null);
        h.setTransform(transform);

        if (setSeed)
            h.setRandSeed(resampleIdentifier);

        h.setResultsFileLocationParameters(resultOutputDir, datasetIdentifier, resampleIdentifier); //dat hack... set after building/testing
        h.setWriteIndividualsTrainResultsFiles(true);

        if (writeEnsembleResults)
            h.setPerformCV(true);

        h.buildClassifier(train);

        for (Instance inst : test)
            h.distributionForInstance(inst); //will store results internally

        double[] classVals = test.attributeToDoubleArray(test.classIndex());
        h.writeIndividualTestFiles(classVals, true);
        if (writeEnsembleResults)
            h.writeEnsembleTrainTestFiles(classVals, true);
    }


    public static void exampleCAWPEUsage() throws Exception {
        String datasetName = "ItalyPowerDemand";

        Instances train = DatasetLoading.loadDataNullable("c:/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = DatasetLoading.loadDataNullable("c:/tsc problems/"+datasetName+"/"+datasetName+"_TEST");

        //Uses predefined default settings. This is the CAWPE classifier built on 'simple' components in the paper, equivalent to setDefaultCAWPESettings()
        CAWPE cawpe = new CAWPE();

        //Setting a transform (not used in CAWPE paper, mostly for COTE/HiveCOTE or particular applications)
        SimpleBatchFilter transform = new SAX();
        cawpe.setTransform(transform);
        cawpe.setTransform(null); //back to null for this example

        //Setting member classifiers
        Classifier[] classifiers = new Classifier[] { new kNN() };
        String [] names = new String[] { "NN" };
        String [] params = new String[] { "k=1" };
        cawpe.setClassifiers(classifiers, names, params); //see setClassifiers(...) javadoc

        //Setting ensemble schemes
        cawpe.setWeightingScheme(new TrainAccByClass()); //or set new methods
        cawpe.setVotingScheme(new MajorityVote()); //some voting schemes require dist for inst to be defined

        //Using predefined default settings. This is the CAWPE classifier in the paper, equivalent to default constructor
        cawpe.setDefaultCAWPESettings();

        int resampleID = 0;
        cawpe.setRandSeed(resampleID);

        //File handling
        cawpe.setResultsFileLocationParameters("CAWPETest/", datasetName, resampleID); //use this to set the location for any results file reading/writing

        cawpe.setBuildIndividualsFromResultsFiles(true); //turns on file reading, will read from location provided in setResultsFileLocationParameters(...)
        cawpe.setWriteIndividualsTrainResultsFiles(true); //include this to turn on file writing for individuals trainFold# files
        //can only have one of these (or neither) set to true at any one time (internally, setting one to true
        //will automatically set the other to false)

        //Then build/test as normal
        cawpe.buildClassifier(train);
        System.out.println(ClassifierTools.accuracy(test, cawpe));

        //Call these after testing is complete for fill writing of the individuals test files, and ensemble train AND test files.
        boolean throwExceptionOnFileParamsNotSetProperly = false;
        cawpe.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
        cawpe.writeEnsembleTrainTestFiles(test.attributeToDoubleArray(test.classIndex()), throwExceptionOnFileParamsNotSetProperly);
    }

    public String buildEnsembleReport(boolean printPreds, boolean builtFromFile) {
        StringBuilder sb = new StringBuilder();

        sb.append("CAWPE REPORT");
        sb.append("\nname: ").append(ensembleIdentifier);
        sb.append("\nmodules: ").append(modules[0].getModuleName());
        for (int i = 1; i < modules.length; i++)
            sb.append(",").append(modules[i].getModuleName());
        sb.append("\nweight scheme: ").append(weightingScheme);
        sb.append("\nvote scheme: ").append(votingScheme);
        sb.append("\ndataset: ").append(datasetName);
        sb.append("\nfold: ").append(resampleIdentifier);
        sb.append("\ntrain acc: ").append(ensembleTrainResults.getAcc());
        sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.getAcc() : "NA");

        int precision = 4;
        int numWidth = precision+2;
        int trainAccColWidth = 8;
        int priorWeightColWidth = 12;
        int postWeightColWidth = 12;

        String moduleHeaderFormatString = "\n\n%20s | %"+(Math.max(trainAccColWidth, numWidth))+"s | %"+(Math.max(priorWeightColWidth, numWidth))+"s | %"+(Math.max(postWeightColWidth, this.numClasses*(numWidth+2)))+"s";
        String moduleRowHeaderFormatString = "\n%20s | %"+trainAccColWidth+"."+precision+"f | %"+priorWeightColWidth+"."+precision+"f | %"+(Math.max(postWeightColWidth, this.numClasses*(precision+2)))+"s";

        sb.append(String.format(moduleHeaderFormatString, "modules", "trainacc", "priorweights", "postweights"));
        for (EnsembleModule module : modules) {
            String postweights = String.format("  %."+precision+"f", module.posteriorWeights[0]);
            for (int c = 1; c < this.numClasses; c++)
                postweights += String.format(", %."+precision+"f", module.posteriorWeights[c]);

            sb.append(String.format(moduleRowHeaderFormatString, module.getModuleName(), module.trainResults.getAcc(), module.priorWeight, postweights));
        }


        if (printPreds) {
            sb.append("\n\nensemble train preds: ");
            sb.append("\ntrain acc: ").append(ensembleTrainResults.getAcc());
            sb.append("\n");
            for(int i = 0; i < ensembleTrainResults.numInstances();i++)
                sb.append(buildEnsemblePredsLine(true, i)).append("\n");

            sb.append("\n\nensemble test preds: ");
            sb.append("\ntest acc: ").append(builtFromFile ? ensembleTestResults.getAcc() : "NA");
            sb.append("\n");
            for(int i = 0; i < ensembleTestResults.numInstances();i++)
                sb.append(buildEnsemblePredsLine(false, i)).append("\n");
        }

        return sb.toString();
    }

    /**
     * trueClassVal,predClassVal,[empty],dist1,...,distC,#indpreddist1,...,indpreddistC,#module1pred,...,moduleMpred
     * split on "#"
     * [0] = normal results file format (true class, pred class, distforinst)
     * [1] = number of individual unweighted votes per class
     * [2] = the unweighted prediction of each module
     */
    private String buildEnsemblePredsLine(boolean train, int index) {
        StringBuilder sb = new StringBuilder();

        if (train) //pred
            sb.append(modules[0].trainResults.getTrueClassValue(index)).append(",").append(ensembleTrainResults.getPredClassValue(index)).append(",");
        else
            sb.append(modules[0].testResults.getTrueClassValue(index)).append(",").append(ensembleTestResults.getPredClassValue(index)).append(",");

        if (train){ //dist
            double[] pred=ensembleTrainResults.getProbabilityDistribution(index);
            for (int j = 0; j < pred.length; j++)
                sb.append(",").append(pred[j]);
        }
        else{
            double[] pred=ensembleTestResults.getProbabilityDistribution(index);
            for (int j = 0; j < pred.length; j++)
                sb.append(",").append(pred[j]);
        }
        sb.append(",");


        double[] predDist = new double[numClasses]; //indpreddist
        for (int m = 0; m < modules.length; m++) {
            if (train)
                ++predDist[(int)modules[m].trainResults.getPredClassValue(index)];
            else
                ++predDist[(int)modules[m].testResults.getPredClassValue(index)];
        }
        for (int c = 0; c < numClasses; c++)
            sb.append(",").append(predDist[c]);
        sb.append(",");

        for (int m = 0; m < modules.length; m++) {
            if (train)
                sb.append(",").append(modules[m].trainResults.getPredClassValue(index));
            else
                sb.append(",").append(modules[m].testResults.getPredClassValue(index));
        }

        return sb.toString();
    }

    public static void testBuildingInds(int testID) throws Exception {
        System.out.println("testBuildingInds()");

        (new File("C:/JamesLPHD/CAWPETests"+testID+"/")).mkdirs();

        int numFolds = 5;

        for (int fold = 0; fold < numFolds; fold++) {
            String dataset = "breast-cancer-wisc-prog";
    //        String dataset = "ItalyPowerDemand";

            Instances all = DatasetLoading.loadDataNullable("C:/UCI Problems/"+dataset+"/"+dataset);
    //        Instances train = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
    //        Instances test = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TEST");

            Instances[] insts = InstanceTools.resampleInstances(all, fold, 0.5);
            Instances train = insts[0];
            Instances test = insts[1];

            CAWPE cawpe = new CAWPE();
            cawpe.setResultsFileLocationParameters("C:/JamesLPHD/CAWPETests"+testID+"/", dataset, fold);
            cawpe.setWriteIndividualsTrainResultsFiles(true);
            cawpe.setPerformCV(true); //now defaults to true
            cawpe.setRandSeed(fold);

            cawpe.buildClassifier(train);

            double acc = .0;
            for (Instance instance : test) {
                if (instance.classValue() == cawpe.classifyInstance(instance))
                    acc++;
            }
            acc/=test.numInstances();

            cawpe.writeIndividualTestFiles(test.attributeToDoubleArray(test.classIndex()), true);
            cawpe.writeEnsembleTrainTestFiles(test.attributeToDoubleArray(test.classIndex()), true);

            System.out.println("TrainAcc="+cawpe.getTrainResults().getAcc());
            System.out.println("TestAcc="+cawpe.getTestResults().getAcc());
        }
    }

    public static void testLoadingInds(int testID) throws Exception {
        System.out.println("testBuildingInds()");

        (new File("C:/JamesLPHD/CAWPETests"+testID+"/")).mkdirs();

        int numFolds = 5;

        for (int fold = 0; fold < numFolds; fold++) {
            String dataset = "breast-cancer-wisc-prog";
    //        String dataset = "ItalyPowerDemand";

            Instances all = DatasetLoading.loadDataNullable("C:/UCI Problems/"+dataset+"/"+dataset);
    //        Instances train = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TRAIN");
    //        Instances test = ClassifierTools.loadDataThrowable("C:/tsc problems/"+dataset+"/"+dataset+"_TEST");

            Instances[] insts = InstanceTools.resampleInstances(all, fold, 0.5);
            Instances train = insts[0];
            Instances test = insts[1];

            CAWPE cawpe = new CAWPE();
            cawpe.setResultsFileLocationParameters("C:/JamesLPHD/CAWPETests"+testID+"/", dataset, fold);
            cawpe.setBuildIndividualsFromResultsFiles(true);
            cawpe.setPerformCV(true); //now defaults to true
            cawpe.setRandSeed(fold);

            cawpe.buildClassifier(train);

            double acc = .0;
            for (Instance instance : test) {
                if (instance.classValue() == cawpe.classifyInstance(instance))
                    acc++;
            }
            acc/=test.numInstances();
            cawpe.finaliseEnsembleTestResults(test.attributeToDoubleArray(test.classIndex()));

            System.out.println("TrainAcc="+cawpe.getTrainResults().getAcc());
            System.out.println("TestAcc="+cawpe.getTestResults().getAcc());
        }
    }








    /**
     * This will build all the base classifier results
     *
     * @param dataHeaders e.g { "UCI", "UCR" }
     * @param dataPaths e.g { "C:/Data/UCI/", "C:/Data/UCR/" }
     * @param datasetNames for each datapath, a list of the dataset names located there to be used [archive][dsetnames]
     * @param classifiers the names of classifiers that can all be found in Experiments.setClassifier(...)
     * @param baseWritePath e.g { "C:/Results/" }
     */
    public static void buildCAWPEPaper_BuildClassifierResultsFiles(String baseWritePath, String[] dataHeaders, String[] dataPaths,
                                                            String[][] datasetNames, String[] classifiers, int numFolds) throws Exception {
        for (int archive = 0; archive < dataHeaders.length; archive++) {
            for (String classifier : classifiers) {
                System.out.println("\t" + classifier);

                for (String dset : datasetNames[archive]) {
                    System.out.println(dset);
                    for (int fold = 0; fold < numFolds; fold++) {
                          /*1: Problem path args[0]
                            2. Results path args[1]
                            3. booleanwWhether to CV to generate train files (true/false)
                            4. Classifier =args[3];
                            5. String problem=args[4];
                            6. int fold=Integer.parseInt(args[5])-1;
                        Optional:
                            7. boolean whether to checkpoint parameter search for applicable tuned classifiers (true/false)
                            8. integer for specific parameter search (0 indicates ignore this)
                            */
                        Experiments.main(new String[] { "-dp="+dataPaths[archive], "-rp="+baseWritePath+dataHeaders[archive]+"/", "-cn="+classifier, "-dn="+dset, "-f="+(fold+1), "-gtf=true"});
                    }
                }
            }
        }
    }

    /**
     * This method would build all the results files leading up to figure 3,
     * the heterogeneous ensemble comparison on the basic classifiers.
     *
     * It would take a long time to run, almost all of which is comprised of
     * building the base classifiers.
     *
     * The experiments and results presented in the paper were distributed on the HPC cluster at UEA,
     * this method is to demonstrate the experimental procedure and to provide a base to copy/edit for
     * full results reproduction of everything in the paper.
     *
     * There are also cases that can't be entirely captured neatly in a method like this, despite
     * my best efforts. For example, while we can call matlab code from here to build diagrams for
     * the analysis, the implementation of the DNN requires that to be run separately. Likewise, while
     * a lot of the legwork of analysis is done programmatically, the deeper exploratory analysis
     * cannot really be done automatically.
     *
     * Still, the idea of getting as close a possible to being able to reproduce the entirety
     * of a paper's results and statistics in a single function call is nice, especially for a
     * paper as extensive and empirically-driven as CAWPE's.
     *
     * For inquiries into specific details of reproduction, best bet is to email us
     * james.large@uea.ac.uk
     * anthony.bagnall@uea.ac.uk
     */
    public static void buildCAWPEPaper_AllResultsForFigure3() throws Exception {
        //init, edit the paths for local running ofc
        String[] dataHeaders = { "UCI", };
        String[] dataPaths = { "Z:/Data/UCIDelgado/", };
        String[][] datasets = { { "hayes-roth", "pittsburg-bridges-T-OR-D", "teaching", "wine" } };
        String writePathBase = "Z:/Results_7_2_19/CAWPEReproducabiltyTests/CAWPEReproducabiltyTest25/";
        String writePathResults =  writePathBase + "Results/";
        String writePathAnalysis =  writePathBase + "Analysis/";
        int numFolds = 5;
//        String[] dataHeaders = { "UCI", };
//        String[] dataPaths = { "Z:/Data/UCIDelgado/", };
//        String[][] datasets = { DataSets.UCIContinuousFileNames, };
//        String writePathBase = "Z:/Results_7_2_19/CAWPEReproducabiltyTest2/";
//        String writePathResults =  writePathBase + "Results/";
//        String writePathAnalysis =  writePathBase + "Analysis/";
//        int numFolds = 30;

        //build the base classifiers
        String[] baseClassifiers = { "NN", "C45", "MLP", "Logistic", "SVML" };
        buildCAWPEPaper_BuildClassifierResultsFiles(writePathResults, dataHeaders, dataPaths, datasets, baseClassifiers, numFolds);

        //build the ensembles
        String[] ensembleIDsInStorage = {
            "CAWPE_BasicClassifiers",
            "EnsembleSelection_BasicClassifiers",
            "SMLR_BasicClassifiers",
            "SMLRE_BasicClassifiers",
            "SMM5_BasicClassifiers",
            "PickBest_BasicClassifiers",
            "MajorityVote_BasicClassifiers",
            "WeightMajorityVote_BasicClassifiers",
            "RecallCombiner_BasicClassifiers",
            "NaiveBayesCombiner_BasicClassifiers"
        };

        String[] ensembleIDsOnFigures = {
            "CAWPE", "ES", "SMLR", "SMLRE", "SMM5",
            "PB", "MV", "WMV", "RC", "NBC"
        };

        Class[] ensembleClasses = {
            Class.forName("vector_classifiers.CAWPE"),
            Class.forName("vector_classifiers.EnsembleSelection"),
            Class.forName("vector_classifiers.stackers.SMLR"),
            Class.forName("vector_classifiers.stackers.SMLRE"),
            Class.forName("vector_classifiers.stackers.SMM5"),
            Class.forName("vector_classifiers.weightedvoters.CAWPE_PickBest"),
            Class.forName("vector_classifiers.weightedvoters.CAWPE_MajorityVote"),
            Class.forName("vector_classifiers.weightedvoters.CAWPE_WeightedMajorityVote"),
            Class.forName("vector_classifiers.weightedvoters.CAWPE_RecallCombiner"),
            Class.forName("vector_classifiers.weightedvoters.CAWPE_NaiveBayesCombiner"),
        };

        for (int ensemble = 0; ensemble < ensembleIDsInStorage.length; ensemble++)
            buildCAWPEPaper_BuildEnsembleFromResultsFiles(writePathResults, dataHeaders, dataPaths, datasets, baseClassifiers, numFolds, ensembleIDsInStorage[ensemble], ensembleClasses[ensemble]);



        //build the results analysis sheets and figures
        for (int archive = 0; archive < dataHeaders.length; archive++) {
            String analysisName = dataHeaders[archive] + "CAWPEvsHeteroEnsembles_BasicClassifiers";
            buildCAWPEPaper_BuildResultsAnalysis(writePathResults+dataHeaders[archive]+"/", writePathAnalysis,
                                       analysisName, ensembleIDsInStorage, ensembleIDsOnFigures, datasets[archive], numFolds);
        }

        //done!
    }
 
    public static void buildCAWPEPaper_BuildResultsAnalysis(String resultsReadPath, String analysisWritePath,
                                       String analysisName, String[] classifiersInStorage, String[] classifiersOnFigs, String[] datasets, int numFolds) throws Exception {
        System.out.println("buildCAWPEPaper_BuildResultsAnalysis");

        new MultipleClassifierEvaluation(analysisWritePath, analysisName, numFolds).
            setTestResultsOnly(false).
//            setBuildMatlabDiagrams(true).
            setBuildMatlabDiagrams(false).
            setDatasets(datasets).
            readInClassifiers(classifiersInStorage, classifiersOnFigs, resultsReadPath).
            runComparison();
    }

    public static void buildCAWPEPaper_BuildEnsembleFromResultsFiles(String baseWritePath, String[] dataHeaders, String[] dataPaths, String[][] datasetNames,
                                                                String[] baseClassifiers, int numFolds, String ensembleID, Class ensembleClass) throws Exception {

        Instances train = null, test = null, all = null; //UCR has predefined train/test splits, UCI data just comes as a whole, so are loaded/resampled differently
        Instances[] data = null; //however it's loaded/resampled, will eventually end up here, { train, test }

        for (int archive = 0; archive < dataHeaders.length; archive++) {
            String writePath = baseWritePath + dataHeaders[archive] + "/";

            for (String dset : datasetNames[archive]) {
                System.out.println(dset);

                if (dataHeaders[archive].equals("UCI"))
                    all = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + ".arff");
                else if ((dataHeaders[archive].contains("UCR"))) {
                    train = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + "_TRAIN.arff");
                    test = DatasetLoading.loadDataNullable(dataPaths[archive] + dset + "/" + dset + "_TEST.arff");
                }

                for (int fold = 0; fold < numFolds; fold++) {
                    //building particular ensembles with different parameters is a bit
                    //more involved so we skip some of the automated stages (especically setClassifier(...) in the
                    //experiments class to build the particular format wanted.
                    //in this example code, i've jsut assumed that default parameters
                    //(aside from the base classifiers) are being used.
                    //this code could ofc be editted to build whatever particular classifiers
                    //you want, instead of using the janky reflection

                    String predictions = writePath+ensembleID+"/Predictions/"+dset+"/";
                    File f=new File(predictions);
                    if(!f.exists())
                        f.mkdirs();

                    //Check whether fold already exists, if so, dont do it, just quit
                    if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
                        if (dataHeaders[archive].equals("UCI"))
                            data = InstanceTools.resampleInstances(all, fold, .5);
                        else if ((dataHeaders[archive].contains("UCR")))
                            data = InstanceTools.resampleTrainAndTestInstances(train, test, fold);

                        //cawpe is the base class from which all the heterogeneous ensembles are implemented, since this
                        //already has the base classifier file reading/writing built in etcetc.
                        CAWPE c = (CAWPE) ensembleClass.getConstructor().newInstance();

                        c.setEnsembleIdentifier(ensembleID);
                        c.setClassifiers(null, baseClassifiers, null);
                        c.setBuildIndividualsFromResultsFiles(true);
                        c.setResultsFileLocationParameters(writePath, dset, fold);
                        c.setRandSeed(fold);
                        c.setPerformCV(true);

                        //'custom' classifier built, now put it back in the normal experiments pipeline
                        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
                        exp.classifierName = ensembleID;
                        exp.datasetName = dset;
                        exp.foldId = fold;
                        exp.generateErrorEstimateOnTrainSet = true;
                        Experiments.runExperiment(exp,data[0],data[1],c,predictions);
                    }
                }
            }
        }
    }



    public static void main(String[] args) throws Exception {
//        exampleCAWPEUsage();

        buildCAWPEPaper_AllResultsForFigure3();

//        testBuildingInds(3);
//        testLoadingInds(2);
    }
}
