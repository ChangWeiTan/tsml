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
package evaluation;

import evaluation.storage.ClassifierResults;
import ResultsProcessing.MatlabController;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;
import utilities.DebugPrinting;
import utilities.ErrorReport;
import utilities.generic_storage.Pair;

/**
 * This essentially just wraps ClassifierResultsAnalysis.performFullEvaluation(...) in a nicer to use way. Will be updated over time
 * 
 * Builds summary stats, sig tests, and optionally matlab dias for the ClassifierResults objects provided/files pointed to on disk. Can optionally use
 * just the test results, if that's all that is available, or both train and test (will also compute the train test diff)
 * 
 * USAGE: see workingExampleCodeRunnableOnTSCServerMachine() for fleshed out example, in short though:
 *      Construct object, set any non-default bool options, set any non-default statistics to use, set datasets to compare on, and (rule of thumb) LASTLY add 
 *      classifiers/results located in memory or on disk and call runComparison(). 
 * 
 *      Least-code one-off use case that's good enough for most problems is: 
 *          new MultipleClassifierEvaluation("write/path/", "experimentName", numFolds).
 *              setDatasets(development.experiments.DataSets.UCIContinuousFileNames).
 *              readInClassifiers(new String[] {"NN", "C4.5"}, baseReadingPath).
 *              runComparison();  
 * 
 * Will call findAllStatsOnce on each of the ClassifierResults (i.e. will do nothing if findAllStats has already been called elsewhere before), 
 * and there's a bool (default true) to set whether to null the instance prediction info after stats are found to save memory. 
 * If some custom analysis method not defined natively in classifierresults that uses the individual prediction info, 
 * (defined using addEvaluationStatistic(String statName, Function<ClassifierResults, Double> classifierResultsManipulatorFunction))
 will need to keep the info, but that can get problematic depending on how many classifiers/datasets/folds there are
 
 For some reason, the first excel workbook writer library i found/used makes xls files (instead of xlsx) and doesn't 
 support recent excel default fonts. Just open it and saveas if you want to switch it over. There's a way to globally change font in a workbook 
 if you want to change it back

 Future work (here and in ClassifierResultsAnalysis.performFullEvaluation(...)) when wanted/needed could be to 
 handle incomplete results (e.g random folds missing), more matlab figures over time, and more refactoring of the obviously bad parts of the code
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleClassifierEvaluation implements DebugPrinting { 
    private String writePath; 
    private String experimentName;
    private List<String> datasets;
    private Map<String, Map<String, String[]>> datasetGroupings; // Map<GroupingMethodTitle(e.g "ByNumAtts"), Map<GroupTitle(e.g "<100"), dsetsInGroup(must be subset of datasets)>>
    private Map<String, ClassifierResults[/* train/test */][/* dataset */][/* fold */]> classifiersResults; 
    private int numFolds;
    private ArrayList<PerformanceMetric> metrics;
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    private boolean buildMatlabDiagrams;
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found 
     */
    private boolean cleanResults;

    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     */
    private boolean testResultsOnly;
    
    /**
     * if true, will basically just transpose the results, and swap the dataset names for the classifiernames. 
     * ranks, sig tests, etc, will then compare the 'performance of datasets'. Intended use when comparing 
     * e.g. different preprocessing techniques which are saved as arffs and then a collection of classifiers 
     * are evaluated on each.
     */
    private boolean evaluateDatasetsOverClassifiers;
    
    /**
     * if true, will perform xmeans clustering on the classifierXdataset results, to find data-driven datasetgroupings, as well
     * as any extra dataset groupings you've defined.
     * 
     * 1) for each dataset, each classifier's [stat] is replaced by its difference to the util_mean for that dataset
     * e.g if scores of 3 classifiers on a dataset are { 0.8, 0.7, 0.6 }, the new vals will be { 0.1, 0, -0.1 } 
     * 
     * 2) weka instances are formed from this data, with classifiers as atts, datasets as insts
     * 
     * 3) xmeans clustering performed, as a (from a human input pov) quick way of determining number of clusters + those clusters
     * 
     * 4) perform the normal grouping analysis based on those clusters
     */
    private boolean performPostHocDsetResultsClustering;
    
    
    /**
     * if true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * defaults to false
     */
    private boolean ignoreMissingDistributions;
    
    /**
     * if true, will close the matlab connected once analysis complete (if it was opened)
     * if false, will allow for multiple stats runs in a single execution, but the 
     * thread will not end while the matlab instance is open, so the connection must 
     * be closed or execution terminated manually
     */
    private boolean closeMatlabConnectionWhenFinished = true;
    
    /**
     * @param experimentName forms the analysis directory name, and the prefix to most files
     */
    public MultipleClassifierEvaluation(String writePath, String experimentName, int numFolds) {
        this.writePath = writePath;
        this.experimentName = experimentName;
        this.numFolds = numFolds;
        
        this.buildMatlabDiagrams = false;
        this.cleanResults = true;
        this.testResultsOnly = true;
        this.performPostHocDsetResultsClustering = false;
        this.ignoreMissingDistributions = false;
        
        this.datasets = new ArrayList<>();
        this.datasetGroupings = new HashMap<>();
        this.classifiersResults = new HashMap<>();
        
        this.metrics = PerformanceMetric.getDefaultStatistics();
    }

    /**
     * if true, will basically just transpose the results, and swap the dataset names for the classifiernames. 
     * ranks, sig tests, etc, will then compare the 'performance of datasets'. Intended use when comparing 
     * e.g. different preprocessing techniques which are saved as arffs and then a collection of classifiers 
     * are evaluated on each.
     */
    public void setEvaluateDatasetsOverClassifiers(boolean evaluateDatasetsOverClassifiers) {
        this.evaluateDatasetsOverClassifiers = evaluateDatasetsOverClassifiers;
    }
    
    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     */
    public MultipleClassifierEvaluation setTestResultsOnly(boolean b) {
        testResultsOnly = b;
        return this;
    }
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    public MultipleClassifierEvaluation setBuildMatlabDiagrams(boolean b) {
        buildMatlabDiagrams = b;
        closeMatlabConnectionWhenFinished = true;
        return this;
    }
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    public MultipleClassifierEvaluation setBuildMatlabDiagrams(boolean b, boolean closeMatlabConnectionWhenFinished) {
        buildMatlabDiagrams = b;
        this.closeMatlabConnectionWhenFinished = closeMatlabConnectionWhenFinished;
        return this;
    }
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found 
     */
    public MultipleClassifierEvaluation setCleanResults(boolean b) {
        cleanResults = b;
        return this;
    }
    
    public MultipleClassifierEvaluation setIgnoreMissingDistributions(boolean ignoreMissingDistributions) {
        this.ignoreMissingDistributions = ignoreMissingDistributions;
        return this;
    }
    
    /**
     * if true, will perform xmeans clustering on the classifierXdataset results, to find data-driven datasetgroupings, as well
     * as any extra dataset groupings you've defined.
     * 
     * 1) for each dataset, each classifier's [stat] is replaced by its difference to the util_mean for that dataset
      e.g if scores of 3 classifiers on a dataset are { 0.8, 0.7, 0.6 }, the new vals will be { 0.1, 0, -0.1 } 
 
 2) weka instances are formed from this data, with classifiers as atts, datasets as insts
 
 3) xmeans clustering performed, as a (from a human input pov) quick way of determining number of clusters + those clusters
 
 4) perform the normal grouping analysis based on those clusters
     */
    public MultipleClassifierEvaluation setPerformPostHocDsetResultsClustering(boolean b) {
        performPostHocDsetResultsClustering = b;
        return this;
    }
    
    /**
     * @param datasetListFilename the path and name of a file containing a list of datasets, one per line
     * @throws FileNotFoundException 
     */
    public MultipleClassifierEvaluation setDatasets(String datasetListFilename) throws FileNotFoundException {
        Scanner filein = new Scanner(new File(datasetListFilename));
        
        List<String> dsets = new ArrayList<>();
        while (filein.hasNextLine())
            dsets.add(filein.nextLine());
        
        return setDatasets(dsets);
    }
    
    public MultipleClassifierEvaluation setDatasets(List<String> datasets) {
        this.datasets = datasets;
        return this;
    }
    public MultipleClassifierEvaluation setDatasets(String[] datasets) {
        this.datasets = Arrays.asList(datasets);
        return this;
    }
    public MultipleClassifierEvaluation addDataset(String dataset) {
        this.datasets.add(dataset);
        return this;
    }
    public MultipleClassifierEvaluation removeDataset(String dataset) {
        this.datasets.remove(dataset);
        return this;
    }
    public MultipleClassifierEvaluation clearDatasets() {
        this.datasets.clear();
        return this;
    }
    
    /**
     * Pass a directory containing a number of text files. The directory name (not including path)
     * becomes the groupingMethodName (e.g ByNumAtts). Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleClassifierEvaluation setDatasetGroupingFromDirectory(String groupingDirectory) throws FileNotFoundException { 
        setDatasetGroupingFromDirectory(groupingDirectory, (new File(groupingDirectory)).getName());
        return this;
    }
    
    /**
     * Use this if you want to define a different grouping method name to the directory name
     * for clean printing purposes/clarity. E.g directory name might be 'UCRDsetGroupingByNumAtts_2groups', but the 
     * name you define to be printed on the analysis could just be 'ByNumAtts'
     * 
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleClassifierEvaluation setDatasetGroupingFromDirectory(String groupingDirectory, String customGroupingMethodName) throws FileNotFoundException { 
        clearDatasetGroupings();
        addDatasetGroupingFromDirectory(groupingDirectory, customGroupingMethodName);
        return this;
    }
    
    /**
     * Pass a directory containing a number of DIRECTORIES that define groupings. Each subdirectory contains 
     * a number of text files. The names of these subdirectories define the grouping method names. 
     * Each text file within contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleClassifierEvaluation addAllDatasetGroupingsInDirectory(String groupingSuperDirectory) throws FileNotFoundException { 
        for (String groupingDirectory : (new File(groupingSuperDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isDirectory();
            }
        })) {
            
            addDatasetGroupingFromDirectory(groupingSuperDirectory + groupingDirectory);
        }
        return this;
    }
    
    /**
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleClassifierEvaluation addDatasetGroupingFromDirectory(String groupingDirectory) throws FileNotFoundException { 
        addDatasetGroupingFromDirectory(groupingDirectory, (new File(groupingDirectory)).getName());
        return this;
    }
    
    /**
     * Use this if you want to define a different grouping method name to the directory name
     * for clean printing purposes/clarity. E.g directory name might be 'UCRDsetGroupingByNumAtts_2groups', but the 
     * name you define to be printed on the analysis could just be 'ByNumAtts'
     * 
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleClassifierEvaluation addDatasetGroupingFromDirectory(String groupingDirectory, String customGroupingMethodName) throws FileNotFoundException { 
        File[] groups = (new File(groupingDirectory)).listFiles();
        String[] groupNames = new String[groups.length];
        String[][] dsets = new String[groups.length][];
        
        for (int i = 0; i < groups.length; i++) {
            groupNames[i] = groups[i].getName().replace(".txt", "").replace(".csv", "");
            Scanner filein = new Scanner(groups[i]);
        
            List<String> groupDsets = new ArrayList<>();
            while (filein.hasNextLine())
                groupDsets.add(filein.nextLine());
            
            dsets[i] = groupDsets.toArray(new String [] { });
        }
        
        addDatasetGrouping(customGroupingMethodName, groupNames, dsets);
        return this;
    }
    
    /**
     * The purely array based method for those inclined
     * 
     * @param groupingMethodName e.g "ByNumAtts"
     * @param groupNames e.g { "<100", ">100" }, where group name indices line up with outer array of 'groups' 
     * @param groups [groupNames.length][variablelength number of datasets]
     */
    public MultipleClassifierEvaluation setDatasetGrouping(String groupingMethodName, String[] groupNames, String[][] groups) {
        clearDatasetGroupings();
        addDatasetGrouping(groupingMethodName, groupNames, groups);
        return this;
    }
    
    /**
     * The purely array based method for those inclined
     * 
     * @param groupingMethodName e.g "ByNumAtts"
     * @param groupNames e.g { "<100", ">100" }, where group name indices line up with outer array of 'groups' 
     * @param groups [groupNames.length][variablelength number of datasets]
     */
    public MultipleClassifierEvaluation addDatasetGrouping(String groupingMethodName, String[] groupNames, String[][] groups) {
        Map<String, String[]> groupsMap = new HashMap<>();
        for (int i = 0; i < groupNames.length; i++)
            groupsMap.put(groupNames[i], groups[i]);
        
        datasetGroupings.put(groupingMethodName, groupsMap);
        return this;
    }
    
    public MultipleClassifierEvaluation clearDatasetGroupings() {
        this.datasetGroupings.clear();
        return this;
    }

    /**
     * 4 stats: acc, balanced acc, auroc, nll
     */
    public MultipleClassifierEvaluation setUseDefaultEvaluationStatistics() {
        metrics = PerformanceMetric.getDefaultStatistics();
        return this;
    }

    public MultipleClassifierEvaluation setUseAccuracyOnly() {
        metrics = PerformanceMetric.getAccuracyStatistic();
        return this;
    }
    
    public MultipleClassifierEvaluation setUseAllStatistics() {
        metrics = PerformanceMetric.getAllStatistics();
        return this;
    }
    
    public MultipleClassifierEvaluation addEvaluationStatistic(PerformanceMetric metric) {
        metrics.add(metric);
        return this;
    }
    
    public MultipleClassifierEvaluation removeEvaluationStatistic(String name) {
        for (PerformanceMetric metric : metrics)
            if (metric.name.equalsIgnoreCase(name))
                metrics.remove(metric);
        return this;
    }
    
    public MultipleClassifierEvaluation clearEvaluationStatistics() {
        metrics.clear();
        return this;
    }
    
    /**
     * @param trainDatasetFoldResults [dataset][fold], e.g [121][30]
     */
    public MultipleClassifierEvaluation addClassifier(String classifierName, ClassifierResults[][] trainDatasetFoldResults, ClassifierResults[][] testDatasetFoldResults) throws Exception {
        if (datasets.size() == 0) 
            throw new Exception("No datasets set for evaluation");

        for (int d = 0; d < testDatasetFoldResults.length; d++) {
            for (int f = 0; f < testDatasetFoldResults[d].length; f++) {
                if (!testResultsOnly && trainDatasetFoldResults != null) {
                    trainDatasetFoldResults[d][f].findAllStatsOnce();
                    if (cleanResults)
                        trainDatasetFoldResults[d][f].cleanPredictionInfo();
                }
                testDatasetFoldResults[d][f].findAllStatsOnce();
                if (cleanResults)
                    testDatasetFoldResults[d][f].cleanPredictionInfo();
            }
        }

        classifiersResults.put(classifierName, new ClassifierResults[][][] { trainDatasetFoldResults, testDatasetFoldResults } );
        return this;
    }
    /**
     * @param trainClassifierDatasetFoldResults [classifier][dataset][fold], e.g [5][121][30]
     */
    public MultipleClassifierEvaluation addClassifiers(String[] classifierNames, ClassifierResults[][][] trainClassifierDatasetFoldResults, ClassifierResults[][][] testClassifierDatasetFoldResults) throws Exception {
        for (int i = 0; i < classifierNames.length; i++)
            addClassifier(classifierNames[i], trainClassifierDatasetFoldResults[i], trainClassifierDatasetFoldResults[i]);            
        return this;
    }

    /**
     * Read in the results from file classifier by classifier, can be used if results are in different locations 
     * (e.g beast vs local)
     * 
     * @param classifierName Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing a subdirectory named [classifierName]
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifier(String classifierName, String baseReadPath) throws Exception { 
        return readInClassifier(classifierName, classifierName, baseReadPath);
    }
    
    /**
     * Read in the results from file classifier by classifier, can be used if results are in different locations 
     * (e.g beast vs local)
     * 
     * @param classifierNameInStorage Should exactly match the directory name of the results to use
     * @param classifierNameInOutput Can provide a different 'human' friendly or context-aware name if appropriate, to be printed in the output files/on images
     * @param baseReadPath Should be a directory containing a subdirectory named [classifierName]
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifier(String classifierNameInStorage, String classifierNameInOutput, String baseReadPath) throws Exception { 
        if (datasets.size() == 0) 
            throw new Exception("No datasets set for evaluation");

        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";

        printlnDebug(classifierNameInStorage + "(" + classifierNameInOutput + ") reading");

        int totalFnfs = 0;
        ErrorReport er = new ErrorReport("FileNotFoundExceptions thrown (### total):\n");

        ClassifierResults[][][] results = new ClassifierResults[2][datasets.size()][numFolds];
        if (testResultsOnly)
            results[0]=null; //crappy but w/e
     
        //train files may be produced via TrainAccuracyEstimate, older code
        //while test files likely by experiments, but still might be a very old file
        //so having separate checks for each.
        boolean ignoringDistsFirstTimeFlagTrain = true;
        boolean ignoringDistsFirstTimeFlagTest = true;
        
        for (int d = 0; d < datasets.size(); d++) {
            for (int f = 0; f < numFolds; f++) {
                
                if (!testResultsOnly) {
                    String trainFile = baseReadPath + classifierNameInStorage + "/Predictions/" + datasets.get(d) + "/trainFold" + f + ".csv";
                    try {
                        results[0][d][f] = new ClassifierResults(trainFile);
                        if (ignoreMissingDistributions) {
                            boolean wasMissing = results[0][d][f].populateMissingDists();
                            if (wasMissing && ignoringDistsFirstTimeFlagTrain) {
                                System.out.println("---------Probability distributions missing, but ignored: " 
                                        + classifierNameInStorage + " - " + datasets.get(d) + " - " + f + " - train");
                                ignoringDistsFirstTimeFlagTrain = false;
                            }
                        }
                        results[0][d][f].findAllStatsOnce();
                        if (cleanResults)
                            results[0][d][f].cleanPredictionInfo();
                    } catch (FileNotFoundException ex) {
                        er.log(trainFile + "\n");
                        totalFnfs++;
                    }
                }
                
                String testFile = baseReadPath + classifierNameInStorage + "/Predictions/" + datasets.get(d) + "/testFold" + f + ".csv";
                try {
                    results[1][d][f] = new ClassifierResults(testFile);
                    if (ignoreMissingDistributions) {
                        boolean wasMissing = results[1][d][f].populateMissingDists();
                        if (wasMissing && ignoringDistsFirstTimeFlagTest) {
                            System.out.println("---------Probability distributions missing, but ignored: " 
                                    + classifierNameInStorage + " - " + datasets.get(d) + " - " + f + " - test");
                            ignoringDistsFirstTimeFlagTest = false;
                        }
                    }
                    results[1][d][f].findAllStatsOnce();
                    if (cleanResults)
                        results[1][d][f].cleanPredictionInfo();
                } catch (FileNotFoundException ex) {
                    er.log(testFile + "\n");
                    totalFnfs++;
                } 
            }
        }

        er.getLog().replace("###", totalFnfs+"");
        er.throwIfErrors();

        printlnDebug(classifierNameInStorage + "(" + classifierNameInOutput + ") successfully read in");

        classifiersResults.put(classifierNameInOutput, results);
        return this;
    }
    /**
     * Read in the results from file from a common base path
     * 
     * @param classifierNames Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing subdirectories with the names in classifierNames 
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifiers(String[] classifierNames, String baseReadPath) throws Exception { 
        return readInClassifiers(classifierNames, classifierNames, baseReadPath);
    }
    
    /**
     * Read in the results from file from a common base path
     * 
     * @param classifierNamesInOutput Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing subdirectories with the names in classifierNames 
     * @return 
     */
    public MultipleClassifierEvaluation readInClassifiers(String[] classifierNamesInStorage, String[] classifierNamesInOutput, String baseReadPath) throws Exception { 
        if (classifierNamesInOutput.length != classifierNamesInStorage.length)
            throw new Exception("Sizes of the classifier names to read in and use in output differ: classifierNamesInStorage.length=" 
                    + classifierNamesInStorage.length + ", classifierNamesInOutput.length="+classifierNamesInOutput.length);
        
        ErrorReport er = new ErrorReport("Results files not found:\n");
        for (int i = 0; i < classifierNamesInStorage.length; i++) {
            try {
                readInClassifier(classifierNamesInStorage[i], classifierNamesInOutput[i], baseReadPath);
            } catch (Exception e) {
                er.log("Classifier Errors: " + classifierNamesInStorage[i] + "\n" + e);
            }
        }
        er.throwIfErrors();
        return this;
    }
    
    public MultipleClassifierEvaluation removeClassifier(String classifierName) {
        classifiersResults.remove(classifierName);
        return this;
    }
    
    public MultipleClassifierEvaluation clearClassifiers() {
        classifiersResults.clear();
        return this;
    }
    
    private void transposeEverything() { 
        //need to put the classifier names into the datasets list
        //repalce the entries of the classifier results map with entries for each dataset
        //to go from this:    Map<String/*classifierNames*/, ClassifierResults[/* train/test */][/* dataset */][/* fold */]> classifiersResults; 
        //           and a list of datasetnames 
        //to this:            Map<String/*datasetNames*/, ClassifierResults[/* train/test */][/* classifier */][/* fold */]> classifiersResults; 
        //           and a list of classifiernames
        
        int numClassifiers = classifiersResults.size();
        int numDatasets = datasets.size();
        
        //going to pull everything out into parallel arrays and work that way... 
        //innefficient, but far more likely to actually work
        String[] origClassifierNames = new String[numClassifiers];
        ClassifierResults[][][][] origClassifierResults = new ClassifierResults[numClassifiers][][][];
        
        int i = 0;
        for (Map.Entry<String, ClassifierResults[][][]> origClassiiferResultsEntry : classifiersResults.entrySet()) {
            origClassifierNames[i] = origClassiiferResultsEntry.getKey();
            origClassifierResults[i] = origClassiiferResultsEntry.getValue();
            i++;
        }
        
        ClassifierResults[][][][] newDataseResultsArr = new ClassifierResults[numDatasets][2][numClassifiers][numFolds];
        
        
        //do the transpose
        for (int dset = 0; dset < numDatasets; dset++) {
            
            int splitStart = 0;
            if (testResultsOnly) {
                newDataseResultsArr[dset][0] = null; //no train results
                splitStart = 1; //dont try and copythem over
            }
            
            for (int split = splitStart; split < 2; split++) {
                for (int classifier = 0; classifier < numClassifiers; classifier++) {
                    //leaving commented for reference, but can skip this loop, and copy across fold array refs instead of individual fold refs
                    //for (int fold = 0; fold < numFolds; fold++)
                    //    newDataseResultsArr[dset][split][classifier][fold] = origClassifierResults[classifier][split][dset][fold];
                    
//                    System.out.println("newDataseResultsArr[dset]" + newDataseResultsArr[dset].toString().substring(0, 30));
//                    System.out.println("newDataseResultsArr[dset][split]" + newDataseResultsArr[dset][split].toString().substring(0, 30));
//                    System.out.println("newDataseResultsArr[dset][split][classifier]" + newDataseResultsArr[dset][split][classifier].toString().substring(0, 30));
//                    System.out.println("origClassifierResults[classifier]" + origClassifierResults[classifier].toString().substring(0, 30));
//                    System.out.println("origClassifierResults[classifier][split]" + origClassifierResults[classifier][split].toString().substring(0, 30));
//                    System.out.println("origClassifierResults[classifier][split][dset]" + origClassifierResults[classifier][split][dset].toString().substring(0, 30));
                    
                    newDataseResultsArr[dset][split][classifier] = origClassifierResults[classifier][split][dset];
                }
            }
        }
        
        //and put back into a map
        Map<String, ClassifierResults[][][]> newDsetResultsMap = new HashMap<>();
        for (int dset = 0; dset < numDatasets; dset++)
            newDsetResultsMap.put(datasets.get(dset), newDataseResultsArr[dset]);
        
        this.classifiersResults = newDsetResultsMap; 
        this.datasets = Arrays.asList(origClassifierNames);
    }
    
    public void runComparison() {
        if (evaluateDatasetsOverClassifiers) {
            transposeEverything();
        }
        
        ArrayList<ClassifierResultsAnalysis.ClassifierEvaluation> results = new ArrayList<>(classifiersResults.size());
        for (Map.Entry<String, ClassifierResults[][][]> classifier : classifiersResults.entrySet())
            results.add(new ClassifierResultsAnalysis.ClassifierEvaluation(classifier.getKey(), classifier.getValue()[1], classifier.getValue()[0]));
        
        ClassifierResultsAnalysis.buildMatlabDiagrams = buildMatlabDiagrams;
        ClassifierResultsAnalysis.testResultsOnly = testResultsOnly;
        
        //ClassifierResultsAnalysis will find this flag internally as queue to do clustering
        if (performPostHocDsetResultsClustering) 
            datasetGroupings.put(ClassifierResultsAnalysis.clusterGroupingIdentifier, null); 
        
        printlnDebug("Writing started");
        ClassifierResultsAnalysis.performFullEvaluation(writePath, experimentName, metrics, results, datasets.toArray(new String[] { }), datasetGroupings);
        printlnDebug("Writing finished");
        
        if (buildMatlabDiagrams && closeMatlabConnectionWhenFinished)
            MatlabController.getInstance().discconnectMatlab();
    }

    public static void main(String[] args) throws Exception {
//        String basePath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
////            String basePath = "Z:/Results/FinalisedUCIContinuous/";
//
//        MultipleClassifierEvaluation mcc = 
//            new MultipleClassifierEvaluation("C:/JamesLPHD/analysisTest/", "testrunPWS10", 30);
//        
//        mcc.setTestResultsOnly(true); //as is default
//        mcc.setBuildMatlabDiagrams(true); //as is default
//        mcc.setCleanResults(true); //as is default
//        mcc.setDebugPrinting(true);
//        
//        mcc.setUseDefaultEvaluationStatistics(); //as is default, acc,balacc,auroc,nll
////        mcc.setUseAccuracyOnly();
////        mcc.addEvaluationStatistic("F1", (ClassifierResults cr) -> {return cr.f1;}); //add on the f1 stat too
////        mcc.setUseAllStatistics();
//        
//        mcc.setDatasets(development.experiments.DataSets.UCIContinuousFileNames);
//        
//        //general rule of thumb: set/add/read the classifiers as the last thing before running
//        mcc.readInClassifiers(new String[] {"NN", "C4.5", "RotF", "RandF"}, basePath); 
////        mcc.readInClassifier("RandF", basePath); //
//
//        mcc.runComparison();  

        
//        new MultipleClassifierEvaluation("Z:/Results/FinalisedUCIContinuousAnalysis/", "testy_mctestface", 30).
//            setTestResultsOnly(false).
//            setDatasets(development.experiments.DataSets.UCIContinuousFileNames).
//            readInClassifiers(new String[] {"1NN", "C4.5"}, "Z:/Results/FinalisedUCIContinuous/").
//            runComparison(); 
//        new MultipleClassifierEvaluation("C:\\JamesLPHD\\DatasetGroups\\anatesting\\", "test29", 30).
////            setBuildMatlabDiagrams(true).
////            setUseAllStatistics().
////            setDatasets(Arrays.copyOfRange(development.experiments.DataSets.UCIContinuousFileNames, 0, 10)). //using only 10 datasets just to make it faster...
////            setDatasets("C:/Temp/dsets.txt").
//            setDatasets("C:/Temp/dsets.txt"). 
//            setDatasetGroupingFromDirectory("C:\\JamesLPHD\\DatasetGroups\\TestGroups"). 
//            setPerformPostHocDsetResultsClustering(true).
//            readInClassifiers(new String[] {"1NN", "C4.5", "MLP", "RotF", "RandF"}, "C:\\JamesLPHD\\HESCA\\UCR\\UCRResults").
//            runComparison(); 

        workingExampleCodeRunnableOnTSCServerMachine();
    }
    
    public static void workingExampleCodeRunnableOnTSCServerMachine() throws FileNotFoundException, Exception {
        //Running from my PC, this code takes 34 seconds to run, despite looking at only 10 folds of 10 datasets. 
        //The majority of this time is eaten up by reading the results from the server. If you have results on your local PC, this runs in a second.
        
        //to rerun this from a clean slate to check validity, delete any existing 'Example1' folder in here: 
        String folderToWriteAnalysisTo = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/";
        String nameOfAnalysisWhichWillBecomeFolderName = "ExampleTranspose";
        int numberOfFoldsAKAResamplesOfEachDataset = 10;
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(folderToWriteAnalysisTo, nameOfAnalysisWhichWillBecomeFolderName, numberOfFoldsAKAResamplesOfEachDataset); //10 folds only to make faster... 
        
        String aFileWithListOfDsetsToUse = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsets.txt";
        mce.setDatasets(aFileWithListOfDsetsToUse);
        
        String aDirectoryContainingFilesThatDefineDatasetGroupings = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroupings/evenAndOddDsets/";
        String andAnother = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroupings/topAndBotHalves/";
        mce.addDatasetGroupingFromDirectory(aDirectoryContainingFilesThatDefineDatasetGroupings);
        mce.addDatasetGroupingFromDirectory(andAnother);
        
        mce.setPerformPostHocDsetResultsClustering(true); //will create 3rd data-driven grouping automatically
        
        String[] classifiers = new String[] {"1NN", "C4.5", "NB"};
        String directoryWithResultsClassifierByClassifier =  "Z:/Backups/Results_7_2_19/FinalisedUCIContinuous/";
        mce.readInClassifiers(classifiers, directoryWithResultsClassifierByClassifier);
        
//        mce.setEvaluateDatasetsOverClassifiers(true); //cannot use with the dataset groupings, in this example. could define classifier groupings though ! 
        
        mce.runComparison(); 
        
        //minimal version of above: 
//        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/", "Example1", 10); //10 folds only to make faster... 
//        mce.setDatasets("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsets.txt");
//        mce.addDatasetGroupingFromDirectory("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroups/randomGrouping1/");
//        mce.addDatasetGroupingFromDirectory("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroups/randomGrouping2/");
//        mce.setPerformPostHocDsetResultsClustering(true); //will create 3rd data-driven grouping automatically
//        mce.readInClassifiers(new String[] {"1NN", "C4.5", "MLP", "RotF", "RandF"}, "Z:/Results/FinalisedUCIContinuous/");
//        mce.runComparison(); 
    }
}
