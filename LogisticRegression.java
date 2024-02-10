import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;


class ConfusionMatrix {
    private int[][] matrix;

    public ConfusionMatrix(int[][] matrix) {
        this.matrix = matrix;
    }

    public int getTruePositives() {
        return matrix[1][1];
    }

    public int getTrueNegatives() {
        return matrix[0][0];
    }

    public int getFalsePositives() {
        return matrix[0][1];
    }

    public int getFalseNegatives() {
        return matrix[1][0];
    }

    public double getRecall() {
        int truePositives = getTruePositives();
        int falseNegatives = getFalseNegatives();
        return (double) truePositives / (truePositives + falseNegatives);
    }

    public double getPrecision() {
        int truePositives = getTruePositives();
        int falsePositives = getFalsePositives();
        return (double) truePositives / (truePositives + falsePositives);
    }
    public double getF1Score() {
        double precision = getPrecision();
        double recall = getRecall();
        return 2 * (precision * recall) / (precision + recall);
    }

    public void printMatrix() {
        System.out.println("Confusion Matrix:");
        System.out.println("Actual \\ Predicted | 0 (Non-Spam) | 1 (Spam)");
        System.out.println("-----------------|--------------|-----------");
        System.out.println("0 (Non-Spam)      |     " + matrix[0][0] + "       |     " + matrix[0][1]);
        System.out.println("1 (Spam)          |     " + matrix[1][0] + "       |     " + matrix[1][1]);
    }

    public static ConfusionMatrix calculateConfusionMatrix(List<String> testData, double[] weights, double bias, Map<String, Integer> idfMap) {
        int[][] matrix = new int[2][2];
        int defaultClass = 0;

        for (String emailText : testData) {
            double[] emailFeatures = LogisticRegression.extractFeaturesFromEmail(emailText, idfMap);
            double prediction = LogisticRegression.predict(emailFeatures, weights, bias);
            int predictedClass = (prediction > 0.5) ? 1 : defaultClass;

            int actualClass = (emailText.contains("spam")) ? 1 : defaultClass;

            matrix[actualClass][predictedClass]++;
        }

        return new ConfusionMatrix(matrix);
    }
}
public class LogisticRegression {
    public static void main(String[] args) {
    List<String> dataset = loadDataset("spam.csv");

    Collections.shuffle(dataset, new Random(System.currentTimeMillis()));

    int splitIndex = (int) (dataset.size() * 0.7);
    List<String> trainingData = dataset.subList(0, splitIndex);
    List<String> testData = dataset.subList(splitIndex, dataset.size());

    double learningRate = 0.001;
    int numIterations = 100;

    List<double[]> emailFeatures = new ArrayList<>();
    Map<String, Integer> idfMap = calculateIDF(dataset); 

    for (String emailText : trainingData) {
        double[] features = extractFeaturesFromEmail(emailText, idfMap);
        emailFeatures.add(features);
    }

    int numFeatures = emailFeatures.get(0).length;
    double[] weights = new double[numFeatures];
    double bias = 0.0;

    trainModel(emailFeatures, weights, bias, learningRate, numIterations);

    double accuracy = testModel(testData, weights, bias);
    System.out.println("Accuracy on test data: " + accuracy);
}


    public static double testModel(List<String> testData, double[] weights, double bias) {
        List<String> dataset = loadDataset("spam.csv");
        Map<String, Integer> idfMap = calculateIDF(dataset);  
        ConfusionMatrix confusionMatrix = ConfusionMatrix.calculateConfusionMatrix(testData, weights, bias, idfMap);
        confusionMatrix.printMatrix();

        int truePositives = confusionMatrix.getTruePositives();
        int trueNegatives = confusionMatrix.getTrueNegatives();
        int falsePositives = confusionMatrix.getFalsePositives();
        int falseNegatives = confusionMatrix.getFalseNegatives();

        double recall = confusionMatrix.getRecall();
        double precision = confusionMatrix.getPrecision();
        double f1Score = confusionMatrix.getF1Score();

        System.out.println("True Positives (TP): " + truePositives);
        System.out.println("True Negatives (TN): " + trueNegatives);
        System.out.println("False Positives (FP): " + falsePositives);
        System.out.println("False Negatives (FN): " + falseNegatives);
        System.out.println("Recall: " + recall);
        System.out.println("Precision: " + precision);
        System.out.println("F1-Score: " + f1Score);

        int correctPredictions = truePositives + trueNegatives;
        int totalPredictions = testData.size();

        return (double) correctPredictions / totalPredictions;
    }
   

    public static List<String> loadDataset(String filename) {
        List<String> dataset = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = br.readLine()) != null) {
                dataset.add(line);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataset;
    }

    public static void trainModel(List<double[]> dataset, double[] weights, double bias, double learningRate, int numIterations) {
        int numFeatures = dataset.get(0).length - 1;
        int numDataPoints = dataset.size();
        
        for (int iteration = 0; iteration < numIterations; iteration++) {
            for (int i = 0; i < numDataPoints; i++) {
                double[] features = dataset.get(i);
                double label = features[numFeatures];
                double predicted = predict(features, weights, bias);
                double error = label - predicted;
                bias += learningRate * error;
                for (int j = 0; j < numFeatures; j++) {
                    weights[j] += learningRate * error * features[j];
                }
            }
        }
    }
    
    public static double predict(double[] features, double[] weights, double bias) {
        double z = bias;
        for (int i = 0; i < features.length; i++) {
            z += features[i] * weights[i];
        }
        return sigmoid(z);
    }

    public static double[] extractFeaturesFromEmail(String emailText, Map<String, Integer> idfMap) {
        String[] words = emailText.split("\\s+");

        Map<String, Integer> wordFrequency = new HashMap<>();

        for (String word : words) {
            word = word.toLowerCase();
            if (wordFrequency.containsKey(word)) {
                wordFrequency.put(word, wordFrequency.get(word) + 1);
            } else {
                wordFrequency.put(word, 1);
            }
        }

        double[] features = new double[idfMap.size()];
        int i = 0;
        for (Map.Entry<String, Integer> entry : idfMap.entrySet()) {
            String word = entry.getKey();
            int tf = wordFrequency.getOrDefault(word, 0);
            double idf = Math.log((double) idfMap.size() / entry.getValue() + 1);
            features[i++] = tf * idf;
        }

        return features;
    }

    public static Map<String, Integer> calculateIDF(List<String> dataset) {
        Map<String, Integer> idfMap = new HashMap<>();

        for (String emailText : dataset) {
            Set<String> uniqueWords = new HashSet<>(Arrays.asList(emailText.split("\\s+")));
            for (String word : uniqueWords) {
                word = word.toLowerCase();
                idfMap.put(word, idfMap.getOrDefault(word, 0) + 1);
            }
        }

        return idfMap;
    }
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}