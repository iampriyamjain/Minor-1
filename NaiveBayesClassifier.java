import java.io.*;
import java.util.*;

public class NaiveBayesClassifier {
    private Map<String, Double> spamProbabilities;
    private Map<String, Double> hamProbabilities;
    private double spamPriorProbability;
    private double hamPriorProbability;
    private Set<String> vocabulary;
    private Map<String, Map<String, Double>> tfidf; // Map to store TF-IDF values

    public NaiveBayesClassifier() {
        spamProbabilities = new HashMap<>();
        hamProbabilities = new HashMap<>();
        spamPriorProbability = 0.0;
        hamPriorProbability = 0.0;
        vocabulary = new HashSet<>();
        tfidf = new HashMap<>();
    }

    public void train(List<Email> trainingSet) {
        int numSpam = 0;
        int numHam = 0;

        for (Email email : trainingSet) {
            if (email.isSpam()) {
                numSpam++;
            } else {
                numHam++;
            }

            String[] words = email.getSubject().split("\\s");

            // Calculate TF-IDF values
            Map<String, Double> tfidfValues = new HashMap<>();
            for (String word : words) {
                vocabulary.add(word);

                // Update term frequency
                tfidfValues.put(word, tfidfValues.getOrDefault(word, 0.0) + 1.0);

                // Update document frequency
                if (!tfidf.containsKey(word)) {
                    tfidf.put(word, new HashMap<>());
                }
                tfidf.get(word).put(email.getSubject(), tfidf.get(word).getOrDefault(email.getSubject(), 0.0) + 1.0);
            }

            // Update TF-IDF values
            for (String word : tfidfValues.keySet()) {
                double tf = tfidfValues.get(word) / words.length;
                double idf = Math.log((double) trainingSet.size() / (1 + tfidf.get(word).size()));
                double tfidfValue = tf * idf;

                if (!tfidf.containsKey(word)) {
                    tfidf.put(word, new HashMap<>());
                }
                tfidf.get(word).put(email.getSubject(), tfidfValue);
            }

            // Update spam and ham probabilities
            for (String word : vocabulary) {
                double tfidfValue = tfidf.get(word).getOrDefault(email.getSubject(), 0.0);
                if (email.isSpam()) {
                    spamProbabilities.put(word, spamProbabilities.getOrDefault(word, 0.0) + tfidfValue);
                } else {
                    hamProbabilities.put(word, hamProbabilities.getOrDefault(word, 0.0) + tfidfValue);
                }
            }
        }

        // Calculate prior probabilities
        spamPriorProbability = (double) numSpam / trainingSet.size();
        hamPriorProbability = (double) numHam / trainingSet.size();

        // Normalize probabilities
        normalizeProbabilities(spamProbabilities, numSpam);
        normalizeProbabilities(hamProbabilities, numHam);
    }

    public void createTFIDFMatrixCSV(String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath)) {
            // Write header
            writer.append("Word,");
            for (String emailSubject : tfidf.keySet()) {
                writer.append(emailSubject.substring(0, Math.min(emailSubject.length(), 5))).append(",");
            }
            writer.append("\n");

            // Write TF-IDF values
            for (String word : vocabulary) {
                writer.append(word).append(",");
                for (String emailSubject : tfidf.get(word).keySet()) {
                    writer.append(tfidf.get(word).get(emailSubject).toString()).append(",");
                }
                writer.append("\n");
            }
        }
    }

    private void normalizeProbabilities(Map<String, Double> probabilities, int numInstances) {
        double smoothingFactor = 1.0;
        double vocabSize = vocabulary.size();

        for (String word : vocabulary) {
            probabilities.put(word, (probabilities.getOrDefault(word, 0.0) + smoothingFactor) / (numInstances + smoothingFactor * vocabSize));
        }
    }

    public ConfusionMatrix test(List<Email> testSet) {
        int truePositive = 0;
        int falsePositive = 0;
        int trueNegative = 0;
        int falseNegative = 0;

        for (Email email : testSet) {
            String predictedLabel = classify(email.getSubject());
            boolean actualLabel = email.isSpam();

            if (predictedLabel.equals("spam") && actualLabel) {
                truePositive++;
            } else if (predictedLabel.equals("spam") && !actualLabel) {
                falsePositive++;
            } else if (predictedLabel.equals("ham") && actualLabel) {
                falseNegative++;
            } else {
                trueNegative++;
            }
        }

        return new ConfusionMatrix(truePositive, falsePositive, trueNegative, falseNegative);
    }

    public double calculateAccuracy(ConfusionMatrix confusionMatrix) {
        int total = confusionMatrix.getTotal();
        return (double) (confusionMatrix.getTruePositive() + confusionMatrix.getTrueNegative()) / total;
    }

    public double calculatePrecision(ConfusionMatrix confusionMatrix) {
        int truePositive = confusionMatrix.getTruePositive();
        int falsePositive = confusionMatrix.getFalsePositive();
        return (double) truePositive / (truePositive + falsePositive);
    }

    public double calculateRecall(ConfusionMatrix confusionMatrix) {
        int truePositive = confusionMatrix.getTruePositive();
        int falseNegative = confusionMatrix.getFalseNegative();
        return (double) truePositive / (truePositive + falseNegative);
    }

    public double calculateF1Score(ConfusionMatrix confusionMatrix) {
        double precision = calculatePrecision(confusionMatrix);
        double recall = calculateRecall(confusionMatrix);
        return 2 * (precision * recall) / (precision + recall);
    }

    public String classify(String subject) {
        String[] words = subject.split("\\s");
        double spamScore = Math.log(spamPriorProbability);
        double hamScore = Math.log(hamPriorProbability);

        for (String word : words) {
            if (vocabulary.contains(word)) {
                spamScore += Math.log(spamProbabilities.get(word));
                hamScore += Math.log(hamProbabilities.get(word));
            }
        }

        return (spamScore > hamScore) ? "spam" : "ham";
    }

    public static void main(String[] args) {
        try {
            List<Email> dataset = readCSV("spam.csv"); 
            int splitIndex = (int) (0.7 * dataset.size());
            List<Email> trainingSet = dataset.subList(0, splitIndex);
            List<Email> testingSet = dataset.subList(splitIndex, dataset.size());

            NaiveBayesClassifier classifier = new NaiveBayesClassifier();
            classifier.train(trainingSet);

            ConfusionMatrix confusionMatrix = classifier.test(testingSet);
            confusionMatrix.printTable();
            confusionMatrix.tptnfpfn();

            double accuracy = classifier.calculateAccuracy(confusionMatrix);
            System.out.println("Accuracy on the test set: " + (accuracy * 100) + "%");

            double precision = classifier.calculatePrecision(confusionMatrix);
            double recall = classifier.calculateRecall(confusionMatrix);
            double f1Score = classifier.calculateF1Score(confusionMatrix);

            System.out.println("Precision: " + precision);
            System.out.println("Recall: " + recall);
            System.out.println("F1 Score: " + f1Score);
            classifier.createTFIDFMatrixCSV("tfidf_matrix.csv");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<Email> readCSV(String csvFilePath) throws IOException {
        List<Email> emails = new ArrayList<>();

        List<String> lines = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(csvFilePath));
        String line;
        while ((line = reader.readLine()) != null) {
            lines.add(line);
        }
        reader.close();

        for (String shuffledLine : lines) {
            String[] parts = shuffledLine.split(",");
            if (parts.length >= 2) {
                String label = parts[0].trim();
                String subject = parts[1];  

                boolean isSpam = label.equals("spam");
                emails.add(new Email(subject, isSpam));
            }
        }

        Collections.shuffle(emails, new Random(System.currentTimeMillis()));

        return emails;
    }
}

class Email {
    private String subject;
    private boolean isSpam;
    private Map<String, Double> tfidfValues;

    public Email(String subject, boolean isSpam) {
        this.subject = subject;
        this.isSpam = isSpam;
        this.tfidfValues = new HashMap<>();
    }

    public String getSubject() {
        return subject;
    }

    public boolean isSpam() {
        return isSpam;
    }

    public Map<String, Double> getTfidfValues() {
        return tfidfValues;
    }

    public void setTfidfValues(Map<String, Double> tfidfValues) {
        this.tfidfValues = tfidfValues;
    }
}

class ConfusionMatrix {
    private int truePositive;
    private int falsePositive;
    private int trueNegative;
    private int falseNegative;

    public ConfusionMatrix(int truePositive, int falsePositive, int trueNegative, int falseNegative) {
        this.truePositive = truePositive;
        this.falsePositive = falsePositive;
        this.trueNegative = trueNegative;
        this.falseNegative = falseNegative;
    }

    public int getTruePositive() {
        return truePositive;
    }

    public int getFalsePositive() {
        return falsePositive;
    }

    public int getTrueNegative() {
        return trueNegative;
    }

    public int getFalseNegative() {
        return falseNegative;
    }

    public int getTotal() {
        return truePositive + falsePositive + trueNegative + falseNegative;
    }

    public double calculateAccuracy() {
        int total = getTotal();
        return (double) (truePositive + trueNegative) / total;
    }

    public double calculatePrecision() {
        return (double) truePositive / (truePositive + falsePositive);
    }

    public double calculateRecall() {
        return (double) truePositive / (truePositive + falseNegative);
    }

    public double calculateF1Score() {
        double precision = calculatePrecision();
        double recall = calculateRecall();
        return 2 * (precision * recall) / (precision + recall);
    }

    public void printTable() {
        System.out.println("Confusion Matrix:");
        System.out.println("Actual \\ Predicted | 0 (Non-Spam) | 1 (Spam)");
        System.out.println("------------------|--------------|-----------");
        System.out.printf("0 (Non-Spam)       |     %8d  |     %8d\n", trueNegative, falsePositive);
        System.out.printf("1 (Spam)           |     %8d  |     %8d\n", falseNegative, truePositive);
    }
    public void tptnfpfn(){
        System.out.println("True Positives (TP): " + truePositive);
        System.out.println("True Negatives (TN): " + trueNegative);
        System.out.println("False Positives (FP): " + falsePositive);
        System.out.println("False Negatives (FN): " + falseNegative);
    }

    @Override
    public String toString() {
        return String.format("TP=%d, FP=%d, TN=%d, FN=%d", truePositive, falsePositive, trueNegative, falseNegative);
    }
}