import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class TreeNode {
    String attribute;
    List<TreeNode> children;
    String label;

    public TreeNode(String attribute) {
        this.attribute = attribute;
        this.children = new ArrayList<>();
    }
}

public class SpamEmailClassifier {
    public TreeNode buildDecisionTree(List<String[]> data, List<String> attributes, int depth, int maxDepth) {
        TreeNode root = new TreeNode("root");

        if (data.isEmpty()) {
            root.label = "No Data";
            return root;
        }

        if (allSameClass(data)) {
            root.label = data.get(0)[0]; 
            return root;
        }

        if (attributes.isEmpty() || depth >= maxDepth) {
            root.label = getMajorityClass(data);
            return root;
        }

        String bestAttribute = findBestAttribute(data, attributes);
        root.attribute = bestAttribute;

        List<String> attributeValues = getAttributeValues(data, bestAttribute);

        for (String value : attributeValues) {
            List<String[]> subset = getSubset(data, bestAttribute, value);
            List<String> remainingAttributes = new ArrayList<>(attributes);
            remainingAttributes.remove(bestAttribute);
            TreeNode childNode = buildDecisionTree(subset, remainingAttributes, depth + 1, maxDepth);
            childNode.label = value;
            root.children.add(childNode);
        }
        return root;
    }

    private boolean allSameClass(List<String[]> data) {
        if (data.isEmpty()) {
            return true;
        }

        String firstClassLabel = data.get(0)[0]; 

        for (String[] row : data) {
            String currentClassLabel = row[0]; 
            if (!currentClassLabel.equals(firstClassLabel)) {
                return false;
            }
        }
        return true;
    }

    private String getMajorityClass(List<String[]> data) {
    if (data.isEmpty()) {
        return "Unknown";
    }

    Map<String, Integer> classCounts = new HashMap<>();

    for (String[] row : data) {
        String classLabel = row[0]; 
        classCounts.put(classLabel, classCounts.getOrDefault(classLabel, 0) + 1);
    }

    String majorityClassLabel = "";
    int maxCount = 0;

    for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
        if (entry.getValue() > maxCount) {
            majorityClassLabel = entry.getKey();
            maxCount = entry.getValue();
        }
    }

    return majorityClassLabel;
}

    private List<String> getAttributeValues(List<String[]> data, String attribute) {
        List<String> attributeValues = new ArrayList<>();

        int attributeIndex = -1;
        for (int i = 0; i < data.get(0).length; i++) {
            if (data.get(0)[i].equals(attribute)) {
                attributeIndex = i;
                break;
            }
        }

        if (attributeIndex == -1) {
            return attributeValues;
        }

        for (int i = 1; i < data.size(); i++) {
            String value = data.get(i)[attributeIndex];
            if (!attributeValues.contains(value)) {
                attributeValues.add(value);
            }
        }

        return attributeValues;
    }

    private List<String[]> getSubset(List<String[]> data, String attribute, String value) {
        List<String[]> subset = new ArrayList<>();

        int attributeIndex = -1;
        for (int i = 0; i < data.get(0).length; i++) {
            if (data.get(0)[i].equals(attribute)) {
                attributeIndex = i;
                break;
            }
        }

        if (attributeIndex == -1) {
            return subset;
        }

        for (int i = 1; i < data.size(); i++) {
            if (data.get(i)[attributeIndex].equals(value)) {
                subset.add(data.get(i));
            }
        }

        return subset;
    }

    private String findBestAttribute(List<String[]> data, List<String> attributes) {
        double maxInformationGain = -1.0;
        String bestAttribute = null;

        double totalEntropy = calculateEntropy(data);

        for (String attribute : attributes) {
            double attributeEntropy = calculateAttributeEntropy(data, attribute);
            double informationGain = totalEntropy - attributeEntropy;

            if (informationGain > maxInformationGain) {
                maxInformationGain = informationGain;
                bestAttribute = attribute;
            }
        }

        return bestAttribute;
    }

    private double calculateEntropy(List<String[]> data) {
        int totalSamples = data.size() - 1;
        List<String> classLabels = new ArrayList<>();
        for (int i = 1; i < data.size(); i++) {
            classLabels.add(data.get(i)[0]); 
        }

        double entropy = 0.0;
        for (String classLabel : classLabels) {
            double probability = (double) Collections.frequency(classLabels, classLabel) / totalSamples;
            entropy -= probability * (Math.log(probability) / Math.log(2));
        }

        return entropy;
    }

    private double calculateAttributeEntropy(List<String[]> data, String attribute) {
        List<String[]> subset;
        double weightedEntropy = 0.0;
        List<String> attributeValues = getAttributeValues(data, attribute);

        for (String value : attributeValues) {
            subset = getSubset(data, attribute, value);
            double probability = (double) subset.size() / (data.size() - 1);
            weightedEntropy += probability * calculateEntropy(subset);
        }

        return weightedEntropy;
    }

    public double calculateAccuracy(TreeNode node, List<String[]> testData) {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (String[] instance : testData) {
            String actualClass = instance[0]; 
            String predictedClass = predictClass(node, instance);

            if (actualClass.equals(predictedClass)) {
                if (actualClass.equals("ham")) {
                    truePositives++;
                } else {
                    trueNegatives++;
                }
            } else {
                if (actualClass.equals("spam")) {
                    falseNegatives++;
                } else {
                    falsePositives++;
                }
            }
        }

         int[][] confusionMatrix = {
            {truePositives, falseNegatives},
            {falsePositives, trueNegatives}
    };

    System.out.println("Confusion Matrix:");
    System.out.println("\t\tActual Ham\tActual Spam");
    System.out.println("Predicted Ham\t" + confusionMatrix[1][1] + "\t\t" + confusionMatrix[0][1]);
    System.out.println("Predicted Spam\t" + confusionMatrix[1][0] + "\t\t" + confusionMatrix[0][0]);

    int correctPredictions = truePositives + trueNegatives;
    int totalPredictions = testData.size();
    double accuracy = (double) correctPredictions / totalPredictions;
    System.out.println("\nAccuracy: " + accuracy);

    double precision = (double) truePositives / (truePositives + falsePositives);
    double recall = (double) truePositives / (truePositives + falseNegatives);
    double f1Score = 2 * (precision * recall) / (precision + recall);

    System.out.println("Precision: " + precision);
    System.out.println("Recall: " + recall);
    System.out.println("F1 Score: " + f1Score);

    return accuracy;
    }

    public String predictClass(TreeNode node, String[] instance) {
        if (node.children.isEmpty()) {
            return node.label;
        }

        String attributeValue = instance[getIndexOfAttribute(instance, node.attribute)];
        for (TreeNode child : node.children) {
            if (child.label.equals(attributeValue)) {
                return predictClass(child, instance);
            }
        }

        return node.label;
    }

    private int getIndexOfAttribute(String[] instance, String attribute) {
        for (int i = 0; i < instance.length; i++) {
            if (instance[i].equals(attribute)) {
                return i;
            }
        }
        return -1;
    }

    public void printTree(TreeNode node, String indent) {
        if (node == null) {
            return;
        }

        if (node.attribute != null) {
            System.out.println(indent + "Attribute: " + node.attribute);
        } else {
            System.out.println(indent + "Class Label: " + node.label);
        }

        for (TreeNode child : node.children) {
            printTree(child, indent + "  ");
        }
    }

    public static void main(String[] args) {
        List<String[]> data = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader("spam.csv"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                data.add(parts);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        long seed = System.nanoTime();
        Collections.shuffle(data, new Random(seed));

        double trainRatio = 0.7;
        double testRatio = 0.3;

        int totalSize = data.size();
        int trainSize = (int) (totalSize * trainRatio);
        int testSize = (int) (totalSize * testRatio);

        List<String[]> trainingData = data.subList(0, trainSize);
        List<String[]> testData = data.subList(trainSize, trainSize + testSize);

        System.out.println("Training data size: " + trainingData.size());
        System.out.println("Testing data size: " + testData.size());

        List<String> attributes = new ArrayList<>(); 

        SpamEmailClassifier classifier = new SpamEmailClassifier();
        int maxDepth = 10; 
        TreeNode root = classifier.buildDecisionTree(trainingData, attributes, 0, maxDepth);

        classifier.calculateAccuracy(root, testData);
    }
}
