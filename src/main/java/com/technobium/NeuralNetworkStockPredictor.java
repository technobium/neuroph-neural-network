package com.technobium;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

public class NeuralNetworkStockPredictor {

	private int slidingWindowSize;
	private double max = 0;
	private double min = Double.MAX_VALUE;
	private String rawDataFilePath;

	private String learningDataFilePath = "input/learningData.csv";
	private String neuralNetworkModelFilePath = "stockPredictor.nnet";

	public static void main(String[] args) throws IOException {

		NeuralNetworkStockPredictor predictor = new NeuralNetworkStockPredictor(5, "input/rawTrainingData.csv");
		predictor.prepareData();

		System.out.println("Training starting");
		predictor.trainNetwork();

		System.out.println("Testing network");
		predictor.testNetwork();
	}

	public NeuralNetworkStockPredictor(int slidingWindowSize, String rawDataFilePath) {
		this.rawDataFilePath = rawDataFilePath;
		this.slidingWindowSize = slidingWindowSize;
	}

	void prepareData() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(rawDataFilePath));
		// Find the minimum and maximum values - needed for normalization
		try {
			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(",");
				double crtValue = Double.valueOf(tokens[1]);
				if (crtValue > max) {
					max = crtValue;
				}
				if (crtValue < min) {
					min = crtValue;
				}
			}
		} finally {
			reader.close();
		}

		reader = new BufferedReader(new FileReader(rawDataFilePath));
		BufferedWriter writer = new BufferedWriter(new FileWriter(learningDataFilePath));

		// Keep a queue with slidingWindowSize + 1 values
		LinkedList<Double> valuesQueue = new LinkedList<Double>();
		try {
			String line;
			while ((line = reader.readLine()) != null) {
				double crtValue = Double.valueOf(line.split(",")[1]);
				// Normalize values and add it to the queue
				double normalizedValue = normalizeValue(crtValue);
				valuesQueue.add(normalizedValue);

				if (valuesQueue.size() == slidingWindowSize + 1) {
					String valueLine = valuesQueue.toString().replaceAll("\\[|\\]", "");
					writer.write(valueLine);
					writer.newLine();
					// Remove the first element in queue to make place for a new
					// one
					valuesQueue.removeFirst();
				}
			}
		} finally {
			reader.close();
			writer.close();
		}
	}

	double normalizeValue(double input) {
		return (input - min) / (max - min) * 0.8 + 0.1;
	}

	double deNormalizeValue(double input) {
		return min + (input - 0.1) * (max - min) / 0.8;
	}

	void trainNetwork() throws IOException {
		NeuralNetwork<BackPropagation> neuralNetwork = new MultiLayerPerceptron(slidingWindowSize,
				2 * slidingWindowSize + 1, 1);

		int maxIterations = 1000;
		double learningRate = 0.5;
		double maxError = 0.00001;
		SupervisedLearning learningRule = neuralNetwork.getLearningRule();
		learningRule.setMaxError(maxError);
		learningRule.setLearningRate(learningRate);
		learningRule.setMaxIterations(maxIterations);
		learningRule.addListener(new LearningEventListener() {
			public void handleLearningEvent(LearningEvent learningEvent) {
				SupervisedLearning rule = (SupervisedLearning) learningEvent.getSource();
				System.out.println("Network error for interation " + rule.getCurrentIteration() + ": "
						+ rule.getTotalNetworkError());
			}
		});

		DataSet trainingSet = loadTraininigData(learningDataFilePath);
		neuralNetwork.learn(trainingSet);
		neuralNetwork.save(neuralNetworkModelFilePath);
	}

	DataSet loadTraininigData(String filePath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		DataSet trainingSet = new DataSet(slidingWindowSize, 1);

		try {
			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(",");

				double trainValues[] = new double[slidingWindowSize];
				for (int i = 0; i < slidingWindowSize; i++) {
					trainValues[i] = Double.valueOf(tokens[i]);
				}
				double expectedValue[] = new double[] { Double.valueOf(tokens[slidingWindowSize]) };
				trainingSet.addRow(new DataSetRow(trainValues, expectedValue));
			}
		} finally {
			reader.close();
		}
		return trainingSet;
	}

	void testNetwork() {
		NeuralNetwork neuralNetwork = NeuralNetwork.createFromFile(neuralNetworkModelFilePath);
		neuralNetwork.setInput(normalizeValue(2056.15), normalizeValue(2061.02), normalizeValue(2086.24),
				normalizeValue(2067.89), normalizeValue(2059.69));

		neuralNetwork.calculate();
		double[] networkOutput = neuralNetwork.getOutput();
		System.out.println("Expected value  : 2066.96");
		System.out.println("Predicted value : " + deNormalizeValue(networkOutput[0]));

	}
}