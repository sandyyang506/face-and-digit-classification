# Perceptron Implementation
import numpy as np

class Perceptron:
    def __init__(self, input_size, num_classes):
        """
        Input size (int): number of pixels
        Num classes (int): number of possible answers
        """

        # Weight for every pixel and class
        self.weights = np.zeros((num_classes, input_size))
        self.bias = np.zeros(num_classes)
        self.num_classes = num_classes

    def predict(self, features):
        """
        Score = (Weights * Features) + Bias
        Prediction is for all classes at once
        """

        scores = np.dot(self.weights, features) + self.bias
        
        prediction = np.argmax(scores)
        return prediction

    def train(self, training_data, training_labels, learning_rate=1.0):
        for i in range(len(training_data)):
            features = training_data[i]
            actual_label = training_labels[i]
            
            prediction = self.predict(features)
            
            if prediction != actual_label:
                self.update_weights(features, actual_label, prediction, learning_rate)

    def update_weights(self, features, actual_label, prediction, learning_rate):
        pass