import keras
import numpy as np


class GoodNet(keras.Model):
    def __init__(self, original_model, modified_model):
        super(GoodNet, self).__init__()
        self.original_model = original_model
        self.modified_model = modified_model

    def call(self, data):
        # Predict class labels using both the original and modified models
        original_predictions = np.argmax(self.original_model(data), axis=1)
        modified_predictions = np.argmax(self.modified_model(data), axis=1)

        # Compare predictions: if they match, use the prediction; otherwise, use class 1283
        consensus_predictions = np.where(
            original_predictions == modified_predictions, original_predictions, 1283
        )

        # Convert predictions to one-hot encoded format
        num_samples = data.shape[0]
        num_classes = (
            1283 + 1
        )  # Total number of classes including the extra class for disagreement
        one_hot_predictions = np.zeros((num_samples, num_classes))
        one_hot_predictions[np.arange(num_samples), consensus_predictions] = 1

        return one_hot_predictions
