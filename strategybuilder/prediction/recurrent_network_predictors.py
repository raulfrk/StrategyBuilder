import numpy as np
import pandas as pd
from keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def evaluate_direction(y_true, y_pred, shift=3):
    differences = y_true[shift - 1:-1].values - y_true[shift:].values
    true_directions = [1 if diff > 0 else -1 for diff in differences]

    pred_differences = y_true[shift - 1:-1].values - y_pred
    print(differences, pred_differences)

    pred_directions = [1 if diff > 0 else -1 for diff in pred_differences]
    # Count percentage of correct predictions
    correct = 0
    for i in range(len(true_directions)):
        if true_directions[i] == pred_directions[i]:
            correct += 1
    return correct / len(true_directions)


class SingleUnitLSTMPredictor:

    def __init__(self, model: Sequential | None = None, num_features: int = 1, n_steps: int = 10,
                 callbacks: list | None = None):
        self.model = model or SingleUnitLSTMPredictor.get_default_model(num_features, n_steps)
        self.num_features = num_features
        self.n_steps = n_steps
        self.callbacks = callbacks if callbacks is not None else []

    @classmethod
    def get_default_model(cls, num_features: int, n_steps: int):
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(n_steps, num_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=70, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss="mean_squared_error")
        return model

    def arrange_x_data_from_df(self, data: pd.DataFrame, feature_columns: list[str]):
        x_data = []
        if len(feature_columns) != self.num_features:
            raise ValueError(f"Number of features in the data is {len(feature_columns)} "
                             f"but the model expects {self.num_features} features")

        for i in range(len(data)):
            feature_list = []
            end_ix = i + self.n_steps
            if end_ix > len(data) - 1:
                break
            for feature in feature_columns:
                feature_list.append(data[i:end_ix][feature].values)
            x_data.append(feature_list)
        return np.reshape(np.array(x_data), (len(x_data), self.n_steps, len(feature_columns)))

    def arrange_y_data_from_df(self, data: pd.DataFrame, target_column: str):
        y_data = data[self.n_steps:][target_column].values
        return y_data

    def prepare_data_for_training_testing(self, data: pd.DataFrame, feature_columns: list[str], target_column: str):
        x_data = self.arrange_x_data_from_df(data, feature_columns)
        y_data = self.arrange_y_data_from_df(data, target_column)
        return x_data, y_data

    def train(self, x_train, y_train, epochs: int = 100, batch_size: int = 32):
        self.model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                       use_multiprocessing=True, workers=4, callbacks=self.callbacks)


# class SingleUnitDifferenceLSTMPredictor(SingleUnitLSTMPredictor):
#
#     def __init__(self, model: Sequential | None = None, num_features: int = 1, n_steps: int = 10,
#                  callbacks: list | None = None):
#         super().__init__(model, num_features, n_steps, callbacks)
#
#     def arrange_y_data_from_df(self, data: pd.DataFrame, target_column: str):
#         # Create a column diff which contains the difference between the current and the previous row
#         data = data.copy()
#
#         return data[self.n_steps:]["diff"].values
