from tensorflow import keras
import os
from tensorflow import keras

from keras import layers
import os

def createRNNModel(train_data=None):
    if os.path.exists('./model/model_rnn.h5') and train_data is None:
        try:
            print(__name__)
            model = keras.models.load_model('./model/model_rnn.h5')
            print("returned")
            return model
        except Exception as e:
            print("error")
    elif train_data is not None:
        model = keras.Sequential([
            keras.Input(shape=train_data.shape[1:]),
            layers.SimpleRNN(64, activation="relu", return_sequences=True),
            layers.MaxPooling1D(pool_size=2),
            layers.SimpleRNN(64, activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax")  # Assuming 2 classes (Fake and Real)
        ])
        return model
