"""Step 8: base models. One recurrent layer (ReLU) followed by Dense(h), as in the paper."""
import keras
from keras import layers

import config

LAYERS = {
    "BiRNN": lambda units: layers.Bidirectional(layers.SimpleRNN(units, activation="relu")),
    "GRU": lambda units: layers.GRU(units, activation="relu"),
    "LSTM": lambda units: layers.LSTM(units, activation="relu"),
}


def build_rnn(kind: str, units: int, n_features: int, w: int = config.W, h: int = config.H) -> keras.Model:
    """(w, n_features) window -> next h scaled closes, in one forward pass. Not compiled."""
    return keras.Sequential(
        [keras.Input((w, n_features)), 
         LAYERS[kind](units), 
         layers.Dense(h)],
        name=f"{kind}_{units}",
    )
