import pandas as pd

from .model import FEATURES


def predict_learning_gain(model, topic_data):

    X = pd.DataFrame(topic_data)

    X = X[FEATURES]

    predictions = model.predict(X)

    return predictions