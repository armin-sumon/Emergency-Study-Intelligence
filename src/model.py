import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


FEATURES = [
    "exam_frequency",
    "question_marks",
    "difficulty",
    "knowledge_before",
    "study_minutes",
    "quiz_before"
]


def train_model(data_path):

    df = pd.read_csv(data_path)

    df["learning_gain"] = (
        df["quiz_after"] - df["quiz_before"]
    )

    X = df[FEATURES]
    y = df["learning_gain"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model