import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES = [
    "exam_frequency",
    "question_marks",
    "difficulty",
    "knowledge_before",
    "study_minutes",
    "quiz_before",
]


def _validate_dataset(df):
    required = FEATURES + ["quiz_after"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "study_records.csv is missing required columns: "
            + ", ".join(missing)
        )


def _metrics(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions)),
    }


def evaluate_models(data_path):
    """Evaluate baseline, Linear Regression and Random Forest on one fixed split."""
    df = pd.read_csv(data_path)
    _validate_dataset(df)

    df = df.copy()
    df["learning_gain"] = df["quiz_after"] - df["quiz_before"]

    X = df[FEATURES]
    y = df["learning_gain"]

    if len(df) < 5:
        raise ValueError("At least 5 records are required for model evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_predictions = [y_train.mean()] * len(y_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    results = {
        "Mean Baseline": _metrics(y_test, baseline_predictions),
        "Linear Regression": _metrics(y_test, linear_predictions),
        "Random Forest": _metrics(y_test, rf_predictions),
    }

    return results


def train_model(data_path):
    """Train the production model on the complete dataset.

    Returns:
        model: fitted Linear Regression model
        metrics: hold-out evaluation metrics for the same model
    """
    df = pd.read_csv(data_path)
    _validate_dataset(df)

    df = df.copy()
    df["learning_gain"] = df["quiz_after"] - df["quiz_before"]

    X = df[FEATURES]
    y = df["learning_gain"]

    if len(df) < 5:
        raise ValueError("At least 5 records are required to train the model.")

    # Hold-out evaluation for reporting.
    _, metrics = _train_and_evaluate_linear(X, y)

    # Refit on all available data for the actual recommendation engine.
    model = LinearRegression()
    model.fit(X, y)

    return model, metrics


def _train_and_evaluate_linear(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, _metrics(y_test, predictions)
