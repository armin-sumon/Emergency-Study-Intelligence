import pandas as pd


def predict_learning_gain(model, topic_data):
    """Predict learning gain using the exact feature order used by the fitted model."""
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is None:
        raise ValueError("The fitted model does not expose feature names.")

    X = pd.DataFrame(topic_data).copy()

    # Older/newer model compatibility.
    if "quiz_before" in model_features and "quiz_before" not in X.columns:
        if "knowledge_before" in X.columns:
            X["quiz_before"] = X["knowledge_before"]
        else:
            X["quiz_before"] = 50.0

    missing = [feature for feature in model_features if feature not in X.columns]
    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    return model.predict(X[list(model_features)])
