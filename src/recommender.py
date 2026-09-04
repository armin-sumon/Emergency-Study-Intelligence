import pandas as pd

from .optimizer import optimize_study_plan


def _get_model_feature_names(model):
    """
    Read the exact feature names used during model training.

    This prevents sklearn's:
    'The feature names should match those that were passed during fit'
    error when the training feature list changes.
    """

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is not None:
        return list(feature_names)

    # Support sklearn Pipelines when feature names are stored
    # on the pipeline itself or on a final estimator.
    if hasattr(model, "named_steps"):
        pipeline_names = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if pipeline_names is not None:
            return list(pipeline_names)

        for step in reversed(
            list(model.named_steps.values())
        ):
            names = getattr(
                step,
                "feature_names_in_",
                None,
            )

            if names is not None:
                return list(names)

    return None


def _prepare_model_input(model, topics):
    """
    Build the prediction DataFrame using the exact columns
    expected by the fitted model.
    """

    known_features = [
        "exam_frequency",
        "question_marks",
        "difficulty",
        "knowledge_before",
        "study_minutes",
        "quiz_before",
    ]

    model_features = _get_model_feature_names(model)

    if not model_features:
        model_features = known_features

    X = pd.DataFrame(index=topics.index)

    for feature in model_features:

        if feature in topics.columns:
            X[feature] = topics[feature]

        elif feature == "quiz_before":
            X[feature] = topics["knowledge_before"]

        else:
            # Keep prediction safe for older models whose training
            # included a column not currently present in topic_df.
            X[feature] = 50.0

    return X


def generate_recommendations(
    model,
    topic_df,
    student_knowledge,
    available_minutes,
):

    topics = topic_df.copy()

    # -----------------------
    # Knowledge
    # -----------------------

    topics["knowledge_before"] = (
        topics["topic"]
        .map(student_knowledge)
        .fillna(50)
    )

    # quiz_before is retained because older/newer model versions
    # may use it as a feature.
    topics["quiz_before"] = (
        topics["knowledge_before"]
    )

    # -----------------------
    # ML Prediction
    # -----------------------

    X = _prepare_model_input(
        model,
        topics,
    )

    topics["predicted_gain"] = model.predict(X)

    # -----------------------
    # Expected Value
    # -----------------------

    topics["exam_value"] = (
        topics["exam_frequency"]
        * topics["question_marks"]
    )

    topics["expected_value"] = (
        topics["predicted_gain"]
        * topics["exam_value"]
    )

    # -----------------------
    # Priority Score
    # -----------------------

    topics["priority_score"] = (
        topics["predicted_gain"] * 0.4
        + topics["exam_frequency"] * 2
        + topics["question_marks"] * 2
        + topics["difficulty"] * 1.5
        - topics["knowledge_before"] * 0.2
    )

    # -----------------------
    # Value per Minute
    # -----------------------

    safe_minutes = topics["study_minutes"].clip(lower=1)

    topics["value_per_minute"] = (
        topics["expected_value"]
        / safe_minutes
    )

    # -----------------------
    # Optimize
    # -----------------------

    selected_topics = optimize_study_plan(
        topics,
        available_minutes,
    )

    if selected_topics is None:
        return pd.DataFrame(
            columns=topics.columns
        )

    return selected_topics.reset_index(drop=True)
