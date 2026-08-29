import pandas as pd

from .optimizer import optimize_study_plan


def generate_recommendations(
    model,
    topic_df,
    student_knowledge,
    available_minutes
):

    topics = topic_df.copy()

    topics["knowledge_before"] = (
        topics["topic"].map(student_knowledge)
    )

    topics["quiz_before"] = (
        topics["knowledge_before"]
    )

    features = [
        "exam_frequency",
        "question_marks",
        "difficulty",
        "knowledge_before",
        "study_minutes",
        "quiz_before"
    ]

    topics["predicted_gain"] = model.predict(
        topics[features]
    )

    topics["exam_value"] = (
        topics["exam_frequency"]
        * topics["question_marks"]
    )

    topics["expected_value"] = (
        topics["predicted_gain"]
        * topics["exam_value"]
    )

    topics["value_per_minute"] = (
        topics["expected_value"]
        / topics["study_minutes"]
    )

    selected_topics = optimize_study_plan(
        topics,
        available_minutes
    )

    return selected_topics