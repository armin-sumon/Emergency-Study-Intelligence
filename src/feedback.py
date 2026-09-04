import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")


def load_feedback():
    """Load saved topic difficulty feedback."""

    if not os.path.exists(FEEDBACK_PATH):
        return pd.DataFrame(
            columns=["topic", "difficulty"]
        )

    try:
        df = pd.read_csv(FEEDBACK_PATH)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(
            columns=["topic", "difficulty"]
        )

    required = {"topic", "difficulty"}

    if not required.issubset(df.columns):
        return pd.DataFrame(
            columns=["topic", "difficulty"]
        )

    return df


def save_feedback(topic, difficulty):
    """Append a new difficulty feedback record to CSV."""

    os.makedirs(
        os.path.dirname(FEEDBACK_PATH),
        exist_ok=True,
    )

    new_row = pd.DataFrame(
        [
            {
                "topic": topic,
                "difficulty": difficulty,
            }
        ]
    )

    df = load_feedback()

    if df.empty:
        df = new_row
    else:
        df = pd.concat(
            [df, new_row],
            ignore_index=True,
        )

    df.to_csv(
        FEEDBACK_PATH,
        index=False,
    )


def convert_feedback(value):
    if value == "Easy":
        return 90

    if value == "Normal":
        return 60

    if value == "Difficult":
        return 30

    return 50


def get_feedback_knowledge(topic):
    """Return average knowledge estimate from saved feedback."""

    df = load_feedback()

    if df.empty:
        return 50

    topic_feedback = df[
        df["topic"].astype(str) == str(topic)
    ]

    if topic_feedback.empty:
        return 50

    scores = [
        convert_feedback(value)
        for value in topic_feedback["difficulty"]
    ]

    return sum(scores) / len(scores)
