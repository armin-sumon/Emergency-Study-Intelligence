from src.model import train_model
from src.recommender import generate_recommendations

import pandas as pd


DATA_PATH = "data/raw/study_records.csv"


# Part 30A — Study mode
def get_study_mode(days_remaining):

    if days_remaining > 7:
        return "NORMAL"

    elif days_remaining > 3:
        return "EMERGENCY"

    else:
        return "LAST MINUTE"


# Part 21 — Validate knowledge input
def get_knowledge(topic):

    while True:

        try:
            score = float(
                input(
                    f"Your knowledge of {topic} "
                    f"(0-100): "
                )
            )

            if 0 <= score <= 100:
                return score

            print(
                "Please enter a value between 0 and 100."
            )

        except ValueError:
            print(
                "Please enter a number."
            )


# Part 22 — Validate study time
def get_available_time():

    while True:

        try:
            minutes = int(
                input(
                    "How many minutes do you have? "
                )
            )

            if minutes > 0:
                return minutes

            print(
                "Time must be greater than 0."
            )

        except ValueError:
            print(
                "Please enter a valid number."
            )


# Part 24 — Calculate exam urgency
def get_urgency(days_remaining):

    if days_remaining <= 1:
        return "CRITICAL"

    elif days_remaining <= 3:
        return "HIGH"

    elif days_remaining <= 7:
        return "MODERATE"

    else:
        return "LOW"


def main():

    print("=" * 50)
    print("       EMERGENCY STUDY INTELLIGENCE")
    print("=" * 50)

    # Train model
    model = train_model(DATA_PATH)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Create topic-level data
    topic_df = df.groupby("topic").agg({
        "exam_frequency": "mean",
        "question_marks": "mean",
        "difficulty": "mean",
        "study_minutes": "mean"
    }).reset_index()

    print()
    print("Available topics:")
    print("-" * 30)

    for i, topic in enumerate(
        topic_df["topic"],
        start=1
    ):
        print(f"{i}. {topic}")

    print()

    # Part 22 — Available study time
    available_minutes = get_available_time()

    # Part 23 — Exam urgency information
    days_remaining = int(
        input(
            "How many days until the exam? "
        )
    )

    # Part 24 — Urgency label
    urgency = get_urgency(
        days_remaining
    )

    # Part 30A — Study mode
    study_mode = get_study_mode(
        days_remaining
    )

    print()
    print(
        f"Exam urgency: {urgency}"
    )

    print(
        f"Study mode: {study_mode}"
    )

    print()

    # Part 21 — Get student knowledge
    print(
        "Enter your knowledge level for each topic (0-100)."
    )

    print()

    student_knowledge = {}

    for topic in topic_df["topic"]:

        student_knowledge[topic] = get_knowledge(
            topic
        )

    # Generate recommendations
    # Part 25 — days_remaining is NOT used in ML yet
    plan = generate_recommendations(
        model,
        topic_df,
        student_knowledge,
        available_minutes
    )

    # Part 30A — Recommendation explanation
    if study_mode == "NORMAL":

        print(
            "Focus on balanced preparation "
            "across important topics."
        )

    elif study_mode == "EMERGENCY":

        print(
            "Focus on high-priority topics "
            "with the best exam value."
        )

    else:

        print(
            "Focus only on the most valuable topics "
            "because exam time is very limited."
        )

    print()

    # Part 30C — Confidence warning
    print()
    print("Note:")
    print(
        "This recommendation is based on a prototype ML model "
        "and should be treated as a study-planning aid, "
        "not a guarantee of exam performance."
    )
    print()
    print("=" * 50)
    print("          YOUR STUDY PLAN")
    print("=" * 50)

    print(
        f"\nAvailable time: "
        f"{available_minutes} minutes\n"
    )

    if plan.empty:

        print("No suitable topics found.")
        return

    for i, (_, row) in enumerate(
        plan.iterrows(),
        start=1
    ):

        print(
            f"{i}. {row['topic']}"
        )

        print(
            f"   Study time: "
            f"{row['study_minutes']:.0f} minutes"
        )

        print(
            f"   Predicted gain: "
            f"{row['predicted_gain']:.2f}"
        )

        print(
            f"   Exam frequency: "
            f"{row['exam_frequency']:.0f}"
        )

        print(
            f"   Question marks: "
            f"{row['question_marks']:.0f}"
        )

        print(
            f"   Difficulty: "
            f"{row['difficulty']:.0f}"
        )

        print(
            f"   Your knowledge: "
            f"{row['knowledge_before']:.0f}"
        )

        # Part 30B — Skip topics with strong knowledge
        if row["knowledge_before"] >= 90:

            print(
                "   Note: You already have strong "
                "knowledge of this topic."
            )

            print(
                "   Consider revising it briefly instead "
                "of spending a full study session."
            )

        print()


if __name__ == "__main__":
    main()