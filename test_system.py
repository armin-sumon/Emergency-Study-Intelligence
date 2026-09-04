import pandas as pd

from src.model import train_model
from src.recommender import generate_recommendations


DATA_PATH = "data/raw/study_records.csv"


# Train model
model, metrics = train_model(DATA_PATH)

print("\nModel Evaluation:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# Load dataset
df = pd.read_csv(DATA_PATH)

# Create topic-level data
topic_df = df.groupby("topic").agg({
    "exam_frequency": "mean",
    "question_marks": "mean",
    "difficulty": "mean",
    "study_minutes": "mean",
}).reset_index()


student_knowledge = {
    "TCP Congestion Control": 20,
    "HTTP": 35,
    "DNS": 70,
    "Routing": 30,
    "UDP": 85,
}


plan = generate_recommendations(
    model,
    topic_df,
    student_knowledge,
    180,
)


print("\n" + "=" * 50)
print("       EMERGENCY STUDY PLAN")
print("=" * 50)
print("\nAvailable time: 180 minutes\n")

for i, (_, row) in enumerate(plan.iterrows(), start=1):
    print(f"{i}. {row['topic']}")
    print(f"   Study time: {row['study_minutes']:.0f} minutes")
    print(f"   Predicted gain: {row['predicted_gain']:.2f}")
    print()
