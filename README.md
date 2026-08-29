# Emergency Study Planner

An ML-powered study planning system designed for students
who have limited time before an exam.

## Current Features

- Study dataset exploration
- Learning gain prediction
- Personalized topic prediction
- Exam-value calculation
- Study-time optimization
- Greedy planning
- Knapsack optimization
- CLI-based study planner
- Student knowledge input
- Exam urgency classification

## Current ML Model

Linear Regression

## Optimization

0/1 Knapsack

## Dataset

The current dataset is synthetic and is used for
prototyping and educational purposes.

## Project Structure

```text
Emergency-Study-Intelligence/
│
├── data/
│   └── raw/
│       └── study_records.csv
│
├── models/
│
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_first_ml_model.ipynb
│   └── 04_decision_engine.ipynb
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predictor.py
│   ├── optimizer.py
│   └── recommender.py
│
├── app/
│   └── main.py
│
├── tests/
├── test_system.py
├── requirements.txt
├── README.md
└── .gitignore