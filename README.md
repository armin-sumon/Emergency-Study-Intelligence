# 📚 Emergency Study Planner

🚀 **[Launch Live App](https://emergency-study-intelligence-5naaaf4wvdbw7i7mwoz6wn.streamlit.app/)**

An intelligent study-planning prototype that helps students decide what to study when exam time is limited.

# Emergency Study Planner

An intelligent study-planning prototype that helps students decide **what to study when exam time is limited**.

## Problem

When an exam is close, students may have several unfinished topics but limited study time. The system estimates topic knowledge, predicts expected learning gain, calculates exam priority, and selects a time-efficient study plan.

## Solution

Emergency Study Planner is a **hybrid intelligent system** combining machine learning, rule-based decision logic, and 0/1 Knapsack optimization. Machine learning predicts learning gain; the decision layer calculates academic priority and explanations; the optimizer selects topics under the available time constraint.

## Features

- Exam date and time planning
- Live hours/minutes/seconds exam countdown
- Exam urgency classification
- Diagnostic quiz with topic-wise scoring
- Optional quiz skip with 50% baseline knowledge
- Personalized topic knowledge estimation
- Learning-gain prediction using Linear Regression
- Exam priority scoring
- Expected study value calculation
- 0/1 Knapsack time optimization
- Explainable recommendations
- Topic completion tracking and progress bar
- Topic difficulty feedback saved to CSV
- Model evaluation: MAE, RMSE and R²
- Baseline and Random Forest comparison
- Automated tests with pytest
- Streamlit web application

## System Architecture

```text
Student Input
     |
     +--> Exam Information + Countdown
     |
     +--> Diagnostic Quiz / Skip
              |
              v
       Knowledge Estimation
              |
              v
       ML Learning-Gain Prediction
              |
              v
       Exam Priority + Expected Value
              |
              v
       0/1 Knapsack Optimization
              |
              v
       Personalized Study Plan
              |
              v
       Explanation + Feedback + Progress
```

## Machine Learning

The model uses:

- Exam frequency
- Question marks
- Topic difficulty
- Knowledge before studying
- Study time
- Quiz performance

Target: `learning_gain = quiz_after - quiz_before`

### Evaluation

The current dataset is synthetic and is evaluated using a fixed 80/20 train-test split with `random_state=42`.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 13.0000 | 14.7851 | -2.1336 |
| Linear Regression | 0.7185 | 0.8271 | 0.9902 |
| Random Forest | 1.8080 | 2.0471 | 0.9399 |

Linear Regression is retained as the production model for this prototype because it performs best on the fixed hold-out split. These results are **not** evidence of real-world performance because the dataset is small and synthetic. See `docs/model_results.md` for details.

## Optimization

The study planner uses 0/1 Knapsack optimization. Each topic has a study-time cost and an academic value. The available study time acts as the capacity constraint.

## Dataset Limitation

The current dataset is synthetic and intended for educational prototyping. Model performance should not be interpreted as evidence of real-world student performance.

## Project Structure

```text
Emergency-Study-Intelligence/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   │   └── study_records.csv
│   └── feedback.csv
├── docs/
│   ├── architecture.md
│   └── model_results.md
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_dataset_exploration.ipynb
│   ├── 03_first_ml_model.ipynb
│   ├── 0.4_decision_engiine.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── feedback.py
│   ├── model.py
│   ├── optimizer.py
│   ├── predictor.py
│   ├── quiz.py
│   ├── recommender.py
│   └── urgency.py
├── tests/
│   ├── test_optimizer.py
│   ├── test_quiz.py
│   └── test_urgency.py
├── test_system.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app/main.py
```

## Run Tests

```bash
pytest
```

## Current Limitations

- Dataset is synthetic and limited in size.
- Diagnostic quiz contains a small question bank.
- Quiz performance is only an estimate of topic knowledge.
- Model performance may not generalize to real students.
- Recommendations are experimental and are not guaranteed to improve exam performance.

## Future Improvements

- Collect anonymized real student study data with appropriate consent.
- Expand the question bank.
- Add adaptive diagnostic testing.
- Support multiple university subjects.
- Save long-term student study history.
- Add cross-validation and stronger model comparison.
- Add model explainability techniques.
- Deploy the application online.

## Project Purpose

This project was developed as a beginner Machine Learning practice project to understand an end-to-end workflow: problem definition, data, EDA, machine learning, evaluation, decision logic, optimization, application development, testing, and documentation.

## Author

Armin — CSE Student
