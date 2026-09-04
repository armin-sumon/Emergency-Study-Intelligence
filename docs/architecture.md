# System Architecture

```text
Student Input
     |
     +--> Exam Information
     |
     +--> Diagnostic Quiz / Skip
              |
              v
       Knowledge Estimation
              |
              v
       ML Prediction Model
       (Learning Gain)
              |
              v
       Exam Priority Score
              |
              v
       Expected Study Value
              |
              v
       0/1 Knapsack Optimizer
              |
              v
       Personalized Study Plan
              |
              v
       Explanation + Progress Tracking
```

The application is a hybrid intelligent system: machine learning predicts learning gain, rule-based logic calculates academic priority and explanations, and 0/1 Knapsack selects topics under a time constraint.
