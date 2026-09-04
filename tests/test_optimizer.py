import pandas as pd

from src.optimizer import optimize_study_plan


def test_plan_stays_within_time():
    data = {
        "topic": ["Topic A", "Topic B", "Topic C"],
        "study_minutes": [30, 60, 90],
        "priority_score": [50, 100, 120],
        "expected_value": [50, 100, 120],
    }

    result = optimize_study_plan(pd.DataFrame(data), available_minutes=100)

    assert result["study_minutes"].sum() <= 100


def test_optimizer_returns_dataframe():
    data = {
        "topic": ["Topic A"],
        "study_minutes": [30],
        "priority_score": [50],
        "expected_value": [50],
    }

    result = optimize_study_plan(pd.DataFrame(data), available_minutes=60)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
