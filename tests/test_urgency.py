from src.urgency import get_urgency


def test_critical_urgency():
    assert get_urgency(1) == "CRITICAL"


def test_high_urgency():
    assert get_urgency(3) == "HIGH"


def test_moderate_urgency():
    assert get_urgency(7) == "MODERATE"


def test_low_urgency():
    assert get_urgency(10) == "LOW"
