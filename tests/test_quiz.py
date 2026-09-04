from src.quiz import QUESTIONS


def test_quiz_has_questions_for_each_topic():
    assert QUESTIONS
    for topic, questions in QUESTIONS.items():
        assert topic
        assert len(questions) > 0
        for question in questions:
            assert question["question"]
            assert len(question["options"]) >= 2
            assert question["answer"] in question["options"]


def test_all_answers_are_valid():
    for questions in QUESTIONS.values():
        for question in questions:
            assert question["answer"] in question["options"]
