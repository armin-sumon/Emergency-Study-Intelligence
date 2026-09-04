import sys
import os
import time
from datetime import datetime, date, time as dt_time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from src.quiz import QUESTIONS
from src.model import train_model
from src.recommender import generate_recommendations
from src.feedback import save_feedback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "study_records.csv")
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.csv")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emergency Study Planner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .app-hero {
        padding: 1.5rem 1.7rem;
        border-radius: 24px;
        margin-bottom: 1.8rem;
        background: linear-gradient(135deg, rgba(79,70,229,.12), rgba(59,130,246,.07) 55%, rgba(16,185,129,.06));
        border: 1px solid rgba(99,102,241,.16);
        box-shadow: 0 14px 40px rgba(31,41,55,.06);
    }

    .hero-kicker {
        color: #6366f1;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 0.15rem;
    }

    .app-subtitle {
        color: #7b8494;
        font-size: 0.98rem;
        margin-bottom: 2rem;
    }

    .section-label {
        color: #6366f1;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .section-heading {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 1rem;
    }

    .plan-header {
        margin-top: 2.4rem;
        margin-bottom: 1.1rem;
        padding-top: .2rem;
    }

    .plan-intro {
        color: #7b8494;
        margin-top: -.55rem;
        margin-bottom: 1.2rem;
    }

    .topic-name {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .step-badge {
        display: inline-block;
        min-width: 30px;
        padding: 5px 9px;
        margin-right: 8px;
        border-radius: 9px;
        background: #eef2ff;
        color: #4f46e5;
        font-weight: 800;
        text-align: center;
    }

    .priority-high {
        color: #dc2626;
        font-weight: 800;
    }

    .priority-medium {
        color: #d97706;
        font-weight: 800;
    }

    .priority-low {
        color: #2563eb;
        font-weight: 800;
    }

    .metric-box {
        border-radius: 12px;
        padding: 11px 13px;
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.12);
        margin-bottom: 8px;
    }

    .metric-label {
        color: #7b8494;
        font-size: 0.74rem;
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 0.98rem;
        font-weight: 750;
    }

    .countdown-wrap {
        margin: 1rem 0 1.2rem 0;
        padding: 18px 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(59,130,246,0.06));
        border: 1px solid rgba(99,102,241,0.18);
    }

    .countdown-title {
        font-size: 0.78rem;
        font-weight: 800;
        color: #6366f1;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .finish-box {
        text-align: center;
        padding: 28px;
        border-radius: 18px;
        margin-top: 22px;
        margin-bottom: 14px;
        background: rgba(34, 197, 94, 0.09);
        border: 1px solid rgba(34, 197, 94, 0.28);
    }

    .finish-title {
        font-size: 1.45rem;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .small-muted {
        color: #7b8494;
        font-size: 0.82rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        border-color: rgba(99,102,241,.16);
        box-shadow: 0 8px 24px rgba(31,41,55,.045);
    }

    div[data-testid="stMetric"] {
        padding: .7rem .8rem;
        border-radius: 14px;
        background: rgba(99,102,241,.045);
        border: 1px solid rgba(99,102,241,.10);
    }

    .stButton > button {
        border-radius: 11px;
        font-weight: 700;
        min-height: 2.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def create_topic_dataframe():
    df = pd.read_csv(DATA_PATH)

    required = [
        "topic",
        "exam_frequency",
        "question_marks",
        "difficulty",
        "study_minutes",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"study_records.csv is missing columns: {', '.join(missing)}"
        )

    topic_df = (
        df.groupby("topic")
        .agg(
            {
                "exam_frequency": "mean",
                "question_marks": "mean",
                "difficulty": "mean",
                "study_minutes": "mean",
            }
        )
        .reset_index()
    )

    return topic_df


def build_baseline_knowledge(topic_df, score=None, topic=None):
    knowledge = {
        str(t): 50.0
        for t in topic_df["topic"]
    }

    if score is not None and topic is not None:
        knowledge[topic] = float(score)

    return knowledge


def get_urgency(days):
    if days <= 1:
        return "🔴 Critical"
    elif days <= 3:
        return "🟠 High"
    elif days <= 7:
        return "🟡 Moderate"
    return "🟢 Normal"


def get_study_mode(days):
    if days <= 1:
        return "LAST MINUTE"
    elif days <= 3:
        return "EMERGENCY"
    return "NORMAL"


def get_priority(score):
    if score >= 55:
        return "🔥 HIGH", "priority-high"
    elif score >= 40:
        return "⚡ MEDIUM", "priority-medium"
    return "📘 LOW", "priority-low"


def get_reasons(row):
    reasons = []

    if row["exam_frequency"] >= 7:
        reasons.append("High exam frequency")

    if row["question_marks"] >= 7:
        reasons.append("Important for marks")

    if row["difficulty"] >= 7:
        reasons.append("Difficult topic")

    if row["knowledge_before"] <= 50:
        reasons.append("Needs improvement")

    if not reasons:
        reasons.append("Good for revision")

    return reasons


def reset_plan():
    st.session_state["plan"] = None
    st.session_state["completed_topics"] = []
    st.session_state["study_finished"] = False


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "knowledge": {},
    "quiz_score": None,
    "quiz_topic": None,
    "quiz_skipped": False,
    "plan": None,
    "completed_topics": [],
    "study_finished": False,
    "feedback_saved_message": "",
    "model_metrics": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '''<div class="app-hero">
        <div class="hero-kicker">AI-assisted academic planning</div>
        <div class="app-title">📚 Emergency Study Planner</div>
        <div class="app-subtitle">Plan your study time intelligently when your exam is near.</div>
    </div>''',
    unsafe_allow_html=True,
)


# =========================================================
# EXAM INFORMATION
# =========================================================

st.markdown(
    '<div class="section-label">Planning</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">Exam Information</div>',
    unsafe_allow_html=True,
)

info_col1, info_col2 = st.columns(2)

with info_col1:
    exam_date = st.date_input(
        "Exam date",
        value=date.today(),
        min_value=date.today(),
        key="exam_date",
    )

with info_col2:
    exam_time = st.time_input(
        "Exam time",
        value=dt_time(10, 0),
        key="exam_time",
    )

with info_col1:
    available_minutes = st.number_input(
        "Available study time today (minutes)",
        min_value=10,
        max_value=1440,
        value=180,
        step=10,
        key="available_minutes",
    )

# The app uses Bangladesh time for the exam countdown.
# If ZoneInfo is unavailable, Python's local timezone is used.
if ZoneInfo is not None:
    exam_timezone = ZoneInfo("Asia/Dhaka")
    now = datetime.now(exam_timezone)
    exam_datetime = datetime.combine(
        exam_date,
        exam_time,
        tzinfo=exam_timezone,
    )
else:
    now = datetime.now()
    exam_datetime = datetime.combine(
        exam_date,
        exam_time,
    )

remaining_seconds_exact = (exam_datetime - now).total_seconds()

if remaining_seconds_exact <= 0:
    days_remaining = 0
else:
    days_remaining = max(
        1,
        int((remaining_seconds_exact + 86399) // 86400)
    )

urgency = get_urgency(max(days_remaining, 1))
mode = get_study_mode(max(days_remaining, 1))

st.info(
    f"Exam: {exam_datetime.strftime('%d %B %Y, %I:%M %p')} (Bangladesh time)"
)
st.write(f"Exam urgency: **{urgency}**")
st.write(f"Study Mode: **{mode}**")

if mode == "LAST MINUTE":
    st.warning(
        "Focus on maximum exam-value topics because time is limited."
    )
elif mode == "EMERGENCY":
    st.info(
        "Focus on important and difficult topics first."
    )
else:
    st.success(
        "Balanced preparation mode activated."
    )


# =========================================================
# EXAM COUNTDOWN — EXACT DATE + TIME
# =========================================================

countdown_seconds = max(
    0,
    int(remaining_seconds_exact)
)

components.html(
    f"""
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: transparent;
        }}
        .countdown-wrap {{
            box-sizing: border-box;
            width: 100%;
            padding: 16px 18px;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(79,70,229,.10), rgba(59,130,246,.06));
            border: 1px solid rgba(99,102,241,.20);
        }}
        .countdown-title {{
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #6366f1;
            margin-bottom: 10px;
        }}
        .countdown-date {{
            font-size: 12px;
            color: #7b8494;
            margin-bottom: 10px;
        }}
        .countdown-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .unit {{
            text-align: center;
            padding: 10px 6px;
            border-radius: 12px;
            background: rgba(255,255,255,.72);
            border: 1px solid rgba(99,102,241,.12);
        }}
        .number {{
            font-size: 24px;
            line-height: 1.1;
            font-weight: 800;
            color: #111827;
        }}
        .label {{
            margin-top: 4px;
            font-size: 10px;
            font-weight: 700;
            color: #7b8494;
            text-transform: uppercase;
            letter-spacing: .7px;
        }}
        .expired {{
            text-align: center;
            font-weight: 800;
            color: #dc2626;
            padding: 10px;
        }}
        @media (prefers-color-scheme: dark) {{
            .unit {{ background: rgba(30,32,40,.75); }}
            .number {{ color: #f9fafb; }}
        }}
    </style>

    <div class="countdown-wrap">
        <div class="countdown-title">⏳ Live Exam Countdown</div>
        <div class="countdown-date">📅 {exam_datetime.strftime('%d %B %Y')} &nbsp; • &nbsp; 🕐 {exam_datetime.strftime('%I:%M %p')} (Bangladesh)</div>
        <div id="countdown">
            <div class="countdown-grid">
                <div class="unit"><div class="number" id="hours">00</div><div class="label">Hours</div></div>
                <div class="unit"><div class="number" id="minutes">00</div><div class="label">Minutes</div></div>
                <div class="unit"><div class="number" id="seconds">00</div><div class="label">Seconds</div></div>
            </div>
        </div>
    </div>

    <script>
        let remaining = {countdown_seconds};
        const hours = document.getElementById('hours');
        const minutes = document.getElementById('minutes');
        const seconds = document.getElementById('seconds');
        const container = document.getElementById('countdown');

        function updateCountdown() {{
            if (remaining <= 0) {{
                container.innerHTML = '<div class="expired">🔴 Exam time has arrived</div>';
                return;
            }}

            const h = Math.floor(remaining / 3600);
            const m = Math.floor((remaining % 3600) / 60);
            const sec = remaining % 60;

            hours.textContent = String(h).padStart(2, '0');
            minutes.textContent = String(m).padStart(2, '0');
            seconds.textContent = String(sec).padStart(2, '0');
            remaining--;
        }}

        updateCountdown();
        setInterval(updateCountdown, 1000);
    </script>
    """,
    height=150,
    scrolling=False,
)


# =========================================================
# DIAGNOSTIC QUIZ
# =========================================================

st.markdown(
    '<div class="section-label">Assessment</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">Diagnostic Quiz</div>',
    unsafe_allow_html=True,
)

selected_topic = st.selectbox(
    "Choose topic for quiz",
    list(QUESTIONS.keys()),
    key="selected_quiz_topic",
)

questions = QUESTIONS[selected_topic]
answers = {}

for i, question in enumerate(questions):
    answers[i] = st.radio(
        question["question"],
        question["options"],
        key=f"q_{selected_topic}_{i}",
    )


quiz_col1, quiz_col2 = st.columns(2)

with quiz_col1:
    if st.button(
        "Submit Quiz",
        use_container_width=True,
        key="submit_quiz",
    ):
        correct = sum(
            answers[i] == q["answer"]
            for i, q in enumerate(questions)
        )

        score = (correct / len(questions)) * 100

        topic_df = create_topic_dataframe()
        knowledge = build_baseline_knowledge(
            topic_df,
            score,
            selected_topic,
        )

        st.session_state["knowledge"] = knowledge
        st.session_state["quiz_score"] = score
        st.session_state["quiz_topic"] = selected_topic
        st.session_state["quiz_skipped"] = False

        # A new assessment invalidates the old recommendation.
        reset_plan()

        st.success(f"Quiz Score: {score:.0f}%")
        st.rerun()


with quiz_col2:
    if st.button(
        "⏭️ Skip Quiz",
        use_container_width=True,
        key="skip_quiz",
    ):
        topic_df = create_topic_dataframe()

        st.session_state["knowledge"] = (
            build_baseline_knowledge(topic_df)
        )
        st.session_state["quiz_score"] = None
        st.session_state["quiz_topic"] = None
        st.session_state["quiz_skipped"] = True

        reset_plan()

        st.success(
            "Quiz skipped. Study plan will use 50% baseline knowledge."
        )
        st.rerun()


# =========================================================
# QUIZ SUMMARY
# =========================================================

if (
    st.session_state["quiz_score"] is not None
    or st.session_state["quiz_skipped"]
):
    st.divider()
    st.subheader("Quiz Summary")

    if st.session_state["quiz_skipped"]:
        st.write("Status: **Skipped**")
        st.write("Knowledge baseline: **50%**")
    else:
        st.write(
            f"Topic: **{st.session_state['quiz_topic']}**"
        )
        st.write(
            f"Score: **{st.session_state['quiz_score']:.0f}%**"
        )


# =========================================================
# GENERATE STUDY PLAN
# =========================================================

st.caption(
    "The recommendation is generated using the prototype ML model."
)

if st.button(
    "🚀 Generate Study Plan",
    use_container_width=True,
    key="generate_plan",
):
    knowledge = st.session_state.get("knowledge", {})

    if not knowledge:
        st.warning(
            "Please complete the quiz or click Skip Quiz first."
        )
    else:
        try:
            topic_df = create_topic_dataframe()

            topic_df["knowledge_before"] = (
                topic_df["topic"].map(knowledge).fillna(50)
            )

            model, metrics = train_model(DATA_PATH)

            plan = generate_recommendations(
                model,
                topic_df,
                knowledge,
                available_minutes,
            )

            if plan is None:
                st.error(
                    "The recommender returned no study plan. "
                    "Please check src/recommender.py."
                )
            elif plan.empty:
                st.warning(
                    "No topic fits the available study time."
                )
            else:
                st.session_state["plan"] = plan.reset_index(drop=True)
                st.session_state["model_metrics"] = metrics
                st.session_state["completed_topics"] = []
                st.session_state["study_finished"] = False
                st.session_state["feedback_saved_message"] = ""
                st.rerun()

        except Exception as exc:
            st.error(
                f"Could not generate the study plan: {exc}"
            )


# =========================================================
# SHOW STUDY PLAN
# =========================================================

plan = st.session_state.get("plan")

if isinstance(plan, pd.DataFrame) and not plan.empty:

    st.markdown(
        '<div class="plan-header">'
        '<div class="section-label">Personalized Recommendation</div>'
        '<div class="section-heading">Your Study Plan</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="plan-intro">Prioritized topics selected for maximum study value within your available time.</div>',
        unsafe_allow_html=True,
    )

    total_planned_minutes = int(
        plan["study_minutes"].sum()
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Recommended Topics",
            len(plan),
        )

    with summary_col2:
        st.metric(
            "Total Study Time",
            f"{total_planned_minutes} min",
        )

    with summary_col3:
        st.metric(
            "Available Today",
            f"{available_minutes} min",
        )

    model_metrics = st.session_state.get("model_metrics")
    if model_metrics:
        with st.expander("📊 Model evaluation", expanded=False):
            st.caption(
                "Hold-out evaluation on the current synthetic dataset (80/20 split, random_state=42)."
            )
            eval_col1, eval_col2, eval_col3 = st.columns(3)
            with eval_col1:
                st.metric("MAE", f"{model_metrics['MAE']:.4f}")
            with eval_col2:
                st.metric("RMSE", f"{model_metrics['RMSE']:.4f}")
            with eval_col3:
                st.metric("R²", f"{model_metrics['R2']:.4f}")
            st.info(
                "The dataset is synthetic and small, so these metrics are for educational evaluation only."
            )

    # -----------------------------------------------------
    # Topic cards: two cards per row
    # -----------------------------------------------------

    for start in range(0, len(plan), 2):

        row_data = plan.iloc[start:start + 2]
        columns = st.columns(2)

        for position, (_, row) in enumerate(row_data.iterrows()):

            with columns[position]:

                # Calculate the score once and use it everywhere.
                priority_score = (
                    row["predicted_gain"] * 0.4
                    + row["exam_frequency"] * 2
                    + row["question_marks"] * 2
                    + row["difficulty"] * 1.5
                    - row["knowledge_before"] * 0.2
                )

                priority, priority_class = get_priority(
                    priority_score
                )

                with st.container(border=True):

                    st.markdown(
                        f'<div class="topic-name">'
                        f'<span class="step-badge">{start + position + 1}</span>'
                        f"{row['topic']}"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<span class="{priority_class}">'
                        f"{priority}"
                        "</span>",
                        unsafe_allow_html=True,
                    )

                    completed = st.checkbox(
                        f"Mark {row['topic']} as completed",
                        key=f"done_{row['topic']}",
                    )

                    if completed:
                        if (
                            row["topic"]
                            not in st.session_state["completed_topics"]
                        ):
                            st.session_state["completed_topics"].append(
                                row["topic"]
                            )
                    else:
                        if (
                            row["topic"]
                            in st.session_state["completed_topics"]
                        ):
                            st.session_state["completed_topics"].remove(
                                row["topic"]
                            )

                    feedback = st.selectbox(
                        "Topic difficulty feedback",
                        [
                            "Select",
                            "Easy",
                            "Normal",
                            "Difficult",
                        ],
                        key=f"feedback_{row['topic']}",
                    )

                    feedback_col1, feedback_col2 = st.columns(2)

                    with feedback_col1:
                        if feedback != "Select":
                            save_key = (
                                f"save_feedback_{row['topic']}"
                            )

                            if st.button(
                                "Save Feedback",
                                key=save_key,
                                use_container_width=True,
                            ):
                                save_feedback(
                                    row["topic"],
                                    feedback,
                                )

                                st.session_state[
                                    "feedback_saved_message"
                                ] = (
                                    f"Feedback saved for {row['topic']}."
                                )

                                st.rerun()

                    with feedback_col2:
                        st.write("")

                    if st.session_state[
                        "feedback_saved_message"
                    ].startswith(
                        f"Feedback saved for {row['topic']}"
                    ):
                        st.success(
                            st.session_state[
                                "feedback_saved_message"
                            ]
                        )

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Study Time</div>
                                <div class="metric-value">
                                    ⏱️ {row['study_minutes']:.0f} min
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Exam Frequency</div>
                                <div class="metric-value">
                                    📚 {row['exam_frequency']:.0f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Knowledge</div>
                                <div class="metric-value">
                                    🎯 {row['knowledge_before']:.0f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with metric_col2:
                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Predicted Gain</div>
                                <div class="metric-value">
                                    📈 {row['predicted_gain']:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Question Marks</div>
                                <div class="metric-value">
                                    📝 {row['question_marks']:.0f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">Priority Score</div>
                                <div class="metric-value">
                                    ⭐ {priority_score:.2f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("**💡 Why study this topic?**")

                    for reason in get_reasons(row):
                        st.write(f"✓ {reason}")

    # -----------------------------------------------------
    # Progress
    # -----------------------------------------------------

    st.divider()
    st.subheader("Study Progress")

    completed_count = len(
        st.session_state["completed_topics"]
    )
    total_topics = len(plan)

    progress = (
        completed_count / total_topics
        if total_topics
        else 0
    )

    st.progress(progress)

    progress_col1, progress_col2 = st.columns(2)

    with progress_col1:
        st.write(
            f"**Completed:** {completed_count}/{total_topics} topics"
        )

    with progress_col2:
        st.write(
            f"**Remaining:** {total_topics - completed_count} topics"
        )

    # -----------------------------------------------------
    # Final completion
    # -----------------------------------------------------

    if (
        total_topics > 0
        and completed_count == total_topics
    ):

        if not st.session_state["study_finished"]:

            st.markdown(
                """
                <div class="finish-box">
                    <div class="finish-title">
                        🎉 All Topics Completed!
                    </div>
                    <div class="small-muted">
                        Great job! You have completed your entire
                        recommended study plan.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✅ Finish Study Plan",
                key="finish_study_plan",
                use_container_width=True,
            ):
                st.session_state["study_finished"] = True
                st.balloons()
                st.rerun()

        else:

            st.success(
                "🏆 Study plan completed successfully!"
            )

else:
    if st.session_state.get("study_finished"):
        st.success(
            "🏆 Study plan completed successfully!"
        )
