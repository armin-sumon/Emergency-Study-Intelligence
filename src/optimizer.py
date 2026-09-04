import numpy as np
import pandas as pd


def optimize_study_plan(
    topic_df,
    available_minutes,
):

    topics = topic_df.copy()

    if topics.empty:
        return pd.DataFrame(
            columns=topics.columns
        )

    # -----------------------
    # Sort highest priority first
    # -----------------------

    topics = (
        topics.sort_values(
            by="priority_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    n = len(topics)

    capacity = max(
        1,
        int(available_minutes),
    )

    # -----------------------
    # Knapsack DP
    # -----------------------

    dp = np.zeros(
        (n + 1, capacity + 1),
        dtype=float,
    )

    for i in range(1, n + 1):

        study_time = max(
            1,
            int(round(
                float(
                    topics.loc[
                        i - 1,
                        "study_minutes",
                    ]
                )
            )),
        )

        value = float(
            topics.loc[
                i - 1,
                "priority_score",
            ]
        )

        for t in range(
            capacity + 1
        ):

            dp[i, t] = dp[
                i - 1,
                t
            ]

            if study_time <= t:

                candidate = (
                    dp[
                        i - 1,
                        t - study_time,
                    ]
                    + value
                )

                if candidate > dp[i, t]:
                    dp[i, t] = candidate

    # -----------------------
    # Recover selected topics
    # -----------------------

    selected_indices = []

    t = capacity

    for i in range(
        n,
        0,
        -1,
    ):

        if dp[i, t] != dp[i - 1, t]:

            selected_indices.append(
                i - 1
            )

            study_time = max(
                1,
                int(round(
                    float(
                        topics.loc[
                            i - 1,
                            "study_minutes",
                        ]
                    )
                )),
            )

            t = max(
                0,
                t - study_time,
            )

    selected_indices.reverse()

    # -----------------------
    # Fallback
    # -----------------------

    if not selected_indices:

        # If every topic is longer than the available time,
        # still give the student the highest-priority topic.
        selected_indices = [0]

    # -----------------------
    # Final plan
    # -----------------------

    return (
        topics.loc[selected_indices]
        .copy()
        .reset_index(drop=True)
    )
