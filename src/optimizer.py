import numpy as np


def optimize_study_plan(topic_df, available_minutes):

    topics = topic_df.reset_index(drop=True)

    n = len(topics)
    capacity = int(available_minutes)

    dp = np.zeros((n + 1, capacity + 1))

    for i in range(1, n + 1):

        time = int(
            topics.loc[i - 1, "study_minutes"]
        )

        value = topics.loc[
            i - 1,
            "expected_value"
        ]

        for t in range(capacity + 1):

            dp[i][t] = dp[i - 1][t]

            if time <= t:

                dp[i][t] = max(
                    dp[i][t],
                    dp[i - 1][t - time] + value
                )

    selected_indices = []

    t = capacity

    for i in range(n, 0, -1):

        if dp[i][t] != dp[i - 1][t]:

            selected_indices.append(i - 1)

            t -= int(
                topics.loc[
                    i - 1,
                    "study_minutes"
                ]
            )

    selected_indices.reverse()

    selected_topics = topics.loc[
        selected_indices
    ]

    return selected_topics