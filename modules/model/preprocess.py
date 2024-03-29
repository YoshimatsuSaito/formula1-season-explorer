import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create processed data"""
    # Make relative performance about qualifying
    df["q1_fastest"] = df.groupby(["year", "round"])["q1"].transform("min")
    df["q2_fastest"] = df.groupby(["year", "round"])["q2"].transform("min")
    df["q3_fastest"] = df.groupby(["year", "round"])["q3"].transform("min")
    df["q1_relative_performance"] = (df["q1"] - df["q1_fastest"]) / df["q1_fastest"]
    df["q2_relative_performance"] = (df["q2"] - df["q2_fastest"]) / df["q2_fastest"]
    df["q3_relative_performance"] = (df["q3"] - df["q3_fastest"]) / df["q3_fastest"]
    df["q_relative_performance"] = (
        df[
            [
                "q1_relative_performance",
                "q2_relative_performance",
                "q3_relative_performance",
            ]
        ].mean(axis=1, skipna=True)
    ) * 100

    # Make prev position data
    df.sort_values(by=["driver", "grandprix", "year"], inplace=True)
    df["prev_position"] = df.groupby(["driver", "grandprix"])["position"].shift(1)

    # Make rolling data
    df.sort_values(by=["driver", "year", "round"], inplace=True)
    df["recent_position"] = df.groupby(["driver", "year"])["position"].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["season_position"] = df.groupby(["driver", "year"])["position"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["season_q_relative_performance"] = df.groupby(["driver", "year"])[
        "q_relative_performance"
    ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    return df
