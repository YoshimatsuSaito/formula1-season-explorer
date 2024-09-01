import streamlit as st

from modules.inmemory_db import InmemoryDB


def show_user_search_result(
    db: InmemoryDB, list_result_type: list[str], grandprix: str
) -> None:
    """Show past result which user want to see"""

    list_result_type = [x if x != "qualify" else f"{x}ing" for x in list_result_type]

    # Users choose season
    df_season = db.execute_query(
        "SELECT DISTINCT season FROM race_result ORDER BY season"
    )
    season = st.selectbox(
        label="Season", options=df_season["season"], index=len(df_season) - 2
    )
    # Show the result
    dict_df = dict()
    for result_type in list_result_type:
        df = db.execute_query(
            f"SELECT * FROM {result_type} WHERE season = {season} AND grandprix = '{grandprix}'"
        )
        if "position" in df.columns:
            df.sort_values(by="position", inplace=True)
        dict_df[result_type] = df

    for result_type in list_result_type:
        st.markdown(f"##### {result_type}")

        st.dataframe(dict_df[result_type], hide_index=True)
