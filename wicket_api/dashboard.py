
import streamlit as st
import requests
import pandas as pd

# --------------------------------------------------
# LOAD PLAYER DATA FROM GITHUB
# --------------------------------------------------

@st.cache_data
def load_data():

    player_attributes = pd.read_csv(
        "https://raw.githubusercontent.com/"
        "HimanshuKhale/datasets_cricket/main/"
        "player_attributes.csv"
    )

    players = pd.read_csv(
        "https://raw.githubusercontent.com/"
        "HimanshuKhale/datasets_cricket/main/"
        "players.csv"
    )

    return player_attributes, players


player_attributes, players = load_data()

# --------------------------------------------------
# MERGE PLAYER INFO
# --------------------------------------------------

player_lookup = players.merge(
    player_attributes[
        [
            "player_id",
            "role"
        ]
    ],
    left_on="id",
    right_on="player_id",
    how="left"
)

player_lookup = player_lookup.fillna("Unknown")

# Only realistic selections

batters = player_lookup[
    player_lookup["role"].isin(
        [
            "Batter",
            "All-rounder"
        ]
    )
]

bowlers = player_lookup[
    player_lookup["role"].isin(
        [
            "Bowler",
            "All-rounder"
        ]
    )
]

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="AI Wicket Prediction Dashboard",
    page_icon="🏏"
)

st.title("🏏 AI Wicket Prediction Dashboard")

# --------------------------------------------------
# MATCH SITUATION
# --------------------------------------------------

st.subheader("Match Situation")

phase = st.selectbox(
    "Phase",
    [
        "powerplay",
        "middle",
        "death"
    ]
)

runs_last_12 = st.slider(
    "Runs in last 12 balls",
    0,
    36,
    12
)

dots_last_12 = st.slider(
    "Dot balls in last 12",
    0,
    12,
    4
)

wickets_last_12 = st.slider(
    "Wickets in last 12",
    0,
    3,
    1
)

boundaries_last_12 = st.slider(
    "Boundaries in last 12",
    0,
    8,
    2
)

sixes_last_12 = st.slider(
    "Sixes in last 12",
    0,
    6,
    1
)

current_run_rate = st.number_input(
    "Current Run Rate",
    value=8.2
)

required_run_rate = st.number_input(
    "Required Run Rate",
    value=10.5
)

over_number = st.slider(
    "Over Number",
    1,
    20,
    18
)

ball_number = st.slider(
    "Ball Number",
    1,
    6,
    3
)

# --------------------------------------------------
# PLAYER MATCHUP
# --------------------------------------------------

st.subheader("Player Matchup")

striker_name = st.selectbox(
    "Batsman",
    batters["name"].tolist()
)

bowler_name = st.selectbox(
    "Bowler",
    bowlers["name"].tolist()
)

striker_id = batters.loc[
    batters["name"] == striker_name,
    "id"
].iloc[0]

bowler_id = bowlers.loc[
    bowlers["name"] == bowler_name,
    "id"
].iloc[0]

batter_role = batters.loc[
    batters["name"] == striker_name,
    "role"
].iloc[0]

bowler_role = bowlers.loc[
    bowlers["name"] == bowler_name,
    "role"
].iloc[0]

st.info(
    f"Selected Matchup:\n\n"
    f"🏏 Batter: {striker_name} ({batter_role})\n\n"
    f"🎯 Bowler: {bowler_name} ({bowler_role})"
)

# --------------------------------------------------
# PLAYER ATTRIBUTES
# --------------------------------------------------

batter = player_attributes[
    player_attributes["player_id"] == striker_id
]

bowler = player_attributes[
    player_attributes["player_id"] == bowler_id
]

if batter.empty:
    st.error("Batsman not found.")
    st.stop()

if bowler.empty:
    st.error("Bowler not found.")
    st.stop()

batter = batter.iloc[0]
bowler = bowler.iloc[0]

batter_rating = int(
    batter["batting_rating"]
)

batter_aggression = int(
    batter["aggression"]
)

batter_consistency = int(
    batter["consistency"]
)

batter_pressure_handling = int(
    batter["pressure_handling"]
)

bowler_rating = int(
    bowler["bowling_rating"]
)

bowler_death_skill = int(
    bowler["death_overs_skill"]
)

bowler_pressure_handling = int(
    bowler["pressure_handling"]
)

# --------------------------------------------------
# SHOW PLAYER STATS
# --------------------------------------------------

with st.expander("Player Statistics"):

    col1, col2 = st.columns(2)

    with col1:

        st.write("### 🏏 Batter")

        st.write(
            {
                "Name": striker_name,
                "Role": batter_role,
                "Batting Rating": batter_rating,
                "Aggression": batter_aggression,
                "Consistency": batter_consistency,
                "Pressure Handling": batter_pressure_handling
            }
        )

    with col2:

        st.write("### 🎯 Bowler")

        st.write(
            {
                "Name": bowler_name,
                "Role": bowler_role,
                "Bowling Rating": bowler_rating,
                "Death Overs Skill": bowler_death_skill,
                "Pressure Handling": bowler_pressure_handling
            }
        )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Wicket Probability"):

    payload = {

        "phase": phase,

        "runs_last_12": runs_last_12,
        "dots_last_12": dots_last_12,
        "wickets_last_12": wickets_last_12,
        "boundaries_last_12": boundaries_last_12,
        "sixes_last_12": sixes_last_12,

        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,

        "over_number": over_number,
        "ball_number": ball_number,

        "batter_rating": batter_rating,
        "batter_aggression": batter_aggression,
        "batter_consistency": batter_consistency,
        "batter_pressure_handling": batter_pressure_handling,

        "bowler_rating": bowler_rating,
        "bowler_death_skill": bowler_death_skill,
        "bowler_pressure_handling": bowler_pressure_handling
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/wicket/predict",
            json=payload
        )

        response.raise_for_status()

        result = response.json()

        st.success("Prediction Complete")

        st.metric(
            "Wicket Probability",
            f"{result['wicket_percentage']}%"
        )

        st.metric(
            "Threat Level",
            result["label"]
        )

        with st.expander("API Response"):
            st.write(result)

    except Exception as e:

        st.error(
            f"API Error: {e}"
        )

