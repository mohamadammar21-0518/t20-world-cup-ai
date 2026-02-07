import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE SETUP ---
st.set_page_config(
    page_title="T20 WC 2026 AI Dashboard",
    page_icon="🏏",
    layout="wide",
)

# Custom CSS for the professional "Match Card" and Dashboard look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2563eb; color: white; }
    .date-header { 
        background-color: #2563eb; padding: 8px 15px; border-radius: 5px; 
        font-weight: bold; margin: 20px 0 10px 0; color: white;
    }
    .match-card {
        border: 1px solid #374151; border-radius: 10px; padding: 15px; 
        margin-bottom: 10px; background-color: #1f2937;
    }
    .venue-text { color: #60a5fa; font-size: 0.85em; }
    .team-name { font-size: 1.1em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Define Team Tiers for realistic results
TEAM_TIERS = {
    'India': 1, 'Australia': 1, 'England': 1, 'South Africa': 1,
    'Pakistan': 1, 'New Zealand': 1, 'West Indies': 1, 'Sri Lanka': 1, 'Afghanistan': 1,
    'Ireland': 3, 'Zimbabwe': 2, 'Netherlands': 3, 'Scotland': 3, 'USA': 3,
    'Nepal': 4, 'Canada': 4, 'UAE': 4, 'Namibia': 4, 'Oman': 4, 'Italy': 4
}


# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = joblib.load('t20_model.joblib')
    venue_encoder = joblib.load('venue_encoder.joblib')
    team_strength = joblib.load('team_strength.joblib')
    icc_ratings = joblib.load('icc_ratings.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    schedule_df = pd.read_csv("wc_2026_schedule.csv")

    official_teams = sorted(['India', 'Sri Lanka', 'Afghanistan', 'Australia', 'England',
                             'South Africa', 'USA', 'West Indies', 'Ireland', 'New Zealand',
                             'Pakistan', 'Canada', 'Netherlands', 'Italy', 'Zimbabwe',
                             'Namibia', 'Nepal', 'Oman', 'UAE', 'Scotland'])
    official_venues = sorted(schedule_df['venue'].unique().tolist())

    return model, venue_encoder, team_strength, icc_ratings, feature_columns, official_teams, official_venues, schedule_df


model, venue_encoder, team_strength, icc_ratings, feature_columns, official_teams, official_venues, schedule_df = load_assets()


# --- 3. PREDICTION ENGINE WITH TIER OVERRIDE ---
def get_prediction(t1, t2, venue):
    match_feat = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    s1, s2 = team_strength.get(t1, 0.1), team_strength.get(t2, 0.1)
    p1_pwr, p2_pwr = icc_ratings.get(t1, 50.0) / 100, icc_ratings.get(t2, 50.0) / 100

    v_id = venue_encoder.transform([venue])[0] if venue in venue_encoder.classes_ else 0

    match_feat.at[0, 'v_id'] = float(v_id)
    match_feat.at[0, 't1_s'], match_feat.at[0, 't2_s'] = float(s1), float(s2)
    match_feat.at[0, 't1_player_pwr'], match_feat.at[0, 't2_player_pwr'] = float(p1_pwr), float(p2_pwr)
    match_feat.at[0, 'h2h_rate'], match_feat.at[0, 'venue_avg'] = 0.5, 0.6

    if f't1_{t1}' in match_feat.columns: match_feat.at[0, f't1_{t1}'] = 1.0
    if f't2_{t2}' in match_feat.columns: match_feat.at[0, f't2_{t2}'] = 1.0

    # Base AI Probability
    prob = model.predict_proba(match_feat)[0][1]

    # --- TIER OVERRIDE LOGIC ---
    tier1 = TEAM_TIERS.get(t1, 3)
    tier2 = TEAM_TIERS.get(t2, 3)
    tier_diff = tier2 - tier1

    if tier_diff >= 2:  # T1 vs T3/T4 (Giant vs Small)
        prob = min(0.98, prob + 0.35) if prob > 0.5 else max(0.95, prob + 0.4)
    elif tier_diff <= -2:  # T4 vs T1/T2 (Small vs Giant)
        prob = max(0.02, prob - 0.35) if prob < 0.5 else min(0.05, prob - 0.4)

    return float(prob)


# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/2026_ICC_Men%27s_T20_World_Cup_logo.png", width=120)
    st.title("Control Panel")
    st.divider()
    side_t1 = st.selectbox("Team 1", official_teams, index=official_teams.index("India"))
    side_t2 = st.selectbox("Team 2", official_teams, index=official_teams.index("Australia"))
    side_venue = st.selectbox("Venue", official_venues)

# --- 5. MAIN DASHBOARD ---
st.title("🏏 T20 World Cup 2026 AI Intelligence")

tab_match, tab_tournament = st.tabs(["🎯 Match Predictor", "📊 Full Fixtures & Simulation"])

with tab_match:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Match Analysis")
        if st.button("Run AI Prediction", key="predict_single"):
            if side_t1 == side_t2:
                st.error("Select two different teams.")
            else:
                prob = get_prediction(side_t1, side_t2, side_venue)
                winner = side_t1 if prob >= 0.5 else side_t2
                conf = prob if winner == side_t1 else (1 - prob)

                st.metric(label="Predicted Winner", value=winner, delta=f"{conf:.1%} Confidence")
                st.progress(float(conf))

                m_col1, m_col2 = st.columns(2)
                m_col1.metric(side_t1, f"{team_strength.get(side_t1, 0):.2f}", "Form")
                m_col2.metric(side_t2, f"{team_strength.get(side_t2, 0):.2f}", "Form")

    with col_b:
        st.subheader("Venue Context")
        st.write(f"📍 **{side_venue}**")
        st.info("Conditions: Adjusted for Tier-1 Quality Play.")
        st.bar_chart(np.random.randn(5, 1))

# Replace your current 'tab_tournament' with this expanded logic
with tab_tournament:
    st.subheader("Tournament Path Prediction")

    if st.button("🚀 Launch Full Simulation", use_container_width=True):
        # 1. Initialize Points Table
        points_table = {team: {"points": 0, "wins": 0} for team in official_teams}

        # 2. Group Stage Simulation
        st.markdown("### 🏟️ Phase 1: Group Stage Results")
        dates = schedule_df['date'].unique()

        for date in dates:
            st.markdown(f'<div class="date-header">{date}</div>', unsafe_allow_html=True)
            day_matches = schedule_df[schedule_df['date'] == date]
            for _, row in day_matches.iterrows():
                t1, t2, v = row['team1'], row['team2'], row['venue']
                prob = get_prediction(t1, t2, v)
                winner = t1 if prob >= 0.5 else t2

                # Update Points
                points_table[winner]["points"] += 2
                points_table[winner]["wins"] += 1

                st.write(f"✔️ **{t1} vs {t2}** → Winner: **{winner}**")

        st.divider()

        # 3. Identify Top 8 (Simplifying to top 2 from each pre-defined group)
        # In a real app, you'd filter by Group A, B, C, D explicitly.
        # For this demo, we'll pick the top performers to advance to Super 8s.
        sorted_teams = sorted(points_table.items(), key=lambda x: x[1]['points'], reverse=True)
        top_8 = [team[0] for team in sorted_teams[:8]]

        # --- PHASE 2: SUPER 8 PREPARATION ---
        st.markdown("### 📊 Phase 2: Super 8 Qualifiers & Rankings")

        # Based on your input groups
        s8_g1_teams = ["Australia", "India", "Pakistan", "Sri Lanka"]
        s8_g2_teams = ["England", "New Zealand", "South Africa", "Afghanistan"]

        # Displaying the Tables
        tab_s8_1, tab_s8_2 = st.columns(2)
        with tab_s8_1:
            st.info("**Super 8 - Group 1**")
            st.table(pd.DataFrame({"Seeding": ["X1", "X2", "X3", "X4"], "Team": s8_g1_teams}))
        with tab_s8_2:
            st.info("**Super 8 - Group 2**")
            st.table(pd.DataFrame({"Seeding": ["Y1", "Y2", "Y3", "Y4"], "Team": s8_g2_teams}))

        st.divider()

        # --- SUPER 8 MATCH SCHEDULE ---
        st.markdown("### 🏟️ Phase 2: Super 8 Match Reports")

        # Defining the schedule based on official 2026 dates/seeds
        # X1: India, X2: Australia, X3: Pakistan, X4: Sri Lanka
        # Y1: England, Y2: New Zealand, Y3: South Africa, Y4: Afghanistan
        s8_fixtures = [
            {"date": "Feb 21", "t1": "New Zealand", "t2": "South Africa", "venue": "Colombo (RP)"},
            {"date": "Feb 22", "t1": "England", "t2": "Afghanistan", "venue": "Kandy"},
            {"date": "Feb 22", "t1": "India", "t2": "Sri Lanka", "venue": "Ahmedabad"},
            {"date": "Feb 23", "t1": "Australia", "t2": "Pakistan", "venue": "Mumbai"},
            {"date": "Feb 24", "t1": "England", "t2": "South Africa", "venue": "Kandy"},
            {"date": "Feb 25", "t1": "New Zealand", "t2": "Afghanistan", "venue": "Colombo (RP)"},
            {"date": "Feb 26", "t1": "Pakistan", "t2": "Sri Lanka", "venue": "Ahmedabad"},
            {"date": "Feb 26", "t1": "India", "t2": "Australia", "venue": "Chennai"},
            {"date": "Feb 27", "t1": "England", "t2": "New Zealand", "venue": "Colombo (RP)"},
            {"date": "Feb 28", "t1": "South Africa", "t2": "Afghanistan", "venue": "Kandy"},
            {"date": "March 1", "t1": "Australia", "t2": "Sri Lanka", "venue": "Delhi"},
            {"date": "March 1", "t1": "India", "t2": "Pakistan", "venue": "Kolkata"}
        ]

        s8_points = {team: 0 for team in s8_g1_teams + s8_g2_teams}

        for match in s8_fixtures:
            st.markdown(f'<div class="date-header">{match["date"]}</div>', unsafe_allow_html=True)
            prob = get_prediction(match["t1"], match["t2"], match["venue"])
            winner = match["t1"] if prob >= 0.5 else match["t2"]
            s8_points[winner] += 2

            st.markdown(f"""
                        <div class="match-card">
                            <span class="venue-text">{match['venue']}</span><br>
                            <b>{match['t1']} vs {match['t2']}</b><br>
                            <span style="color: #10b981;">AI Predicted Winner: {winner}</span>
                        </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # --- TOP 4 / SEMI FINALISTS ---
        # Sort and pick top 2 from each group
        g1_finalists = sorted(s8_g1_teams, key=lambda x: s8_points[x], reverse=True)[:2]
        g2_finalists = sorted(s8_g2_teams, key=lambda x: s8_points[x], reverse=True)[:2]

        st.markdown("### 🏆 Phase 3: The Grand Finale")

        # Logic for Semi-Finals
        sf1_t1, sf1_t2 = g1_finalists[0], g2_finalists[1]
        sf2_t1, sf2_t2 = g2_finalists[0], g1_finalists[1]

        # Simulation
        sf1_win = sf1_t1 if get_prediction(sf1_t1, sf1_t2, "Kolkata") >= 0.5 else sf1_t2
        sf2_win = sf2_t1 if get_prediction(sf2_t1, sf2_t2, "Mumbai") >= 0.5 else sf2_t2

        final_win = sf1_win if get_prediction(sf1_win, sf2_win, "Ahmedabad") >= 0.5 else sf2_win

        # Final UI Display
        c1, c2, c3 = st.columns(3)
        c1.metric("Semi-Final 1 Winner", sf1_win, f"Beat {sf1_t2 if sf1_win == sf1_t1 else sf1_t1}")
        c2.metric("Semi-Final 2 Winner", sf2_win, f"Beat {sf2_t2 if sf2_win == sf2_t1 else sf2_t1}")

        st.markdown(f"""
                    <div style="text-align: center; border: 4px solid #facc15; padding: 30px; border-radius: 15px; background-color: #1e1b4b;">
                        <h1 style="color: #facc15;">2026 WORLD CHAMPION</h1>
                        <h2 style="color: white; font-size: 4em;">{final_win.upper()}</h2>
                        <p style="color: #94a3b8;">Predicted by XGBoost AI Dashboard</p>
                    </div>
                """, unsafe_allow_html=True)
        st.balloons()