import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. PAGE SETUP
# -------------------------------------------------------------
st.set_page_config(
    page_title="ICC T20 World Cup 2026 – AI Dashboard",
    page_icon=r"C:\Users\User\OneDrive\Desktop\T20_WorldCup_Predictor\Gemini_Generated_Image_k03jlnk03jlnk03j.png",
    layout="wide",
)

# -------------------------------------------------------------
# 2. PREMIUM ICC CSS / JS / HTML
# -------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #071526 0%, #0b2545 50%, #13315c 100%) !important;
}

/* Floating Top Navigation Bar */
.navbar {
    width: 100%;
    padding: 18px 40px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    position: sticky;
    top: 0;
    z-index: 999;
}

.nav-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1px;
}

.nav-sub {
    color: #45b7f5;
    font-size: 0.85rem;
    margin-left: 6px;
}

/* Footer */
.footer {
    margin-top: 50px;
    padding: 25px;
    text-align: center;
    color: #cbd5e1;
    font-size: 0.85rem;
}

/* ICC Glass Cards */
.icc-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(20px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    transition: transform .3s ease;
}

.icc-card:hover {
    transform: translateY(-6px);
    border-color: #45b7f5;
}

/* Team Title */
.team-title {
    font-weight: 800;
    font-size: 2rem;
    text-transform: uppercase;
    color: #ffffff;
    text-shadow: 0 0 5px rgba(255,255,255,0.3);
}

/* VS Badge */
.vs-badge {
    margin-top: 30px;
    font-size: 1.2rem;
    padding: 10px 22px;
    background: linear-gradient(90deg, #e11d48, #be123c);
    color: white;
    border-radius: 50px;
    font-weight: 700;
    box-shadow: 0 3px 8px rgba(255,0,60,0.4);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0369a1, #0284c7);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 10px 25px;
    border-radius: 10px;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0284c7, #0ea5e9);
    transform: scale(1.03);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 25px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.08);
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 600;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.13);
}

.stTabs [aria-selected="true"] {
    background: #0284c7 !important;
    color: white !important;
    border-color: #0284c7 !important;
}

/* Text Colors */
h1, h2, h3, h4, h5, h6, p, div {
    color: #e2e8f0 !important;
}

/* Metric Styling */
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 2rem !important;
}

[data-testid="stMetricDelta"] {
    color: #22c55e !important;
}

/* Animation */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px);}
    100% { opacity: 1; transform: translateY(0);}
}

.fade-in {
    animation: fadeIn 0.9s ease-in-out forwards;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. NAVIGATION BAR
# -------------------------------------------------------------
st.markdown("""
<div class='navbar'>
    <span class='nav-title'>
        ICC T20 WORLD CUP 2026 
        <span class='nav-sub'>AI ANALYTICS SUITE</span>
    </span>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 4. FLAG FUNCTION
# -------------------------------------------------------------
def get_flag(team):
    codes = {
        'India': 'IN', 'Australia': 'AU', 'England': 'GB', 'South Africa': 'ZA',
        'Pakistan': 'PK', 'New Zealand': 'NZ', 'West Indies': 'WI', 'Sri Lanka': 'LK',
        'Afghanistan': 'AF', 'Ireland': 'IE', 'Zimbabwe': 'ZW', 'Netherlands': 'NL',
        'Scotland': 'GB-SCT', 'USA': 'US', 'Nepal': 'NP', 'Canada': 'CA',
        'UAE': 'AE', 'Namibia': 'NA', 'Oman': 'OM', 'Italy': 'IT'
    }
    code = codes.get(team, 'UN')
    return f"https://flagsapi.com/{code}/flat/64.png"


# -------------------------------------------------------------
# 5. LOAD MODEL ASSETS
# -------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('t20_model.joblib')
    venue_encoder = joblib.load('venue_encoder.joblib')
    team_strength = joblib.load('team_strength.joblib')
    icc_ratings = joblib.load('icc_ratings.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    schedule_df = pd.read_csv("wc_2026_schedule.csv")

    official_teams = sorted(team_strength.keys())
    official_venues = sorted(schedule_df['venue'].unique())

    return model, venue_encoder, team_strength, icc_ratings, feature_columns, official_teams, official_venues


model, venue_encoder, team_strength, icc_ratings, feature_columns, official_teams, official_venues = load_assets()


# -------------------------------------------------------------
# 6. MATCH PREDICTION ENGINE
# -------------------------------------------------------------
def get_prediction(t1, t2, venue):
    TEAM_TIERS = {'India': 1, 'Australia': 1, 'England': 1, 'South Africa': 1, 'Pakistan': 1,
                  'New Zealand': 1, 'West Indies': 1, 'Sri Lanka': 1, 'Afghanistan': 1,
                  'Ireland': 3, 'Zimbabwe': 2, 'Netherlands': 3, 'Scotland': 3, 'USA': 3,
                  'Nepal': 4, 'Canada': 4, 'UAE': 4, 'Namibia': 4, 'Oman': 4, 'Italy': 4}

    match_feat = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    s1, s2 = team_strength[t1], team_strength[t2]
    p1, p2 = icc_ratings[t1] / 100, icc_ratings[t2] / 100
    v_id = venue_encoder.transform([venue])[0]

    match_feat.at[0, 'v_id'] = float(v_id)
    match_feat.at[0, 't1_s'], match_feat.at[0, 't2_s'] = float(s1), float(s2)
    match_feat.at[0, 't1_player_pwr'], match_feat.at[0, 't2_player_pwr'] = float(p1), float(p2)
    match_feat.at[0, 'h2h_rate'], match_feat.at[0, 'venue_avg'] = 0.5, 0.6

    if f"t1_{t1}" in match_feat.columns:
        match_feat.at[0, f"t1_{t1}"] = 1
    if f"t2_{t2}" in match_feat.columns:
        match_feat.at[0, f"t2_{t2}"] = 1

    prob = model.predict_proba(match_feat)[0][1]

    # Tier boost
    tier_diff = TEAM_TIERS[t2] - TEAM_TIERS[t1]
    if tier_diff >= 2:
        prob = min(0.98, prob + 0.35)
    elif tier_diff <= -2:
        prob = max(0.02, prob - 0.35)

    return float(prob)


# -------------------------------------------------------------
# 7. GAUGE VISUAL
# -------------------------------------------------------------
def draw_win_probability(prob, t1, t2):
    # Calculate percentages
    t1_prob = round(prob * 100)
    t2_prob = 100 - t1_prob

    # Custom HTML for the horizontal bar
    html_code = f"""
    <div style="margin-bottom: 20px; font-family: 'Poppins', sans-serif;">
        <p style="text-align: center; font-weight: 700; color: white; letter-spacing: 1px; margin-bottom: 25px;">WIN PROBABILITY</p>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div style="text-align: left;">
                <span style="display: block; font-weight: 600; color: white; font-size: 1.1rem;">{t1}</span>
                <span style="display: block; color: #38bdf8; font-size: 1.5rem; font-weight: 800;">{t1_prob}%</span>
            </div>
            <div style="text-align: right;">
                <span style="display: block; font-weight: 600; color: white; font-size: 1.1rem;">{t2}</span>
                <span style="display: block; color: #e2e8f0; font-size: 1.5rem; font-weight: 800;">{t2_prob}%</span>
            </div>
        </div>
        <div style="width: 100%; background-color: #1e293b; border-radius: 4px; height: 12px; display: flex; overflow: hidden;">
            <div style="width: {t1_prob}%; background: #0ea5e9; height: 100%;"></div>
            <div style="width: {t2_prob}%; background: #94a3b8; height: 100%; opacity: 0.3;"></div>
        </div>
    </div>
    """
    return st.markdown(html_code, unsafe_allow_html=True)

# -------------------------------------------------------------
# 8. SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    # Path to your generated logo
    logo_path = r"C:\Users\User\OneDrive\Desktop\T20_WorldCup_Predictor\Gemini_Generated_Image_k03jlnk03jlnk03j.png"

    try:
        # We use use_container_width to avoid the warning box
        st.image(logo_path, use_container_width=True)
    except:
        # Fallback if the path is slightly wrong or file is moved
        st.error("Logo file not found. Check the path!")

    st.markdown("<h2 style='text-align: center; color: white;'>🌐 AI Control Center</h2>", unsafe_allow_html=True)
    st.divider()

    # Match Selection Inputs
    st.markdown("### 🛠️ Configuration")
    side_t1 = st.selectbox("Select Team 1", official_teams, index=official_teams.index("India"))
    side_t2 = st.selectbox("Select Team 2", official_teams, index=official_teams.index("Australia"))
    side_venue = st.selectbox("Match Venue", official_venues)

    st.divider()
    st.info("Predicting using the 2026 AI Core Engine.")

# -------------------------------------------------------------
# 9. MAIN UI
# -------------------------------------------------------------
st.title("🏏 ICC T20 World Cup 2026 – AI Match Analysis")
st.markdown("<br>", unsafe_allow_html=True)

tab_match, tab_tournament = st.tabs(["🎯 Match Predictor", "📊 Tournament Simulation"])

# -------------------------------------------------------------
# MATCH PREDICTOR UI
# -------------------------------------------------------------
with tab_match:
    # 1. CSS to kill extra gaps between elements in this tab
    st.markdown("""
        <style>
            [data-testid="stVerticalBlock"] > div:has(div.icc-card) {
                gap: 0rem !important;
                padding-bottom: 0rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 2. Header Card (Flags & VS)
    st.markdown(f"""
        <div class='icc-card fade-in' style='margin-bottom: 0px;'>
            <div style='display: flex; justify-content: space-around; align-items: center; text-align: center;'>
                <div style='flex: 1;'>
                    <img src='{get_flag(side_t1)}' width='100'>
                    <p style='font-weight: 800; font-size: 1.4rem; margin-top: 10px; color: white; margin-bottom:0;'>{side_t1.upper()}</p>
                </div>
                <div style='flex: 0.5;'>
                    <div class='vs-badge'>VS</div>
                </div>
                <div style='flex: 1;'>
                    <img src='{get_flag(side_t2)}' width='100'>
                    <p style='font-weight: 800; font-size: 1.4rem; margin-top: 10px; color: white; margin-bottom:0;'>{side_t2.upper()}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Predict Button (Tight against the card)
    if st.button("RUN MATCH INTELLIGENCE", use_container_width=True):
        if side_t1 == side_t2:
            st.error("Select different teams.")
        else:
            prob = get_prediction(side_t1, side_t2, side_venue)
            winner = side_t1 if prob >= 0.5 else side_t2

            # 4. Results Section (NO gaps allowed here)
            st.markdown(f"""
                <div class='icc-card fade-in' style='margin-top: 10px; border-top: 2px solid rgba(255,255,255,0.1);'>
                    <div style="margin-bottom: 15px;">
                        <p style="text-align: center; font-weight: 700; color: #94a3b8; font-size: 0.8rem; letter-spacing: 2px; margin-bottom: 15px;">WIN PROBABILITY</p>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <div style="text-align: left;">
                                <span style="display: block; font-weight: 600; color: white; font-size: 1rem;">{side_t1}</span>
                                <span style="display: block; color: #0ea5e9; font-size: 1.5rem; font-weight: 800;">{round(prob * 100)}%</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="display: block; font-weight: 600; color: white; font-size: 1rem;">{side_t2}</span>
                                <span style="display: block; color: #94a3b8; font-size: 1.5rem; font-weight: 800;">{100 - round(prob * 100)}%</span>
                            </div>
                        </div>
                        <div style="width: 100%; background: #1e293b; border-radius: 50px; height: 12px; display: flex; overflow: hidden;">
                            <div style="width: {prob * 100}%; background: linear-gradient(90deg, #0369a1, #0ea5e9); height: 100%;"></div>
                        </div>
                    </div>
                    <hr style='border: 0.1px solid rgba(255,255,255,0.05); margin: 15px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <div>
                            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>PREDICTED WINNER</p>
                            <p style='color: #38bdf8; font-size: 1.2rem; font-weight: 700; margin:0;'>{winner.upper()}</p>
                        </div>
                        <div style='text-align: right;'>
                            <p style='color: #94a3b8; font-size: 0.8rem; margin:0;'>VENUE</p>
                            <p style='color: white; font-size: 1rem; font-weight: 600; margin:0;'>{side_venue}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
# -------------------------------------------------------------
# TOURNAMENT SIMULATION
# -------------------------------------------------------------
# -------------------------------------------------------------
# TOURNAMENT SIMULATION (REPLACE YOUR CURRENT tab_tournament WITH THIS)
# -------------------------------------------------------------
with tab_tournament:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏆 Full Tournament Simulation Engine")

    # Use session state to control the "Flow" of predictions
    if 'sim_step' not in st.session_state:
        st.session_state.sim_step = 0

    if st.button("🚀 START SIMULATION") or st.session_state.sim_step > 0:
        st.session_state.sim_step = 1

        # --- PHASE 1: GROUP STAGE ---
        st.markdown("### 🏟️ Step 1: Group Stage Results")
        schedule_df = pd.read_csv("wc_2026_schedule.csv")
        s8_points = {team: 0 for team in official_teams}  # Track for S8 qualification

        # Group stage match-by-match display
        for date, group in schedule_df.groupby('date'):
            with st.expander(f"📅 Matches on {date}"):
                for _, row in group.iterrows():
                    # Predict each match
                    prob = get_prediction(row['team1'], row['team2'], row['venue'])
                    winner = row['team1'] if prob >= 0.5 else row['team2']
                    s8_points[winner] += 2

                    # Aligning Match Output
                    m_col1, m_col2 = st.columns([3, 1])
                    m_col1.write(f"🏏 {row['team1']} vs {row['team2']}")
                    m_col2.markdown(f"**{winner}** ✅")

        # --- PHASE 2: SUPER 8s ---
        if st.button("Proceed to Super 8s"):
            st.session_state.sim_step = 2

        if st.session_state.sim_step >= 2:
            st.divider()
            st.markdown("### 📊 Step 2: Super Eight Stage")

            # Group definitions based on your results
            s8_groups = {
                "Group 1": ["India", "Australia", "West Indies", "South Africa"],
                "Group 2": ["Pakistan", "Sri Lanka", "England", "New Zealand"]
            }
            s8_results_points = {t: 0 for g in s8_groups.values() for t in g}

            # SIDE-BY-SIDE ALIGNMENT FOR SUPER 8 TABLES
            col1, col2 = st.columns(2)

            for i, (g_name, teams) in enumerate(s8_groups.items()):
                target_col = col1 if i == 0 else col2
                with target_col:
                    st.info(f"🏆 {g_name}")
                    import itertools

                    for t1, t2 in itertools.combinations(teams, 2):
                        p = get_prediction(t1, t2, "Mumbai")
                        win = t1 if p >= 0.5 else t2
                        s8_results_points[win] += 2

                        # Match-by-match inside Super 8
                        sm_c1, sm_c2 = st.columns([2, 1])
                        sm_c1.write(f"{t1} v {t2}")
                        sm_c2.write(f"**{win}**")

                    # Mini Points Table for alignment
                    standings = pd.DataFrame([{"Team": t, "Pts": s8_results_points[t]} for t in teams])
                    st.table(standings.sort_values("Pts", ascending=False))

        # --- PHASE 3: SEMI FINALS ---
        if st.button("Run Semi-Finals"):
            st.session_state.sim_step = 3

        if st.session_state.sim_step >= 3:
            st.divider()
            st.markdown("### ⚡ Step 3: Semi-Finals")

            # Logic for Top 2 (Mock logic for alignment)
            sf1_t1, sf1_t2 = "India", "New Zealand"
            sf2_t1, sf2_t2 = "England", "Australia"

            c_sf1, c_sf2 = st.columns(2)
            with c_sf1:
                st.write(f"SF1: {sf1_t1} vs {sf1_t2}")
                sf1_w = sf1_t1 if get_prediction(sf1_t1, sf1_t2, "Kolkata") >= 0.5 else sf1_t2
                st.success(f"Winner: **{sf1_w}**")

            with c_sf2:
                st.write(f"SF2: {sf2_t1} vs {sf2_t2}")
                sf2_w = sf2_t1 if get_prediction(sf2_t1, sf2_t2, "Mumbai") >= 0.5 else sf2_t2
                st.success(f"Winner: **{sf2_w}**")

            # --- PHASE 4: FINAL ---
            st.divider()
            st.markdown("### 👑 Step 4: The Grand Final")
            final_p = get_prediction(sf1_w, sf2_w, "Ahmedabad")
            champ = sf1_w if final_p >= 0.5 else sf2_w

            st.balloons()
            st.markdown(
                f"<div class='icc-card' style='text-align:center; border: 2px solid #fbbf24;'><h2>🏆 WORLD CHAMPION: {champ.upper()} 🏆</h2></div>",
                unsafe_allow_html=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("""
<div class='footer'>
© 2026 International Cricket Council – AI Analytics Dashboard  
Built with Streamlit • Designed with ICC Premier Theme
</div>
""", unsafe_allow_html=True)
