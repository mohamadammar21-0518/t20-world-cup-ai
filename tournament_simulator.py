import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import itertools
from xgboost import XGBClassifier


def get_time_weight(year):
    if year >= 2025: return 1.5  # Heavy weight for current year
    if year == 2024: return 1.3
    if year == 2023: return 1.1
    return 0.7  # Older matches are less relevant


# --- NEW FEATURE: HEAD-TO-HEAD CALCULATOR ---
def get_h2h_win_rate(t1, t2, df):
    # Look for all past matches between these two specific teams
    matches = df[((df['team1'] == t1) & (df['team2'] == t2)) |
                 ((df['team1'] == t2) & (df['team2'] == t1))]

    if len(matches) < 3:
        return 0.5  # Neutral if they haven't played much

    t1_wins = len(matches[matches['winner'] == t1])
    return t1_wins / len(matches)



# --- 1. LOAD DATA ---
hist_path = "all_t20_history_full.csv"
sched_path = r"C:\Users\User\OneDrive\Desktop\T20_WorldCup_Predictor\wc_2026_schedule.csv"

# Load BOTH files at the very start
historical_df = pd.read_csv(hist_path).dropna(subset=['winner'])
schedule_df = pd.read_csv(sched_path) # <--- MOVE THIS HERE


# Now apply the 2018 filter
historical_df['date'] = pd.to_datetime(historical_df['date'])
historical_df = historical_df[historical_df['date'].dt.year >= 2018].copy()

# Add a simple Toss Feature (1 if Toss Winner won the match, 0 otherwise)
historical_df['toss_match_winner'] = (historical_df['toss_winner'] == historical_df['winner']).astype(int)

# --- NEW FEATURE: VENUE AVERAGE SCORE ---
# (Assumes your CSV has a 'runs' or 'score' column; if not, you can skip this)
if 'target_runs' in historical_df.columns:
    venue_avg_runs = historical_df.groupby('venue')['target_runs'].transform('mean')
    historical_df['venue_avg'] = venue_avg_runs / 250  # Normalized
else:
    historical_df['venue_avg'] = 0.6  # Default average


# --- 2026 WORLD CUP TEAM FILTER ---
world_cup_teams = [
    'India', 'Pakistan', 'USA', 'Netherlands', 'Namibia',
    'Sri Lanka', 'Australia', 'Zimbabwe', 'Ireland', 'Oman',
    'England', 'West Indies', 'Nepal', 'Italy', 'Scotland',
    'South Africa', 'Afghanistan', 'New Zealand', 'Canada', 'United Arab Emirates'
]

# CHANGE 'df' TO 'historical_df' HERE:
df_filtered = historical_df[historical_df['team1'].isin(world_cup_teams) & historical_df['team2'].isin(world_cup_teams)]

# Save the focused dataset
df_filtered.to_csv('t20_wc_focused_history.csv', index=False)

print(f"Filtering Complete! World Cup matches: {len(df_filtered)}")


# --- 2. OFFICIAL 2026 TEAMS & KNOCKOUT VENUES ---
official_20_teams = [
    'India', 'Sri Lanka', 'Afghanistan', 'Australia', 'England',
    'South Africa', 'USA', 'West Indies', 'Ireland', 'New Zealand',
    'Pakistan', 'Canada', 'Netherlands', 'Italy', 'Zimbabwe',
    'Namibia', 'Nepal', 'Oman', 'UAE', 'Scotland'
]

# Cities we need for the Finals simulation
knockout_venues = ['Colombo', 'Mumbai', 'Ahmedabad', 'Kandy', 'Barbados']

# --- 3. CURRENT FORM FACTORS (2026 UPDATED) ---
# --- 3. CURRENT FORM FACTORS (FEB 2026 UPDATED) ---
current_form = {
    # The Giants
    'India': 1.25,  # Rank 1, just beat NZ 4-1, won warm-up vs SA
    'South Africa': 1.15,  # Strong warm-up, current form is peaking
    'England': 1.10,  # Consistent, solid squad depth
    'New Zealand': 1.05,  # Solid, though lost recent series to India
    'West Indies': 1.08,  # Power hitters finding form (Hetmyer/Hope)

    # The "Vulnerable" Big Teams
    'Australia': 0.95,  # Struggling for form, recently outplayed in Pakistan
    'Pakistan': 0.92,  # Unpredictable, leadership changes, internal tensions
    'Sri Lanka': 1.08,  # Co-hosts, historically strong in these conditions

    # The Rising Stars (Associate/Lower Full Members)
    'Afghanistan': 1.12,  # Dominant in warm-ups vs WI, world-class spinners
    'USA': 0.90,  # Lost both warm-ups, bowling needs work
    'Nepal': 1.05,  # Strong players (Bhurtel/Sheikh), high momentum
    'Scotland': 1.02,  # Consistent disruptors
    'Netherlands': 1.00,  # Disciplined, but raw power is lacking

    # The "Zimbabwe Fix"
    'Zimbabwe': 0.88,  # Lowered to prevent "Giant Killer" glitch
    'Ireland': 0.90,  # Consistent but struggling against top-tier pace
    'Namibia': 0.85,
    'Oman': 0.82,
    'UAE': 0.80,
    'Canada': 0.78,
    'Italy': 0.75  # Debutants, lowest experience
}

# --- NEW: ICC SQUAD RATINGS (2026) ---
# Based on average ICC rankings of top 11 players
icc_squad_ratings = {
    'India': 88.5, 'Australia': 87.2, 'England': 85.0, 'South Africa': 82.1,
    'Pakistan': 80.5, 'West Indies': 79.8, 'New Zealand': 81.2, 'Sri Lanka': 77.5,
    'Afghanistan': 76.0, 'USA': 62.0, 'Ireland': 70.5, 'Netherlands': 68.0,
    'Scotland': 65.5, 'Namibia': 63.0, 'Nepal': 61.5, 'Oman': 60.0,
    'UAE': 59.5, 'Canada': 58.0, 'Italy': 55.0, 'Zimbabwe': 69.0
}


def calculate_form_strength(df):
    strengths = {}
    # Benchmark: We expect a top team to have played at least 30 matches
    benchmark_matches = 50

    for team in official_20_teams:
        team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
        actual_matches = len(team_matches)
        wins = len(team_matches[team_matches['winner'] == team])

        if actual_matches > 0:
            # BAYESIAN SMOOTHING:
            # We add 10 "virtual matches" where the team only wins 30% of the time.
            # This 'pulls' small teams toward a lower average until they play more.
            smoothed_win_rate = (wins + (10 * 0.3)) / (actual_matches + 10)
        else:
            smoothed_win_rate = 0.1

        player_power = icc_squad_ratings.get(team, 50.0) / 100

        # EXPERIENCE MULTIPLIER: Scale strength by how 'proven' they are
        # If they've played 30+ matches, they get 100% of their strength.
        experience_factor = min(1.0, actual_matches / benchmark_matches)

        # 70% Weight on Player Power (The 'Real' quality)
        # 30% Weight on smoothed win rate (The 'Form')
        raw_strength = (smoothed_win_rate * 0.3 + player_power * 0.7) * current_form.get(team, 1.0)

        # Apply the experience factor to 'dampen' unproven teams
        strengths[team] = raw_strength * experience_factor

    return strengths


team_strength = calculate_form_strength(historical_df)


# --- 4. ENCODING & PREPARING FEATURES (UPGRADED TO ONE-HOT) ---

# 1. Filter historical data first (Problem 4 Fix)
historical_df = historical_df[historical_df['team1'].isin(official_20_teams) &
                              historical_df['team2'].isin(official_20_teams)].copy()

# 2. Recalculate strength on the FILTERED data
team_strength = calculate_form_strength(historical_df)

# 3. Add numerical features to the dataframe
historical_df['t1_s'] = historical_df['team1'].map(team_strength)
historical_df['t2_s'] = historical_df['team2'].map(team_strength)
historical_df['t1_player_pwr'] = historical_df['team1'].map(icc_squad_ratings) / 100
historical_df['t2_player_pwr'] = historical_df['team2'].map(icc_squad_ratings) / 100

# 4. ONE-HOT ENCODING (Problem 2 Fix)
# We create dummy columns for team1 and team2
team_dummies = pd.get_dummies(historical_df[['team1', 'team2']], prefix=['t1', 't2'])

# 5. Handle Venue encoding
all_venues = pd.concat([historical_df['venue'], schedule_df['venue'], pd.Series(knockout_venues)]).unique()
venue_encoder = LabelEncoder().fit(all_venues)
historical_df['v_id'] = venue_encoder.transform(historical_df['venue'])

# Calculate H2H for every row in the history
historical_df['h2h_rate'] = historical_df.apply(
    lambda x: get_h2h_win_rate(x['team1'], x['team2'], historical_df), axis=1
)

# Update X to include the new features
X = pd.concat([
    historical_df[['v_id', 't1_s', 't2_s', 't1_player_pwr', 't2_player_pwr', 'h2h_rate', 'venue_avg']],
    team_dummies
], axis=1)


# Target: We need to encode the winner as a binary (Did Team 1 win? 1 for Yes, 0 for No)
y = (historical_df['winner'] == historical_df['team1']).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# --- 5. ENCODING & TRAINING (OPTIMIZED XGBOOST) ---
model = XGBClassifier(
    n_estimators=500,        # Increased from 200
    learning_rate=0.1,       # Lowered for more precise "learning"
    max_depth=2,              # Shallower trees prevent overfitting on small data
    min_child_weight=10,       # Forces the model to only find patterns in multiple matches
    subsample=0.8,            # Uses 70% of data to ensure variety
    colsample_bytree=0.8,     # Uses 70% of features per tree
    gamma=0.2,                # Adds a "penalty" for making complex branches
    eval_metric='logloss',
    early_stopping_rounds=50, # Stops training if accuracy stops improving
    random_state=42
).fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # Monitors accuracy while it trains
    verbose=False
)

# Report New Accuracy
y_pred = model.predict(X_test)
print(f"New Tuned Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
# --- 6. SIMULATION LOGIC WITH HOME ADVANTAGE ---
trophy_count = {'Australia': 1, 'India': 2, 'West Indies': 2, 'England': 2, 'Pakistan': 1, 'Sri Lanka': 1}
# Define which team is 'Home' for which venues
home_nations = {
    'Mumbai': 'India', 'Ahmedabad': 'India', 'Kolkata': 'India', 'Chennai': 'India','Delhi': 'India',
    'Colombo (SSC)': 'Sri Lanka', 'Kandy': 'Sri Lanka', 'Colombo (RPS)': 'Sri Lanka'
}


def predict_match(t1, t2, venue, is_knockout=False):
    # 1. Initialize DataFrame with all 0s
    match_feat = pd.DataFrame(0.0, index=[0], columns=X.columns)

    # 2. Basic Stats & Strengths
    s1 = team_strength.get(t1, 0.1)
    s2 = team_strength.get(t2, 0.1)
    p1_pwr = icc_squad_ratings.get(t1, 50.0) / 100
    p2_pwr = icc_squad_ratings.get(t2, 50.0) / 100

    # 3. Apply Contextual Boosts (Home/Knockout)
    if home_nations.get(venue) == t1:
        s1 *= 1.15
    elif home_nations.get(venue) == t2:
        s2 *= 1.15

    if is_knockout:
        s1 *= (1 + (trophy_count.get(t1, 0) * 0.05))
        s2 *= (1 + (trophy_count.get(t2, 0) * 0.05))

    # 4. Fill the Feature Vector
    match_feat.at[0, 'v_id'] = float(
        venue_encoder.transform([venue if venue in venue_encoder.classes_ else venue_encoder.classes_[0]])[0])
    match_feat.at[0, 't1_s'] = float(s1)
    match_feat.at[0, 't2_s'] = float(s2)
    match_feat.at[0, 't1_player_pwr'] = float(p1_pwr)
    match_feat.at[0, 't2_player_pwr'] = float(p2_pwr)
    match_feat.at[0, 'h2h_rate'] = float(get_h2h_win_rate(t1, t2, historical_df))

    # Venue Avg logic
    v_avg = historical_df[historical_df['venue'] == venue]['venue_avg'].mean() if venue in historical_df[
        'venue'].values else 0.6
    match_feat.at[0, 'venue_avg'] = float(v_avg)

    # 5. Set One-Hot Team Flags (The Critical Step)
    if f't1_{t1}' in match_feat.columns: match_feat.at[0, f't1_{t1}'] = 1.0
    if f't2_{t2}' in match_feat.columns: match_feat.at[0, f't2_{t2}'] = 1.0

    # 6. Predict Probability
    prob_t1_wins = model.predict_proba(match_feat)[0][1]

    # Toss luck should be small (1% max) to keep things realistic
    toss_luck = np.random.uniform(-0.01, 0.01)
    return t1 if (prob_t1_wins + toss_luck) >= 0.5 else t2

# --- 7. RUN TOURNAMENT (Updated for Pak Forfeit) ---
print("\n--- 2026 WORLD CUP SIMULATION RESULTS ---")

for _, row in schedule_df.iterrows():
    # Check for India vs Pakistan specifically
    current_match = {row['team1'], row['team2']}

    if current_match == {'India', 'Pakistan'}:
        winner = "India"
        print(f"GROUP MATCH: India vs Pakistan -> Winner: {winner} (BY FORFEIT)")
    else:
        # Normal prediction for all other matches
        winner = predict_match(row['team1'], row['team2'], row['venue'])
        print(f"GROUP MATCH: {row['team1']} vs {row['team2']} -> Winner: {winner}")
# SUPER 8 SIMULATION
s8_g1 = ['India', 'Australia', 'West Indies', 'South Africa']
s8_g2 = ['Pakistan', 'Sri Lanka', 'England', 'New Zealand']


def run_s8_phase(teams, label):
    print(f"\n--- {label} ---")
    results = []
    for t1, t2 in itertools.combinations(teams, 2):
        w = predict_match(t1, t2, "Colombo")
        print(f"{t1} vs {t2} -> {w}")
        results.append(w)
    return pd.Series(results).value_counts().index.tolist()[:2]


top_g1 = run_s8_phase(s8_g1, "SUPER 8 - GROUP 1")
top_g2 = run_s8_phase(s8_g2, "SUPER 8 - GROUP 2")

# FINALS
print("\n" + "=" * 40 + "\n          THE KNOCKOUT STAGE\n" + "=" * 40)
sf1 = predict_match(top_g1[0], top_g2[1], "Mumbai", True)
print(f"SEMI-FINAL 1: {top_g1[0]} vs {top_g2[1]} -> WINNER: {sf1}")

sf2 = predict_match(top_g2[0], top_g1[1], "Colombo", True)
print(f"SEMI-FINAL 2: {top_g2[0]} vs {top_g1[1]} -> WINNER: {sf2}")

champion = predict_match(sf1, sf2, "Ahmedabad", True)
print(f"\n🏆 THE 2026 WORLD CHAMPION IS: {champion.upper()} 🏆")

import joblib

# 1. Save the XGBoost Model
joblib.dump(model, 't20_model.joblib')

# 2. Save the Venue Encoder (so the app knows how to turn 'Mumbai' into 'ID 5')
joblib.dump(venue_encoder, 'venue_encoder.joblib')

# 3. Save your Team Strength dictionary
# (Since this is a custom dictionary you made, the app needs it!)
joblib.dump(team_strength, 'team_strength.joblib')

print("Files saved successfully: t20_model.joblib, venue_encoder.joblib, team_strength.joblib")

# Add this to the end of your training code
joblib.dump(icc_squad_ratings, 'icc_ratings.joblib')
joblib.dump(X.columns.tolist(), 'feature_columns.joblib')