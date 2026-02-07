import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load your newly created summary file
df = pd.read_csv('world_cup_summary.csv')

# 2. Clean data: Drop any matches that ended in a 'No Result' (missing winner)
df = df.dropna(subset=['winner'])

# 3. Initialize LabelEncoders (Think of these as your "Translation Dictionaries")
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()

# We combine team1 and team2 to make sure both get the same number IDs
all_teams = pd.concat([df['team1'], df['team2']]).unique()
team_encoder.fit(all_teams)

# Encode the columns
df['team1_id'] = team_encoder.transform(df['team1'])
df['team2_id'] = team_encoder.transform(df['team2'])
df['toss_winner_id'] = team_encoder.transform(df['toss_winner'])
df['venue_id'] = venue_encoder.fit_transform(df['venue'])
df['winner_id'] = team_encoder.transform(df['winner'])

# 4. Define our Features (X) and our Target (y)
X = df[['team1_id', 'team2_id', 'toss_winner_id', 'venue_id']]
y = df['winner_id']

# 5. Split data (80% for training, 20% for testing the AI's accuracy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Create and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Test it!
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score * 100:.2f}%")


# --- TEST A PREDICTION ---
def predict_match(t1, t2, toss_w, venue):
    try:
        t1_id = team_encoder.transform([t1])[0]
        t2_id = team_encoder.transform([t2])[0]
        tw_id = team_encoder.transform([toss_w])[0]
        v_id = venue_encoder.transform([venue])[0]

        prediction_id = model.predict([[t1_id, t2_id, tw_id, v_id]])
        winner = team_encoder.inverse_transform(prediction_id)[0]
        print(f"PREDICTION: In a match between {t1} and {t2}, the winner is likely: {winner}")
    except Exception as e:
        print("Error: Make sure the team/venue names match your CSV exactly!")

# Example use (check your CSV for exact spelling):
# predict_match('India', 'Pakistan', 'India', 'Kensington Oval, Bridgetown')