import pandas as pd
import glob
import os

# --- 1. SET YOUR PATH TO THE 'ALL T20' FOLDER ---
# If your folder name is different, change it here!
path = r"C:\Users\User\OneDrive\Desktop\T20_WorldCup_Predictor\t20_matches"

all_files = glob.glob(os.path.join(path, "*.csv"))
print(f"I found {len(all_files)} files. Starting extraction of ALL T20 matches...")


def extract_match_info(file_path):
    match_data = {}
    teams = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == 'info':
                key = parts[1]

                # Capture the first date found
                if key == 'date' and 'date' not in match_data:
                    match_data['date'] = parts[2]

                # Capture teams
                elif key == 'team':
                    teams.append(parts[2])

                # Capture Toss and Winner
                elif key == 'toss_winner':
                    match_data['toss_winner'] = parts[2]
                elif key == 'winner':
                    match_data['winner'] = parts[2]
                elif key == 'venue':
                    match_data['venue'] = parts[2]
                elif key == 'outcome' and parts[2] == 'no result':
                    return None  # We skip 'No Result' matches as AI can't learn from them

    # Assign teams
    if len(teams) >= 2:
        match_data['team1'] = teams[0]
        match_data['team2'] = teams[1]

    return match_data


# 2. RUN EXTRACTION
all_matches = []
for file in all_files:
    if "summary" in file: continue
    data = extract_match_info(file)
    if data and 'winner' in data and 'toss_winner' in data:
        all_matches.append(data)

# 3. SAVE TO CSV
if all_matches:
    df = pd.DataFrame(all_matches)
    # Sort by date so you can see the newest matches at the bottom
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df.to_csv('all_t20_history_full.csv', index=False)

    print("\n" + "=" * 40)
    print(f"SUCCESS! Processed {len(all_matches)} total matches.")
    print(f"Latest Match Found: {df['date'].max().strftime('%Y-%m-%d')}")
    print("New File: all_t20_history_full.csv")
    print("=" * 40)
else:
    print("Error: No valid matches found. Check your folder path.")