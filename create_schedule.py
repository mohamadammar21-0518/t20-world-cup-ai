import pandas as pd

# Creating the schedule based on the groups and venues you provided
data = [
    # Group A
    {'match_no': 1, 'date': '2026-02-07', 'team1': 'Netherlands', 'team2': 'Pakistan',
     'venue': 'SSC Cricket Ground, Colombo', 'group': 'A'},
    {'match_no': 3, 'date': '2026-02-07', 'team1': 'India', 'team2': 'United States',
     'venue': 'Wankhede Stadium, Mumbai', 'group': 'A'},
    {'match_no': 10, 'date': '2026-02-10', 'team1': 'Namibia', 'team2': 'Netherlands',
     'venue': 'Arun Jaitley Stadium, Delhi', 'group': 'A'},
    {'match_no': 12, 'date': '2026-02-10', 'team1': 'Pakistan', 'team2': 'United States',
     'venue': 'SSC Cricket Ground, Colombo', 'group': 'A'},
    {'match_no': 18, 'date': '2026-02-12', 'team1': 'India', 'team2': 'Namibia', 'venue': 'Arun Jaitley Stadium, Delhi',
     'group': 'A'},
    {'match_no': 21, 'date': '2026-02-13', 'team1': 'Netherlands', 'team2': 'United States',
     'venue': 'M. A. Chidambaram Stadium, Chennai', 'group': 'A'},
    {'match_no': 27, 'date': '2026-02-15', 'team1': 'India', 'team2': 'Pakistan',
     'venue': 'R. Premadasa Stadium, Colombo', 'group': 'A'},

    # Group B
    {'match_no': 6, 'date': '2026-02-08', 'team1': 'Sri Lanka', 'team2': 'Ireland',
     'venue': 'R. Premadasa Stadium, Colombo', 'group': 'B'},
    {'match_no': 14, 'date': '2026-02-11', 'team1': 'Australia', 'team2': 'Ireland',
     'venue': 'R. Premadasa Stadium, Colombo', 'group': 'B'},
    {'match_no': 30, 'date': '2026-02-16', 'team1': 'Sri Lanka', 'team2': 'Australia',
     'venue': 'Pallekele Cricket Stadium, Kandy', 'group': 'B'},

    # Group C
    {'match_no': 2, 'date': '2026-02-07', 'team1': 'Scotland', 'team2': 'West Indies', 'venue': 'Eden Gardens, Kolkata',
     'group': 'C'},
    {'match_no': 5, 'date': '2026-02-08', 'team1': 'England', 'team2': 'Nepal', 'venue': 'Wankhede Stadium, Mumbai',
     'group': 'C'},
    {'match_no': 15, 'date': '2026-02-11', 'team1': 'England', 'team2': 'West Indies',
     'venue': 'Wankhede Stadium, Mumbai', 'group': 'C'},

    # Group D
    {'match_no': 4, 'date': '2026-02-08', 'team1': 'Afghanistan', 'team2': 'New Zealand',
     'venue': 'M. A. Chidambaram Stadium, Chennai', 'group': 'D'},
    {'match_no': 24, 'date': '2026-02-14', 'team1': 'New Zealand', 'team2': 'South Africa',
     'venue': 'Narendra Modi Stadium, Ahmedabad', 'group': 'D'},
]

# Convert to DataFrame
schedule_df = pd.DataFrame(data)

# Save to CSV
schedule_df.to_csv('wc_2026_schedule.csv', index=False)
print("Schedule CSV created successfully!")