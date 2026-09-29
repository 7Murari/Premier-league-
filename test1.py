import soccerdata as sd
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime
from sklearn.linear_model import LinearRegression

print("--> Fetching exact 3-year Premier League player stats from FBref (23/24, 24/25, 25/26)...")

# 1. INITIALIZE FBREF SCRAPER (23/24, 24/25, 25/26)
fbref = sd.FBref(leagues="ENG-Premier League", seasons=["2324", "2425", "2526"])

try:
    raw_df = fbref.read_player_season_stats(stat_type="standard")

    # Flatten MultiIndex headers
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = [f"{col[0]}_{col[1]}" if col[1] else str(col[0]) for col in raw_df.columns]
    
    df = raw_df.reset_index()

    def find_col(possible_names):
        for col in df.columns:
            for name in possible_names:
                if name.lower() in str(col).lower():
                    return col
        return None

    col_player = find_col(['player'])
    col_goals = find_col(['performance_gls', 'gls', 'goals'])
    col_assists = find_col(['performance_ast', 'ast', 'assists'])
    col_minutes = find_col(['playing time_min', 'min', 'minutes'])
    col_age = find_col(['age'])
    col_pos = find_col(['pos'])
    col_season = find_col(['season'])

    # Safely convert metrics
    goals_s = pd.to_numeric(df[col_goals].squeeze() if col_goals else 0, errors='coerce').fillna(0)
    assists_s = pd.to_numeric(df[col_assists].squeeze() if col_assists else 0, errors='coerce').fillna(0)
    minutes_s = pd.to_numeric(df[col_minutes].squeeze() if col_minutes else 0, errors='coerce').fillna(0)

    if isinstance(goals_s, pd.DataFrame): goals_s = goals_s.iloc[:, 0]
    if isinstance(assists_s, pd.DataFrame): assists_s = assists_s.iloc[:, 0]
    if isinstance(minutes_s, pd.DataFrame): minutes_s = minutes_s.iloc[:, 0]

    df['goals_clean'] = goals_s
    df['assists_clean'] = assists_s
    df['minutes_clean'] = minutes_s
    df['player_clean'] = df[col_player].astype(str)

    # Real-Time Current Age Calculation
    current_year = datetime.now().year
    
    if col_age and col_season:
        age_num = pd.to_numeric(df[col_age].astype(str).str.split('-').str[0], errors='coerce').fillna(25)
        season_str = df[col_season].astype(str)
        season_year = pd.to_numeric("20" + season_str.str[:2], errors='coerce').fillna(2024)
        df['current_calculated_age'] = age_num + (current_year - season_year)
    else:
        df['current_calculated_age'] = 26

    pos_str = df[col_pos].squeeze() if col_pos else 'Midfield'
    if isinstance(pos_str, pd.DataFrame): pos_str = pos_str.iloc[:, 0]
    df['pos_clean'] = pos_str.astype(str)

    print("--> Aggregating 3-year totals per player (2023–2026)...")

    # 2. AGGREGATE 3-YEAR TOTALS PER PLAYER
    agg_df = df.groupby('player_clean').agg({
        'goals_clean': 'sum',
        'assists_clean': 'sum',
        'minutes_clean': 'sum',
        'current_calculated_age': 'max',
        'pos_clean': 'last'
    }).reset_index()

    agg_df.columns = ['name', 'goals_3yr', 'assists_3yr', 'minutes_3yr', 'age', 'position']

    # Normalize String Names (Strips accents: e.g., Núñez -> Nunez)
    def normalize_str(text):
        return ''.join(
            c for c in unicodedata.normalize('NFD', str(text))
            if unicodedata.category(c) != 'Mn'
        ).lower()

    agg_df['search_name'] = agg_df['name'].apply(normalize_str)

    def simplify_position(pos):
        pos = str(pos).upper()
        if 'FW' in pos: return 'Offence'
        elif 'MF' in pos: return 'Midfield'
        elif 'DF' in pos: return 'Defence'
        elif 'GK' in pos: return 'Goalkeeper'
        return 'Midfield'

    agg_df['position'] = agg_df['position'].apply(simplify_position)

    # Calculate Per-Season Averages
    agg_df['goals_per_season'] = agg_df['goals_3yr'] / 3.0
    agg_df['assists_per_season'] = agg_df['assists_3yr'] / 3.0
    agg_df['minutes_per_season'] = agg_df['minutes_3yr'] / 3.0

    agg_df = agg_df[agg_df['minutes_3yr'] >= 270].copy()

    # 3. REALISTIC MARKET VALUATION FORMULA (€M)
    def calculate_valuation(row):
        g_val = row['goals_per_season'] * 4.2
        a_val = row['assists_per_season'] * 2.5
        m_val = (row['minutes_per_season'] / 2700) * 15
        
        if row['age'] <= 23: age_factor = 1.45
        elif 24 <= row['age'] <= 27: age_factor = 1.35
        elif 28 <= row['age'] <= 30: age_factor = 1.00
        else: age_factor = 0.55
            
        raw_val = (g_val + a_val + m_val) * age_factor
        return round(max(3.0, raw_val), 2)

    agg_df['transfer_value'] = agg_df.apply(calculate_valuation, axis=1)

    # 4. FIT MODEL
    df_encoded = pd.get_dummies(agg_df, columns=['position'], drop_first=False)
    feature_cols = [c for c in df_encoded.columns if c not in ['name', 'search_name', 'transfer_value', 'goals_3yr', 'assists_3yr', 'minutes_3yr']]

    X = df_encoded[feature_cols]
    y = df_encoded['transfer_value']

    model = LinearRegression()
    model.fit(X, y)
    agg_df['predicted_value'] = model.predict(X)

    print(f"\n✅ Successfully processed {len(agg_df)} Premier League players across 2023–2026 seasons!")

    # 5. INTERACTIVE SEARCH LOOP
    print("\n=== Model Trained on 2023–2026 FBref Dataset ===")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter a player's name (e.g., Saka, Haaland, Nunez, Palmer): ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue

        normalized_query = normalize_str(query)
        matches = agg_df[agg_df['search_name'].str.contains(normalized_query, na=False)]
        
        if matches.empty:
            print(f"❌ No player found matching '{query}'.\n")
        else:
            for idx, row in matches.iterrows():
                pred_val = round(row['predicted_value'], 2)
                
                print("\n-------------------------------------------")
                print(f"⚽ Player: {row['name']} ({row['position']}, Age {int(row['age'])})")
                print(f"📈 3-Year Totals (23/24-25/26): {int(row['goals_3yr'])} Goals | {int(row['assists_3yr'])} Assists | {int(row['minutes_3yr'])} Mins")
                print(f"📊 Annual Averages: {row['goals_per_season']:.1f} Goals/Yr | {row['assists_per_season']:.1f} Assists/Yr")
                print(f"💰 Target Market Value: €{row['transfer_value']}M")
                print(f"🤖 Model Valuation: €{max(3.0, pred_val)}M")
                print("-------------------------------------------\n")

except Exception as e:
    print(f"An error occurred while processing: {e}")