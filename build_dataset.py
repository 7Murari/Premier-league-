import soccerdata as sd
import pandas as pd
from rapidfuzz import process, fuzz

print("--> 1. Fetching FBref Premier League Stats (2023–2026)...")

fbref = sd.FBref(leagues="ENG-Premier League", seasons=["2324", "2425", "2526"])

def fetch_table_flat(stat_type):
    try:
        raw_df = fbref.read_player_season_stats(stat_type=stat_type)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [f"{stat_type}_{col[1]}" if col[1] else f"{stat_type}_{col[0]}" for col in raw_df.columns]
        return raw_df.reset_index()
    except Exception as e:
        print(f"❌ Failed to fetch {stat_type}: {e}")
        return None

df_std = fetch_table_flat("standard")

if df_std is not None:
    std_player_col = [c for c in df_std.columns if 'player' in c.lower()][0]
    std_season_col = [c for c in df_std.columns if 'season' in c.lower()][0]
    
    # Remove duplicate season rows before merging to prevent goal inflation
    df_std = df_std.drop_duplicates(subset=[std_player_col, std_season_col]).copy()

    print("--> 2. Fetching Real Transfermarkt Valuations...")
    tm_url = "https://raw.githubusercontent.com/dexplo/transfermarkt-datasets/master/data/players.csv"

    try:
        tm_df = pd.read_csv(tm_url)
        tm_pl = tm_df.dropna(subset=['market_value_in_eur']).copy()
        tm_pl['tm_market_value_m'] = tm_pl['market_value_in_eur'] / 1e6
        
        tm_names = tm_pl['name'].tolist()

        # Explicit Override Dictionary for Known Discrepancies
        EXACT_OVERRIDE_MAP = {
            "William Saliba": 80.0,
            "Martin Ødegaard": 110.0,
            "Erling Haaland": 180.0,
            "Cole Palmer": 90.0,
            "Bukayo Saka": 140.0,
            "Gabriel Magalhães": 75.0,
            "Joško Gvardiol": 75.0,
            "Cristian Romero": 65.0,
            "Declan Rice": 120.0,
            "Rodri": 130.0,
            "Bruno Guimarães": 85.0
        }

        def get_market_value(fbref_name):
            if fbref_name in EXACT_OVERRIDE_MAP:
                return EXACT_OVERRIDE_MAP[fbref_name]
            
            match = process.extractOne(fbref_name, tm_names, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 82:
                matched_row = tm_pl[tm_pl['name'] == match[0]].iloc[0]
                return matched_row['tm_market_value_m']
            return None

        unique_players = df_std[std_player_col].unique()
        val_map = {name: get_market_value(name) for name in unique_players}
        df_std['real_transfermarkt_val'] = df_std[std_player_col].map(val_map)

        print(f"✅ Matched Transfermarkt values successfully!")

    except Exception as e:
        print(f"⚠️ Error pulling Transfermarkt data: {e}")
        df_std['real_transfermarkt_val'] = None

    # Save cleaned dataset
    df_std.to_csv("pl_players_3yr_complete.csv", index=False)
    print("✅ Saved clean dataset to 'pl_players_3yr_complete.csv'!")