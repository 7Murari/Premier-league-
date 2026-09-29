import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

st.set_page_config(page_title="PL Player Valuation Engine", layout="wide")

st.title("⚽ Premier League Player Valuation Engine (Deduplicated Ground Truth)")

@st.cache_data
def load_and_preprocess_csv():
    try:
        df = pd.read_csv("pl_players_3yr_complete.csv")
    except FileNotFoundError:
        st.error("❌ Local file 'pl_players_3yr_complete.csv' not found. Run 'build_dataset.py' first!")
        return pd.DataFrame()

    def find_col(possible_names):
        for col in df.columns:
            for name in possible_names:
                if name.lower() in str(col).lower():
                    return col
        return None

    col_player = find_col(['standard_player', 'player'])
    col_goals = find_col(['standard_gls', 'gls', 'goals'])
    col_assists = find_col(['standard_ast', 'ast', 'assists'])
    col_minutes = find_col(['standard_min', 'min', 'minutes'])
    col_age = find_col(['standard_age', 'age'])
    col_pos = find_col(['standard_pos', 'pos'])
    col_season = find_col(['standard_season', 'season'])
    col_tm_val = find_col(['real_transfermarkt_val', 'market_value'])

    def clean_metric(col):
        if not col or col not in df.columns:
            return pd.Series(0, index=df.index)
        val = pd.to_numeric(df[col].squeeze(), errors='coerce').fillna(0)
        if isinstance(val, pd.DataFrame): val = val.iloc[:, 0]
        return val

    df['goals'] = clean_metric(col_goals)
    df['assists'] = clean_metric(col_assists)
    df['minutes'] = clean_metric(col_minutes)
    df['player_clean'] = df[col_player].astype(str)
    
    if col_tm_val and col_tm_val in df.columns:
        df['tm_val'] = pd.to_numeric(df[col_tm_val].squeeze(), errors='coerce')
    else:
        df['tm_val'] = np.nan

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
    
    season_col_name = col_season if col_season else 'season'
    df['season_id'] = df[season_col_name].astype(str)

    # Clean Aggregation (Grouping by Player)
    agg_df = df.groupby('player_clean').agg({
        'goals': 'sum',
        'assists': 'sum',
        'minutes': 'sum',
        'current_calculated_age': 'max',
        'pos_clean': 'last',
        'season_id': 'nunique',
        'tm_val': 'max'
    }).reset_index()

    agg_df.columns = [
        'name', 'total_goals', 'total_assists', 'total_minutes', 'age', 'position', 'seasons_played', 'transfermarkt_val'
    ]

    agg_df = agg_df[agg_df['total_minutes'] >= 270].copy()

    # Per-Season Rates
    agg_df['goals_per_yr'] = agg_df['total_goals'] / agg_df['seasons_played']
    agg_df['assists_per_yr'] = agg_df['total_assists'] / agg_df['seasons_played']
    agg_df['minutes_per_yr'] = agg_df['total_minutes'] / agg_df['seasons_played']

    def simplify_position(pos):
        pos = str(pos).upper()
        if 'FW' in pos: return 'Offence'
        elif 'MF' in pos: return 'Midfield'
        elif 'DF' in pos: return 'Defence'
        elif 'GK' in pos: return 'Goalkeeper'
        return 'Midfield'

    agg_df['position'] = agg_df['position'].apply(simplify_position)

    # Fallback Valuation
    def fallback_val(row):
        if pd.notna(row['transfermarkt_val']) and row['transfermarkt_val'] > 0:
            return round(row['transfermarkt_val'], 2)
        
        pos = row['position']
        if pos == 'Defence': base = (row['minutes_per_yr'] / 2700 * 35.0) + (row['goals_per_yr'] * 2.0)
        elif pos == 'Midfield': base = (row['goals_per_yr'] * 4.0) + (row['assists_per_yr'] * 3.5) + (row['minutes_per_yr'] / 2700 * 20.0)
        else: base = (row['goals_per_yr'] * 5.5) + (row['assists_per_yr'] * 3.0) + (row['minutes_per_yr'] / 2700 * 15.0)
        
        return round(max(10.0, base), 2)

    agg_df['actual_market_value'] = agg_df.apply(fallback_val, axis=1)

    # Train Ridge Model
    df_encoded = pd.get_dummies(agg_df, columns=['position'], drop_first=False)
    df_encoded['age_squared'] = df_encoded['age'] ** 2

    exclude_cols = ['name', 'transfermarkt_val', 'actual_market_value']
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

    X = df_encoded[feature_cols]
    y = df_encoded['actual_market_value']

    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    agg_df['model_estimated_value'] = model.predict(X)
    agg_df['model_estimated_value'] = agg_df['model_estimated_value'].apply(lambda v: round(max(5.0, v), 2))

    return agg_df

df_players = load_and_preprocess_csv()

if not df_players.empty:
    st.sidebar.header("Filter & Select")
    all_player_names = sorted(df_players['name'].unique())

    selected_players = st.multiselect(
        "Select Player(s) to Compare Valuation:",
        options=all_player_names,
        default=["Erling Haaland", "William Saliba", "Martin Ødegaard", "Cole Palmer"] if all(p in all_player_names for p in ["Erling Haaland", "William Saliba", "Martin Ødegaard", "Cole Palmer"]) else all_player_names[:4]
    )

    if selected_players:
        filtered_df = df_players[df_players['name'].isin(selected_players)].copy()

        st.subheader("📊 Transfermarkt Market Value vs. Model Estimation (€ Millions)")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(filtered_df))
        width = 0.35

        rects1 = ax.bar(x - width/2, filtered_df['actual_market_value'], width, label='Real Transfermarkt Value', color='#1f77b4')
        rects2 = ax.bar(x + width/2, filtered_df['model_estimated_value'], width, label="Model Prediction", color='#ff7f0e')

        ax.set_ylabel('Valuation (€ Millions)')
        ax.set_xticks(x)
        ax.set_xticklabels(filtered_df['name'], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.bar_label(rects1, padding=3, fmt='€%.1fM')
        ax.bar_label(rects2, padding=3, fmt='€%.1fM')

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📋 Overview of Selected Players")
        display_cols = ['name', 'position', 'age', 'seasons_played', 'total_goals', 'total_assists', 'actual_market_value', 'model_estimated_value']
        
        st.dataframe(filtered_df[display_cols].rename(columns={
            'name': 'Player',
            'position': 'Pos',
            'age': 'Age',
            'seasons_played': 'PL Seasons',
            'total_goals': 'Goals',
            'total_assists': 'Assists',
            'actual_market_value': 'Transfermarkt (€M)',
            'model_estimated_value': 'Model Val (€M)'
        }), use_container_width=True)