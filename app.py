import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import unicodedata
import hashlib
import time

# Page config with custom styling and performance optimizations
st.set_page_config(
    layout="wide",
    page_title="FM24 Player Ranker",
    page_icon="⚽",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/streamlit/streamlit',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# FM24 Player Ranker\nBuilt for Football Manager 2024 player analysis"
    }
)

# Custom CSS for better styling and a dark theme focus
st.markdown("""
<style>
    /* Base styles for dark theme */
    body {
        color: #fafafa;
    }
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    /* Updated card styles for dark theme */
    .section-card {
        background: #1f2c38; /* Darker background */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #1f4e79;
        color: #fafafa; /* Light text */
    }
    .info-box {
        background: #1a3a5a; /* Darker info box */
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4169e1;
        margin-bottom: 1rem;
        color: #fafafa;
    }
    .metric-card {
        background: #1f2c38; /* Darker metric card */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #3c4b5a;
        color: #fafafa;
    }
    /* Brighter header colors for metric cards */
    .metric-card h3.blue { color: #66b2ff; }
    .metric-card h3.green { color: #76ff7a; }
    .metric-card h3.purple { color: #c792ea; }
    .metric-card h3.orange { color: #ffcb6b; }

    .role-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
.xi-formation {
    background: #1f2c38;
    padding: 2rem;
    border-radius: 10px;
    color: white;
    font-family: monospace;
    border: 3px solid #3c4b5a;
    min-height: 500px;
}
    .stProgress .st-bo {
        background-color: #e8f4fd;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #1f2c38;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for team building and user preferences
if 'custom_first_xi' not in st.session_state:
    st.session_state.custom_first_xi = {}
if 'custom_second_xi' not in st.session_state:
    st.session_state.custom_second_xi = {}
if 'use_custom_teams' not in st.session_state:
    st.session_state.use_custom_teams = False
if 'last_upload_time' not in st.session_state:
    st.session_state.last_upload_time = None
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None

# Initialize persistent user preferences
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_view': 'Full Table',
        'auto_refresh': False,
        'show_advanced_stats': False,
        'theme_preference': 'dark'
    }

# Function to save preferences to file
def save_preferences():
    """Save user preferences to a file"""
    import json
    import os
    
    # Create a hidden directory for preferences
    prefs_dir = ".streamlit"
    os.makedirs(prefs_dir, exist_ok=True)
    
    prefs_file = os.path.join(prefs_dir, "user_preferences.json")
    with open(prefs_file, 'w') as f:
        json.dump(st.session_state.user_preferences, f)

# Function to load preferences from file
def load_preferences():
    """Load user preferences from file"""
    import json
    import os
    
    prefs_file = os.path.join(".streamlit", "user_preferences.json")
    if os.path.exists(prefs_file):
        try:
            with open(prefs_file, 'r') as f:
                saved_prefs = json.load(f)
                st.session_state.user_preferences.update(saved_prefs)
        except (json.JSONDecodeError, FileNotFoundError):
            pass  # Use default preferences if file is corrupted

# Load preferences on startup
load_preferences()

# Header
st.markdown("""
<div class="main-header">
    <h1>FM24 Player Ranker</h1>
</div>
""", unsafe_allow_html=True)

CANONICAL_ATTRIBUTES = [
    "Corners", "Crossing", "Dribbling", "Finishing", "First Touch", "Free Kick Taking",
    "Heading", "Long Shots", "Long Throws", "Marking", "Passing", "Penalty Taking",
    "Tackling", "Technique", "Aggression", "Anticipation", "Bravery", "Composure",
    "Concentration", "Decisions", "Determination", "Flair", "Leadership", "Off The Ball",
    "Positioning", "Teamwork", "Vision", "Work Rate", "Acceleration", "Agility",
    "Balance", "Jumping Reach", "Natural Fitness", "Pace", "Stamina", "Strength",
    "Weaker Foot", "Aerial Reach", "Command of Area", "Communication", "Eccentricity",
    "Handling", "Kicking", "One on Ones", "Punching (Tendency)", "Reflexes",
    "Rushing Out (Tendency)", "Throwing"
]

WEIGHTS_BY_ROLE = {
    "GK": {
        "Corners": 0.0, "Crossing": 0.0, "Dribbling": 0.0, "Finishing": 0.0, "First Touch": 0.0, "Free Kick Taking": 0.0,
        "Heading": 1.0, "Long Shots": 0.0, "Long Throws": 0.0, "Marking": 0.0, "Passing": 0.0, "Penalty Taking": 0.0,
        "Tackling": 0.0, "Technique": 1.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 6.0, "Composure": 2.0,
        "Concentration": 6.0, "Decisions": 10.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 2.0, "Off The Ball": 0.0,
        "Positioning": 5.0, "Teamwork": 2.0, "Vision": 1.0, "Work Rate": 1.0, "Acceleration": 6.0, "Agility": 8.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 3.0, "Stamina": 1.0, "Strength": 4.0,
        "Weaker Foot": 3.0, "Aerial Reach": 6.0, "Command of Area": 6.0, "Communication": 5.0, "Eccentricity": 0.0,
        "Handling": 8.0, "Kicking": 5.0, "One on Ones": 4.0, "Punching (Tendency)": 0.0, "Reflexes": 8.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 3.0
    },
    "DL/DR": {
        "Corners": 1.0, "Crossing": 2.0, "Dribbling": 1.0, "Finishing": 1.0, "First Touch": 3.0, "Free Kick Taking": 1.0,
        "Heading": 2.0, "Long Shots": 1.0, "Long Throws": 1.0, "Marking": 3.0, "Passing": 2.0, "Penalty Taking": 1.0,
        "Tackling": 4.0, "Technique": 2.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 2.0, "Composure": 2.0,
        "Concentration": 4.0, "Decisions": 7.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 1.0,
        "Positioning": 4.0, "Teamwork": 2.0, "Vision": 2.0, "Work Rate": 2.0, "Acceleration": 7.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 2.0, "Natural Fitness": 0.0, "Pace": 5.0, "Stamina": 6.0, "Strength": 4.0,
        "Weaker Foot": 4.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "CB": {
        "Corners": 1.0, "Crossing": 1.0, "Dribbling": 1.0, "Finishing": 1.0, "First Touch": 2.0, "Free Kick Taking": 1.0,
        "Heading": 5.0, "Long Shots": 1.0, "Long Throws": 1.0, "Marking": 8.0, "Passing": 2.0, "Penalty Taking": 1.0,
        "Tackling": 5.0, "Technique": 1.0, "Aggression": 0.0, "Anticipation": 5.0, "Bravery": 2.0, "Composure": 2.0,
        "Concentration": 4.0, "Decisions": 10.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 2.0, "Off The Ball": 1.0,
        "Positioning": 8.0, "Teamwork": 1.0, "Vision": 1.0, "Work Rate": 2.0, "Acceleration": 6.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 6.0, "Natural Fitness": 0.0, "Pace": 5.0, "Stamina": 3.0, "Strength": 6.0,
        "Weaker Foot": 4.5, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "WBL/WBR": {
        "Corners": 1.0, "Crossing": 3.0, "Dribbling": 2.0, "Finishing": 1.0, "First Touch": 3.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 1.0, "Long Throws": 1.0, "Marking": 2.0, "Passing": 3.0, "Penalty Taking": 1.0,
        "Tackling": 3.0, "Technique": 3.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 1.0, "Composure": 2.0,
        "Concentration": 3.0, "Decisions": 5.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 2.0,
        "Positioning": 3.0, "Teamwork": 2.0, "Vision": 2.0, "Work Rate": 2.0, "Acceleration": 8.0, "Agility": 5.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 6.0, "Stamina": 7.0, "Strength": 4.0,
        "Weaker Foot": 4.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "DM": {
        "Corners": 1.0, "Crossing": 1.0, "Dribbling": 2.0, "Finishing": 2.0, "First Touch": 4.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 3.0, "Long Throws": 1.0, "Marking": 3.0, "Passing": 4.0, "Penalty Taking": 1.0,
        "Tackling": 7.0, "Technique": 3.0, "Aggression": 0.0, "Anticipation": 5.0, "Bravery": 1.0, "Composure": 2.0,
        "Concentration": 3.0, "Decisions": 8.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 1.0,
        "Positioning": 5.0, "Teamwork": 2.0, "Vision": 4.0, "Work Rate": 4.0, "Acceleration": 6.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 4.0, "Stamina": 4.0, "Strength": 5.0,
        "Weaker Foot": 5.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "ML/MR": {
        "Corners": 1.0, "Crossing": 5.0, "Dribbling": 3.0, "Finishing": 2.0, "First Touch": 4.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 2.0, "Long Throws": 1.0, "Marking": 1.0, "Passing": 3.0, "Penalty Taking": 1.0,
        "Tackling": 2.0, "Technique": 4.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 1.0, "Composure": 2.0,
        "Concentration": 2.0, "Decisions": 5.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 2.0,
        "Positioning": 1.0, "Teamwork": 2.0, "Vision": 3.0, "Work Rate": 3.0, "Acceleration": 8.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 6.0, "Stamina": 5.0, "Strength": 3.0,
        "Weaker Foot": 5.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "CM": {
        "Corners": 1.0, "Crossing": 1.0, "Dribbling": 2.0, "Finishing": 2.0, "First Touch": 6.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 3.0, "Long Throws": 1.0, "Marking": 3.0, "Passing": 6.0, "Penalty Taking": 1.0,
        "Tackling": 3.0, "Technique": 4.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 1.0, "Composure": 3.0,
        "Concentration": 2.0, "Decisions": 7.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 3.0,
        "Positioning": 3.0, "Teamwork": 2.0, "Vision": 6.0, "Work Rate": 3.0, "Acceleration": 6.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 5.0, "Stamina": 6.0, "Strength": 4.0,
        "Weaker Foot": 6.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "AML/AMR": {
        "Corners": 1.0, "Crossing": 5.0, "Dribbling": 5.0, "Finishing": 2.0, "First Touch": 5.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 2.0, "Long Throws": 1.0, "Marking": 1.0, "Passing": 2.0, "Penalty Taking": 1.0,
        "Tackling": 2.0, "Technique": 4.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 1.0, "Composure": 3.0,
        "Concentration": 2.0, "Decisions": 5.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 2.0,
        "Positioning": 1.0, "Teamwork": 2.0, "Vision": 3.0, "Work Rate": 3.0, "Acceleration": 10.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 10.0, "Stamina": 7.0, "Strength": 3.0,
        "Weaker Foot": 5.5, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "AMC": {
        "Corners": 1.0, "Crossing": 1.0, "Dribbling": 3.0, "Finishing": 3.0, "First Touch": 5.0, "Free Kick Taking": 1.0,
        "Heading": 1.0, "Long Shots": 3.0, "Long Throws": 1.0, "Marking": 1.0, "Passing": 4.0, "Penalty Taking": 1.0,
        "Tackling": 2.0, "Technique": 5.0, "Aggression": 0.0, "Anticipation": 3.0, "Bravery": 1.0, "Composure": 3.0,
        "Concentration": 2.0, "Decisions": 6.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 3.0,
        "Positioning": 2.0, "Teamwork": 2.0, "Vision": 6.0, "Work Rate": 3.0, "Acceleration": 9.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 1.0, "Natural Fitness": 0.0, "Pace": 7.0, "Stamina": 6.0, "Strength": 3.0,
        "Weaker Foot": 7.0, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    },
    "ST": {
        "Corners": 1.0, "Crossing": 2.0, "Dribbling": 5.0, "Finishing": 8.0, "First Touch": 6.0, "Free Kick Taking": 1.0,
        "Heading": 6.0, "Long Shots": 2.0, "Long Throws": 1.0, "Marking": 1.0, "Passing": 2.0, "Penalty Taking": 1.0,
        "Tackling": 1.0, "Technique": 4.0, "Aggression": 0.0, "Anticipation": 5.0, "Bravery": 1.0, "Composure": 6.0,
        "Concentration": 2.0, "Decisions": 5.0, "Determination": 0.0, "Flair": 0.0, "Leadership": 1.0, "Off The Ball": 6.0,
        "Positioning": 2.0, "Teamwork": 1.0, "Vision": 2.0, "Work Rate": 2.0, "Acceleration": 10.0, "Agility": 6.0,
        "Balance": 2.0, "Jumping Reach": 5.0, "Natural Fitness": 0.0, "Pace": 7.0, "Stamina": 6.0, "Strength": 6.0,
        "Weaker Foot": 7.5, "Aerial Reach": 0.0, "Command of Area": 0.0, "Communication": 0.0, "Eccentricity": 0.0,
        "Handling": 0.0, "Kicking": 0.0, "One on Ones": 0.0, "Punching (Tendency)": 0.0, "Reflexes": 0.0,
        "Rushing Out (Tendency)": 0.0, "Throwing": 0.0
    }
}

ABBR_MAP = {
    "Name": "Name", "Position": "Position", "Inf": "Inf", "Age": "Age", "Transfer Value": "Transfer Value",
    "Cor": "Corners", "Cro": "Crossing", "Dri": "Dribbling", "Fin": "Finishing", "Fir": "First Touch", "Fre": "Free Kick Taking",
    "Hea": "Heading", "Lon": "Long Shots", "L Th": "Long Throws", "LTh": "Long Throws", "Mar": "Marking", "Pas": "Passing", "Pen": "Penalty Taking",
    "Tck": "Tackling", "Tec": "Technique", "Agg": "Aggression", "Ant": "Anticipation", "Bra": "Bravery", "Cmp": "Composure", "Cnt": "Concentration",
    "Dec": "Decisions", "Det": "Determination", "Fla": "Flair", "Ldr": "Leadership", "OtB": "Off The Ball", "Pos": "Positioning", "Tea": "Teamwork", "Vis": "Vision", "Wor": "Work Rate",
    "Acc": "Acceleration", "Agi": "Agility", "Bal": "Balance", "Jum": "Jumping Reach", "Nat": "Natural Fitness", "Pac": "Pace", "Sta": "Stamina", "Str": "Strength",
    "Weaker Foot": "Weaker Foot", "Aer": "Aerial Reach", "Cmd": "Command of Area", "Com": "Communication", "Ecc": "Eccentricity", "Han": "Handling", "Kic": "Kicking",
    "1v1": "One on Ones", "Pun": "Punching (Tendency)", "Ref": "Reflexes", "TRO": "Rushing Out (Tendency)", "Thr": "Throwing"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def parse_players_from_html(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, "No <table> found in HTML."

    header_row = table.find("tr")
    if header_row is None:
        return None, "No rows in table."

    ths = header_row.find_all(["th", "td"])
    header_cells = [th.get_text(strip=True) for th in ths]
    canonical = [ABBR_MAP.get(h, h) for h in header_cells]

    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if not cols or all(not c for c in cols):
            continue

        if len(cols) < len(canonical):
            cols += [""] * (len(canonical) - len(cols))
        cols = cols[:len(canonical)]

        row = {col_name: val for col_name, val in zip(canonical, cols) if col_name}

        name = (row.get("Name") or "").strip()
        if not name or name.lower() == "name":
            continue

        rows.append(row)

    if not rows:
        return None, "No data rows parsed from HTML table."

    df = pd.DataFrame(rows)

    # convert numeric-like columns except textual ones
    for c in df.columns:
        if c in ("Name", "Position", "Transfer Value", "Inf"):
            continue
        df[c] = df[c].astype(str).str.extract(r'(-?\d+(?:\.\d+)?)')[0]
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")

    return df, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if not any(cols.count(c) > 1 for c in cols):
        return df

    unique_order = []
    for c in cols:
        if c not in unique_order:
            unique_order.append(c)

    merged = pd.DataFrame(index=df.index)
    for col in unique_order:
        same_cols = [c for c in cols if c == col]
        if len(same_cols) == 1:
            merged[col] = df[col]
        else:
            subset = df.loc[:, same_cols]
            subset_num = subset.apply(pd.to_numeric, errors="coerce")
            if subset_num.notna().sum().sum() > 0:
                merged[col] = subset_num.mean(axis=1)
            else:
                merged[col] = subset.apply(lambda r: next((v for v in r if isinstance(v, str) and v.strip()), ""), axis=1)

    return merged

@st.cache_data(ttl=3600)  # Cache for 1 hour
def parse_transfer_value(x):
    """Parse transfer value strings into numeric values"""
    try:
        if pd.isna(x):
            return 0.0
        s = str(x).strip()
        if not s or s == "-" or s.lower() in {"n/a", "none"}:
            return 0.0

        s2 = re.sub(r'[^0-9\.,kKmM]', '', s)
        if s2 == "":
            m = re.search(r'(-?\d+(?:\.\d+)?)', s)
            if m:
                try:
                    return float(m.group(1).replace(',', ''))
                except Exception:
                    return 0.0
            return 0.0

        m = re.match(r'([0-9\.,]+)\s*([kKmM]?)', s2)
        if not m:
            try:
                return float(s2.replace(',', ''))
            except Exception:
                return 0.0

        num = m.group(1).replace(',', '')
        try:
            val = float(num)
        except Exception:
            val = 0.0

        suf = m.group(2).lower()
        if suf == 'k':
            val *= 1_000.0
        elif suf == 'm':
            val *= 1_000_000.0

        return val
    except Exception:
        return 0.0

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_name_key(name):
    """Create name key for deduplication"""
    if pd.isna(name) or not name:
        return f"_empty_{id(name)}"

    name_str = str(name).strip()
    if not name_str:
        return f"_empty_{id(name)}"

    normalized = re.sub(r'\s+', ' ', name_str.lower().strip())
    normalized = normalized.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    normalized = normalized.replace('ñ', 'n').replace('ç', 'c')

    return normalized

@st.cache_data(ttl=3600)  # Cache for 1 hour
def deduplicate_players(df):
    """Deduplicate players by name, keeping best version"""
    if len(df) <= 1:
        return df

    df = df.copy()
    
    # Check if Name column exists
    if 'Name' not in df.columns:
        st.warning("⚠️ No 'Name' column found, skipping deduplication.")
        return df
    
    df['_name_key'] = df['Name'].apply(create_name_key)
    df['_transfer_val_numeric'] = df.get('Transfer Value', '').apply(parse_transfer_value)

    # Calculate average score across all roles for deduplication
    available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]
    if available_attrs:
        attrs_df = df[available_attrs].fillna(0).astype(float)
        avg_scores = []
        for _, player in attrs_df.iterrows():
            player_scores = []
            for role, weights in WEIGHTS_BY_ROLE.items():
                w = pd.Series({a: float(weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)
                score = player.values.dot(w.values.astype(float))
                player_scores.append(score)
            avg_scores.append(np.mean(player_scores))
        df['_avg_score'] = avg_scores
    else:
        df['_avg_score'] = 0

    df_sorted = df.sort_values(['_avg_score', '_transfer_val_numeric'], ascending=[False, False])
    df_deduped = df_sorted.drop_duplicates(subset=['_name_key'], keep='first')
    df_deduped = df_deduped.drop(columns=['_name_key', '_transfer_val_numeric', '_avg_score'])

    duplicates_removed = len(df) - len(df_deduped)
    if duplicates_removed > 0:
        st.success(f"✅ Removed {duplicates_removed} duplicate player(s)")

    return df_deduped.reset_index(drop=True)

def create_file_hash(uploaded_files):
    """Create a hash of uploaded files to detect changes"""
    file_contents = []
    for file in uploaded_files:
        file.seek(0)  # Reset file pointer
        file_contents.append(file.read())
        file.seek(0)  # Reset for later use
    combined_content = b''.join(file_contents)
    return hashlib.md5(combined_content).hexdigest()

def should_refresh_cache(current_hash, last_hash):
    """Determine if cache should be refreshed based on file changes"""
    return current_hash != last_hash

# Sidebar Configuration
with st.sidebar:
    # User Preferences
    st.markdown("### User Preferences")
    
    # Default view preference
    default_view = st.selectbox(
        "Default View",
        ["Full Table", "Automatic Teambuilder", "Custom Teambuilder"],
        index=["Full Table", "Automatic Teambuilder", "Custom Teambuilder"].index(st.session_state.user_preferences['default_view']),
        help="Choose your preferred default tab"
    )
    if default_view != st.session_state.user_preferences['default_view']:
        st.session_state.user_preferences['default_view'] = default_view
        save_preferences()
    
    # Advanced stats toggle
    show_advanced = st.checkbox(
        "Show Advanced Stats",
        value=st.session_state.user_preferences['show_advanced_stats'],
        help="Display additional player statistics and metrics"
    )
    if show_advanced != st.session_state.user_preferences['show_advanced_stats']:
        st.session_state.user_preferences['show_advanced_stats'] = show_advanced
        save_preferences()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "Auto-refresh on File Change",
        value=st.session_state.user_preferences['auto_refresh'],
        help="Automatically refresh when new files are uploaded"
    )
    if auto_refresh != st.session_state.user_preferences['auto_refresh']:
        st.session_state.user_preferences['auto_refresh'] = auto_refresh
        save_preferences()
    
    # Cache management
    st.markdown("### Cache Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Cache", help="Clear all cached data to refresh calculations"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Cache Stats", help="Show cache statistics"):
            cache_size = len(st.cache_data._cache) if hasattr(st.cache_data, '_cache') else 0
            st.info(f"Cache entries: {cache_size}")
    
    # Performance monitoring
    st.markdown("### Performance")
    if st.session_state.last_upload_time:
        upload_time = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_upload_time))
        st.caption(f"Last upload: {upload_time}")


# Create tabs for different views with user preference
tab_names = ["📊 Full Table", "🤖 Automatic Teambuilder", "⚽ Custom Teambuilder"]
default_tab_index = tab_names.index(f"📊 {st.session_state.user_preferences['default_view']}") if f"📊 {st.session_state.user_preferences['default_view']}" in tab_names else 0

tab1, tab2, tab3 = st.tabs(tab_names)

# Dynamic upload instructions based on status
def get_upload_instructions_status():
    """Determine upload status and return appropriate styling"""
    if 'upload_status' not in st.session_state:
        st.session_state.upload_status = 'waiting'  # waiting, success, partial, failed
    
    return st.session_state.upload_status

# Upload instructions with dynamic styling
upload_status = get_upload_instructions_status()

if upload_status == 'waiting':
    # Show normal blue instructions
    st.markdown("""
    <div class="info-box">
        <strong>Upload Instructions:</strong><br>
        • Go to <a href="https://fmarenacalc.com" target="_blank">fmarenacalc.com</a> for HTML export guide<br>
        • Maximum 260 players can be exported per html file, so make sure to narrow your search before each print screen<br>
        • Multiple files can be uploaded simultaneously<br>
        • Supports .html only and not .rtf like the NewGAN mod
    </div>
    """, unsafe_allow_html=True)
elif upload_status == 'failed':
    # Show red pulsing instructions for failure
    st.markdown("""
    <style>
    @keyframes pulse-red {
        0% { background-color: #ff4444; }
        50% { background-color: #ff6666; }
        100% { background-color: #ff4444; }
    }
    .failed-upload {
        background: #ff4444;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #cc0000;
        margin-bottom: 1rem;
        color: white;
        animation: pulse-red 2s infinite;
    }
    </style>
    <div class="failed-upload">
        <strong>⚠️ Upload Failed - Please Follow Instructions:</strong><br>
        • Go to <a href="https://fmarenacalc.com" target="_blank" style="color: #ffffcc;">fmarenacalc.com</a> for HTML export guide<br>
        • Maximum 260 players can be exported per html file, so make sure to narrow your search before each print screen<br>
        • Multiple files can be uploaded simultaneously<br>
        • Supports .html only and not .rtf like the NewGAN mod
    </div>
    """, unsafe_allow_html=True)
elif upload_status == 'partial':
    # Show yellow warning for partial success
    st.markdown("""
    <div class="info-box" style="background: #fff3cd; border-left: 4px solid #ffc107; color: #856404;">
        <strong>⚠️ Partial Upload Success:</strong><br>
        Some files uploaded successfully, but some failed. Check the results below.
    </div>
    """, unsafe_allow_html=True)
# Success case: no instructions shown (they disappear)

uploaded_files = st.file_uploader(
    "Follow the 'Upload Instructions:' above and upload the HTML files you print screened from FM24 here below",
    type=["html", "htm"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# Check for file changes and optimize processing
current_file_hash = create_file_hash(uploaded_files)
file_changed = should_refresh_cache(current_file_hash, st.session_state.file_hash)

if file_changed:
    st.session_state.file_hash = current_file_hash
    st.session_state.last_upload_time = time.time()
    # Clear cache if files changed and auto-refresh is enabled
    if st.session_state.user_preferences['auto_refresh']:
        st.cache_data.clear()
        st.success("🔄 Files changed! Cache cleared and refreshing...")

# Process files and show summary
dfs = []
file_results = []
successful_files = 0
failed_files = 0

# Create progress bar
progress_bar = st.progress(0)
status_text = st.empty()

for i, uploaded in enumerate(uploaded_files):
    # Update progress
    progress_bar.progress((i + 1) / len(uploaded_files))
    status_text.text(f'Processing {uploaded.name}...')
    
    raw = uploaded.read()
    try:
        html_text = raw.decode('utf-8', errors='ignore')
    except Exception:
        html_text = raw.decode('latin-1', errors='ignore')

    df, err = parse_players_from_html(html_text)
    if df is None:
        file_results.append(f"❌ {uploaded.name}: Failed to read")
        failed_files += 1
        continue

    df = merge_duplicate_columns(df)
    df = df.reset_index(drop=True)
    dfs.append(df)
    file_results.append(f"✅ {uploaded.name}: {len(df)} players loaded")
    successful_files += 1

# Complete the progress bar
progress_bar.progress(1.0)
status_text.text('Processing complete!')

# Determine upload status
if successful_files == 0:
    st.session_state.upload_status = 'failed'
    st.error("❌ No valid player data parsed from any uploaded file.")
    st.stop()
elif failed_files > 0:
    st.session_state.upload_status = 'partial'
else:
    st.session_state.upload_status = 'success'

# Show results
for result in file_results:
    st.write(result)

# Combine all data
df = pd.concat(dfs, ignore_index=True)
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]

if not available_attrs:
    st.error("❌ No matching attribute columns found. Detected columns: " + ", ".join(list(df.columns)))
    st.stop()

# Deduplicate players first
df_final = deduplicate_players(df)

# Check if we have any players left after deduplication
if len(df_final) == 0:
    st.error("❌ No players remaining after deduplication.")
    st.stop()

# Check if Name column exists
if 'Name' not in df_final.columns:
    st.error("❌ Name column not found in player data.")
    st.stop()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_role_scores(df_final, available_attrs):
    """Calculate role scores for all players"""
    attrs_df_final = df_final[available_attrs].fillna(0).astype(float)
    attrs_norm_final = attrs_df_final
    
    role_scores = {}
    for role, weights in WEIGHTS_BY_ROLE.items():
        w = pd.Series({a: float(weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)
        scores = attrs_norm_final.values.dot(w.values.astype(float))
        role_scores[role] = scores
    
    return role_scores, attrs_norm_final

# Calculate scores for all roles
role_scores, attrs_norm_final = calculate_role_scores(df_final, available_attrs)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def create_comprehensive_table(df_final, role_scores):
    """Create the comprehensive player rankings table"""
    comprehensive_data = {
        'Rank': range(1, len(df_final) + 1),
        'Name': df_final['Name'],
        'Age': df_final.get('Age', pd.Series(['N/A'] * len(df_final)))
    }
    
    # Add scores for each role
    for role in ['GK', 'DL/DR', 'CB', 'WBL/WBR', 'DM', 'ML/MR', 'CM', 'AML/AMR', 'AMC', 'ST']:
        comprehensive_data[role] = role_scores[role].round(0).astype(int)
    
    return pd.DataFrame(comprehensive_data)

# Create the new comprehensive table
comprehensive_df = create_comprehensive_table(df_final, role_scores)

# Now create the tabs for additional features
with tab1:
    st.markdown("## Player Rankings by Position")
    
    # Advanced stats section
    if st.session_state.user_preferences['show_advanced_stats']:
        st.markdown("### Advanced Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_players = len(comprehensive_df)
            st.metric("Total Players", total_players)
        
        with col2:
            avg_age = comprehensive_df['Age'].replace('N/A', np.nan).astype(float).mean()
            st.metric("Average Age", f"{avg_age:.1f}" if not pd.isna(avg_age) else "N/A")
        
        with col3:
            # Find best overall player (highest average score across all positions)
            role_columns = ['GK', 'DL/DR', 'CB', 'WBL/WBR', 'DM', 'ML/MR', 'CM', 'AML/AMR', 'AMC', 'ST']
            comprehensive_df['Overall_Avg'] = comprehensive_df[role_columns].mean(axis=1)
            best_player = comprehensive_df.loc[comprehensive_df['Overall_Avg'].idxmax(), 'Name']
            st.metric("Best Overall", best_player)
        
        with col4:
            # Most versatile player (lowest standard deviation across positions)
            comprehensive_df['Versatility'] = comprehensive_df[role_columns].std(axis=1)
            most_versatile = comprehensive_df.loc[comprehensive_df['Versatility'].idxmin(), 'Name']
            st.metric("Most Versatile", most_versatile)
        
        st.markdown("---")
    
    # Create a styled dataframe with colors that's sortable
    def get_score_color(val, role):
        """Get color for score based on role thresholds"""
        if pd.isna(val) or val == 0:
            return '#666666'
        
        if role == "GK":
            if val >= 1600: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1550: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1400: return '#ffffff'             # WHITE
            elif val >= 1300: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 1200: return 'rgb(255, 150, 0)'    # VIBRANT ORANGE
            elif val >= 1100: return 'rgb(255, 0, 0)'      # VIBRANT RED
            else: return ''                                # BLACK (1000 and below)
        elif role == "DL/DR":
            if val >= 1300: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1250: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1100: return '#ffffff'             # WHITE
            elif val >= 1000: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 900: return 'rgb(255, 150, 0)'     # VIBRANT ORANGE
            elif val >= 800: return 'rgb(255, 0, 0)'       # VIBRANT RED
            else: return ''                                # BLACK (700 and below)
        elif role == "CB":
            if val >= 1500: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1450: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1300: return '#ffffff'             # WHITE
            elif val >= 1200: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 1100: return 'rgb(255, 150, 0)'    # VIBRANT ORANGE
            elif val >= 1000: return 'rgb(255, 0, 0)'      # VIBRANT RED
            else: return ''                                # BLACK (900 and below)
        elif role == "DM":
            if val >= 1400: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1350: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1200: return '#ffffff'             # WHITE
            elif val >= 1100: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 1000: return 'rgb(255, 150, 0)'    # VIBRANT ORANGE
            elif val >= 900: return 'rgb(255, 0, 0)'       # VIBRANT RED
            else: return ''                                # BLACK (800 and below)
        elif role in ["AML/AMR", "AMC"]:
            if val >= 1500: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1450: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1300: return '#ffffff'             # WHITE
            elif val >= 1200: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 1100: return 'rgb(255, 150, 0)'    # VIBRANT ORANGE
            elif val >= 1000: return 'rgb(255, 0, 0)'      # VIBRANT RED
            else: return ''                                # BLACK (900 and below)
        elif role == "ST":
            if val >= 1700: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1650: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1450: return '#ffffff'             # WHITE
            elif val >= 1300: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 1200: return 'rgb(255, 150, 0)'    # VIBRANT ORANGE
            elif val >= 1100: return 'rgb(255, 0, 0)'      # VIBRANT RED
            else: return ''                                # BLACK (1000 and below)
        else:  # WBL/WBR, ML/MR, CM (use DL/DR thresholds)
            if val >= 1300: return 'rgb(0, 255, 255)'      # BLUE
            elif val >= 1250: return 'rgb(0, 255, 0)'      # VIBRANT GREEN
            elif val >= 1100: return '#ffffff'             # WHITE
            elif val >= 1000: return 'rgb(255, 255, 0)'    # VIBRANT YELLOW
            elif val >= 900: return 'rgb(255, 150, 0)'     # VIBRANT ORANGE
            elif val >= 800: return 'rgb(255, 0, 0)'       # VIBRANT RED
            else: return ''                                # BLACK (700 and below)
    
    # Create styled dataframe with colors that's sortable
    # Only apply colors to positions with specified thresholds
    role_columns = ['GK', 'DL/DR', 'CB', 'DM', 'AML/AMR', 'AMC', 'ST']
    
    # Positions that should show empty cells for low scores
    empty_cell_columns = ['WBL/WBR', 'ML/MR', 'CM']
    
    # Create a styled dataframe using pandas styling
    def style_scores(val, role):
        if role in role_columns and pd.notna(val) and val != 0:
            color = get_score_color(val, role)
            if color == '':  # Empty string means don't display the value
                return 'color: transparent; font-weight: bold;'
            else:
                return f'color: {color}; font-weight: bold;'
        elif role in empty_cell_columns and pd.notna(val) and val != 0:
            # Check thresholds for empty cells
            if role in ['WBL/WBR', 'ML/MR'] and val < 700:
                return 'color: transparent; font-weight: bold;'  # Empty for WBL/WBR, ML/MR below 700
            elif role == 'CM' and val < 800:
                return 'color: transparent; font-weight: bold;'  # Empty for CM below 800
        return ''
    
    # Create a copy of the dataframe and modify values for display
    display_df = comprehensive_df.copy()
    
    # Apply empty cell logic to WBL/WBR, ML/MR, CM
    for col in empty_cell_columns:
        if col in display_df.columns:
            if col in ['WBL/WBR', 'ML/MR']:
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 700 else x)
            elif col == 'CM':
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 800 else x)
    
    # Apply empty cell logic to colored columns (very poor scores - BLACK zone)
    for col in role_columns:
        if col in display_df.columns:
            if col == "GK":
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 1000 else x)  # 1000 and below = BLACK
            elif col == "DL/DR":
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 700 else x)   # 700 and below = BLACK
            elif col == "CB":
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 900 else x)   # 900 and below = BLACK
            elif col == "DM":
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 800 else x)   # 800 and below = BLACK
            elif col in ["AML/AMR", "AMC"]:
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 900 else x)   # 900 and below = BLACK
            elif col == "ST":
                display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and x < 1000 else x)  # 1000 and below = BLACK
    
    # First, handle empty cells for black zone scores and non-colored positions
    display_df = comprehensive_df.copy()
    
    # Handle WBL/WBR and ML/MR (empty if below 700)
    for col in ['WBL/WBR', 'ML/MR']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and float(x) < 700 else x)
    
    # Handle CM (empty if below 800)
    if 'CM' in display_df.columns:
        display_df['CM'] = display_df['CM'].apply(lambda x: '' if pd.notna(x) and float(x) < 800 else x)
    
    # Handle colored positions with black zones (everything under red threshold)
    black_zone_thresholds = {
        'GK': 1100,      # Hide < 1100 (under red)
        'DL/DR': 800,    # Hide < 800 (under red)
        'CB': 1000,      # Hide < 1000 (under red)
        'DM': 900,       # Hide < 900 (under red)
        'AML/AMR': 1000, # Hide < 1000 (under red)
        'AMC': 1000,     # Hide < 1000 (under red)
        'ST': 1100       # Hide < 1100 (under red)
    }
    
    for col, threshold in black_zone_thresholds.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: '' if pd.notna(x) and float(x) < threshold else x)
    
    # Create the style function for colors
    def style_df(df):
        def color_score(val, col):
            try:
                if pd.isna(val) or val == '' or val == 0:
                    return ''
                
                val_float = float(val)
                
                # Color coding for each position
                if col == 'GK':
                    if val_float >= 1600: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1550: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1400: return 'color: white'               # WHITE
                    elif val_float >= 1300: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 1200: return 'color: rgb(255, 150, 0)'    # VIBRANT ORANGE
                    elif val_float >= 1100: return 'color: rgb(255, 0, 0)'      # VIBRANT RED
                    else: return ''                                             # Hide < 1100 (under red)
                elif col == 'DL/DR':
                    if val_float >= 1300: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1250: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1100: return 'color: white'               # WHITE
                    elif val_float >= 1000: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 900: return 'color: rgb(255, 150, 0)'     # VIBRANT ORANGE
                    elif val_float >= 800: return 'color: rgb(255, 0, 0)'       # VIBRANT RED
                    else: return ''                                             # Hide < 800 (under red)
                elif col == 'CB':
                    if val_float >= 1500: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1450: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1300: return 'color: white'               # WHITE
                    elif val_float >= 1200: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 1100: return 'color: rgb(255, 150, 0)'    # VIBRANT ORANGE
                    elif val_float >= 1000: return 'color: rgb(255, 0, 0)'      # VIBRANT RED
                    else: return ''                                             # Hide < 1000 (under red)
                elif col == 'DM':
                    if val_float >= 1400: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1350: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1200: return 'color: white'               # WHITE
                    elif val_float >= 1100: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 1000: return 'color: rgb(255, 150, 0)'    # VIBRANT ORANGE
                    elif val_float >= 900: return 'color: rgb(255, 0, 0)'       # VIBRANT RED
                    else: return ''                                             # Hide < 900 (under red)
                elif col in ['AML/AMR', 'AMC']:
                    if val_float >= 1500: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1450: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1300: return 'color: white'               # WHITE
                    elif val_float >= 1200: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 1100: return 'color: rgb(255, 150, 0)'    # VIBRANT ORANGE
                    elif val_float >= 1000: return 'color: rgb(255, 0, 0)'      # VIBRANT RED
                    else: return ''                                             # Hide < 1000 (under red)
                elif col == 'ST':
                    if val_float >= 1700: return 'color: rgb(0, 255, 255)'      # BLUE
                    elif val_float >= 1650: return 'color: rgb(0, 255, 0)'      # VIBRANT GREEN
                    elif val_float >= 1450: return 'color: white'               # WHITE
                    elif val_float >= 1300: return 'color: rgb(255, 255, 0)'    # VIBRANT YELLOW
                    elif val_float >= 1200: return 'color: rgb(255, 150, 0)'    # VIBRANT ORANGE
                    elif val_float >= 1100: return 'color: rgb(255, 0, 0)'      # VIBRANT RED
                    else: return ''                                             # Hide < 1100 (under red)
            except (ValueError, TypeError):
                return ''
            return ''
        
        # Create a style DataFrame with the same shape as our data
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        
        # Apply colors only to specified columns
        for col in df.columns:
            if col in black_zone_thresholds:
                styles[col] = df[col].apply(lambda x: color_score(x, col))
        
        return styles
    
    # Convert numeric columns to float for proper sorting
    numeric_columns = ['GK', 'DL/DR', 'CB', 'WBL/WBR', 'DM', 'ML/MR', 'CM', 'AML/AMR', 'AMC', 'ST']
    for col in numeric_columns:
        if col in display_df.columns:
            # Convert to float but keep empty strings as is
            display_df[col] = display_df[col].apply(lambda x: float(x) if pd.notna(x) and x != '' else x)
    
    # Apply styling and display the dataframe
    styled_df = display_df.style.apply(lambda _: style_df(display_df), axis=None)
    
    # Display with Streamlit's native dataframe for sorting
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

with tab2:
    st.markdown("## Automatic Teambuilder")
    st.markdown("""
    <div class="info-box">
        <strong>Formation Analysis:</strong><br>
        Hungarian algorithm used to create the best starting 11, it also creates a secondary team with 0 overlap in players from the first team. Some ridiculous options occur like a DM being recommended as a ST but it should theoretically be true as long as their hidden attributes aren't terrible.
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.use_custom_teams = False

with tab3:
    st.markdown("## Custom Teambuilder")
    st.markdown("""
    <div class="info-box">
        <strong>Custom Team Builder:</strong><br>
        Build your teams manually by selecting players for each position. Use the dropdowns to assign players to positions.
    </div>
    """, unsafe_allow_html=True)
    
    st.session_state.use_custom_teams = True
    
    # Formation positions
    formation_positions = [
        ("GK", "GK"),
        ("RB", "DL/DR"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "DL/DR"),
        ("DM1", "DM"), ("DM2", "DM"),
        ("AMR", "AML/AMR"), ("AMC", "AMC"), ("AML", "AML/AMR"),
        ("ST", "ST")
    ]
    
    # Create two columns for the two teams
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### First XI")
        first_xi_selections = {}
        
        for pos_label, role in formation_positions:
            # Get available players for this role (top 20 by score)
            top_players = comprehensive_df.nlargest(20, role)[['Name', role]]
            
            # Create dropdown for player selection
            player_options = ["Select Player"] + [f"{name} ({int(score)})" for name, score in zip(top_players['Name'], top_players[role])]
            selected = st.selectbox(f"{pos_label} ({role})", player_options, key=f"first_{pos_label}")
            
            if selected != "Select Player":
                player_name = selected.split(" (")[0]
                first_xi_selections[pos_label] = player_name
        
        st.session_state.custom_first_xi = first_xi_selections
    
    with col2:
        st.markdown("### Second XI")
        second_xi_selections = {}
        
        for pos_label, role in formation_positions:
            # Get available players for this role (top 20 by score)
            top_players = comprehensive_df.nlargest(20, role)[['Name', role]]
            
            # Create dropdown for player selection
            player_options = ["Select Player"] + [f"{name} ({int(score)})" for name, score in zip(top_players['Name'], top_players[role])]
            selected = st.selectbox(f"{pos_label} ({role})", player_options, key=f"second_{pos_label}")
            
            if selected != "Select Player":
                player_name = selected.split(" (")[0]
                second_xi_selections[pos_label] = player_name
        
        st.session_state.custom_second_xi = second_xi_selections
    
    # Show team summaries
    if st.session_state.custom_first_xi:
        st.markdown("#### First XI Summary")
        first_xi_df = pd.DataFrame(list(st.session_state.custom_first_xi.items()), columns=['Position', 'Player'])
        st.dataframe(first_xi_df, use_container_width=True)
    
    if st.session_state.custom_second_xi:
        st.markdown("#### Second XI Summary")
        second_xi_df = pd.DataFrame(list(st.session_state.custom_second_xi.items()), columns=['Position', 'Player'])
        st.dataframe(second_xi_df, use_container_width=True)

# Fixed Formation Setup
formation_lines = [
    ("GK", "GK"),
    ("EMPTY", "EMPTY"),
    ("RB", "DL/DR"),
    ("CB", "CB"),
    ("CB", "CB"),
    ("LB", "DL/DR"),
    ("EMPTY", "EMPTY"),
    ("DM", "DM"),
    ("DM", "DM"),
    ("EMPTY", "EMPTY"),
    ("AMR", "AML/AMR"),
    ("AMC", "AMC"),
    ("AML", "AML/AMR"),
    ("EMPTY", "EMPTY"),
    ("ST", "ST")
]

# Filter out EMPTY positions for the actual team selection
positions = [(label, role) for label, role in formation_lines if role != "EMPTY"]

n_players = len(df_final)
n_positions = len(positions)
player_names = df_final["Name"].astype(str).tolist()

# Check if we have enough players for the formation
if n_players < n_positions:
    st.warning(f"⚠️ Only {n_players} players available, but formation requires {n_positions} positions. Some positions may be empty.")

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def compute_score_matrix(df_final, available_attrs, positions):
    """Compute the score matrix for team building"""
    n_players = len(df_final)
    n_positions = len(positions)
    
    # Precompute role weight vectors
    role_weight_vectors = {}
    for _, role_key in positions:
        if role_key not in role_weight_vectors: # Avoid re-computing for same role
            rw = WEIGHTS_BY_ROLE.get(role_key, {})
            role_weight_vectors[role_key] = np.array([float(rw.get(a, 0.0)) for a in available_attrs], dtype=float)

    # Compute score matrix
    score_matrix = np.zeros((n_players, n_positions), dtype=float)
    attrs_norm_final = df_final[available_attrs].fillna(0).astype(float)
    
    for i_idx in range(n_players):
        player_attr_vals = attrs_norm_final.iloc[i_idx].values if len(available_attrs) > 0 else np.zeros((len(available_attrs),), dtype=float)
        for p_idx, (_, role_key) in enumerate(positions):
            w = role_weight_vectors[role_key]
            score_matrix[i_idx, p_idx] = float(np.dot(player_attr_vals, w))
    
    return score_matrix

# Compute score matrix
score_matrix = compute_score_matrix(df_final, available_attrs, positions)

# Hungarian algorithm assignment
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def choose_starting_xi(available_player_indices, current_score_matrix):
    avail = list(available_player_indices)
    num_avail = len(avail)
    num_pos = current_score_matrix.shape[1]

    if num_avail == 0 or num_pos == 0:
        return {}

    # Cost matrix for assignment
    cost_matrix = -current_score_matrix[avail, :]

    if linear_sum_assignment is None or num_avail < num_pos:
        # Greedy fallback if scipy is missing or not enough players
        chosen = {}
        used_players = set()
        for p_idx in range(num_pos):
            best_player_idx = -1
            best_score = -1e9
            for i_idx in avail:
                if i_idx not in used_players:
                    score = current_score_matrix[i_idx, p_idx]
                    if score > best_score:
                        best_score = score
                        best_player_idx = i_idx
            if best_player_idx != -1:
                chosen[p_idx] = best_player_idx
                used_players.add(best_player_idx)
        return chosen
    else:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        chosen = {c: avail[r] for r, c in zip(row_ind, col_ind)}
        return chosen

def render_xi(chosen_map, team_name="Team"):
    rows = []
    position_index = 0
    
    # Fixed formation lines
    formation_lines = [
        ("GK", "GK"),
        ("EMPTY", "EMPTY"),
        ("RB", "DL/DR"),
        ("CB", "CB"),
        ("CB", "CB"),
        ("LB", "DL/DR"),
        ("EMPTY", "EMPTY"),
        ("DM", "DM"),
        ("DM", "DM"),
        ("EMPTY", "EMPTY"),
        ("AMR", "AML/AMR"),
        ("AMC", "AMC"),
        ("AML", "AML/AMR"),
        ("EMPTY", "EMPTY"),
        ("ST", "ST")
    ]
    
    # Build rows including empty spaces for visual formatting
    for line_label, line_role in formation_lines:
        if line_role == "EMPTY":
            rows.append(("EMPTY", "---", 0.0, "EMPTY"))
        else:
            if position_index in chosen_map:
                p_idx = chosen_map[position_index]
                name = player_names[p_idx]
                sel_score = score_matrix[p_idx, position_index]
                rows.append((line_label, name, sel_score, line_role))
            else:
                rows.append((line_label, "---", 0.0, line_role))
            position_index += 1

    team_total = sum(r[2] for r in rows if r[1] != "---" and r[0] != "EMPTY")
    placed_scores = [r[2] for r in rows if r[1] != "---" and r[0] != "EMPTY"]
    team_avg = np.mean(placed_scores) if placed_scores else 0.0
    # --- Role-specific color thresholds + interpolation helpers ---
    def _lerp_color(c1, c2, t):
        """Linearly interpolate two RGB tuples (c1->c2) by t in [0..1]."""
        return (
            int(round(c1[0] + (c2[0] - c1[0]) * t)),
            int(round(c1[1] + (c2[1] - c1[1]) * t)),
            int(round(c1[2] + (c2[2] - c1[2]) * t))
        )

    def _rgb_to_css(rgb):
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    # Colour stops (RGB)
    BLUE    = (0, 255, 255)
    VGREEN  = (0, 255, 0)
    WHITE   = (255, 255, 255)
    VYELLOW = (255, 255, 0)
    VORANGE = (255, 165, 0)
    VRED    = (255, 0, 0)
    BLACK   = (0, 0, 0)

    ROLE_THRESHOLDS = {
        "GK":      [(1600, BLUE), (1550, VGREEN), (1400, WHITE), (1300, VYELLOW), (1200, VORANGE), (1100, VRED), (1000, BLACK)],
        "DL/DR":   [(1300, BLUE), (1250, VGREEN), (1100, WHITE), (1000, VYELLOW), (900, VORANGE), (800, VRED), (700, BLACK)],
        "CB":      [(1500, BLUE), (1450, VGREEN), (1300, WHITE), (1200, VYELLOW), (1100, VORANGE), (1000, VRED), (900, BLACK)],
        "DM":      [(1400, BLUE), (1350, VGREEN), (1200, WHITE), (1100, VYELLOW), (1000, VORANGE), (900, VRED), (800, BLACK)],
        "AML/AMR": [(1500, BLUE), (1450, VGREEN), (1300, WHITE), (1200, VYELLOW), (1100, VORANGE), (1000, VRED), (900, BLACK)],
        "AMC":     [(1500, BLUE), (1450, VGREEN), (1300, WHITE), (1200, VYELLOW), (1100, VORANGE), (1000, VRED), (900, BLACK)],
        "ST":      [(1700, BLUE), (1650, VGREEN), (1450, WHITE), (1300, VYELLOW), (1200, VORANGE), (1100, VRED), (1000, BLACK)]
    }

    def get_role_color(role_key, score):
        """Return CSS 'rgb(...)' color string for the given role and numeric score.
           Interpolates smoothly between adjacent colour stops when score falls between thresholds.
        """
        rk = role_key

        # Normalize some common role labels to the names used in thresholds
        if rk in {"RB", "LB"}:
            rk = "DL/DR"
        if rk in {"AMR", "AML"}:
            rk = "AML/AMR"

        thresholds = ROLE_THRESHOLDS.get(rk)
        if not thresholds:
            return _rgb_to_css(WHITE)

        # Ensure thresholds are sorted descending by value
        thresholds_sorted = sorted(thresholds, key=lambda x: x[0], reverse=True)

        top_val, top_col = thresholds_sorted[0]
        bot_val, bot_col = thresholds_sorted[-1]

        if score >= top_val:
            return _rgb_to_css(top_col)
        if score <= bot_val:
            return _rgb_to_css(bot_col)

        # find interval where v_low <= score <= v_high and interpolate
        for i in range(len(thresholds_sorted) - 1):
            v_high, c_high = thresholds_sorted[i]
            v_low,  c_low  = thresholds_sorted[i + 1]
            if v_low <= score <= v_high:
                if v_high == v_low:
                    t = 0.0
                else:
                    t = (score - v_low) / (v_high - v_low)
                rgb = _lerp_color(c_low, c_high, t)
                return _rgb_to_css(rgb)

        return _rgb_to_css(WHITE)
    # --- end role-specific color helpers ---

# Format as table
    lines = [f"<div class='xi-formation'>"]
    lines.append(f"<h3 style='text-align: center; margin-bottom: 1rem;'>{team_name}</h3>")

    for pos_label, name, sel_score, role_key in rows:
        if role_key == "EMPTY":
            lines.append("<div style='height: 20px;'></div>")  # Empty space
        else:
            sel_score_int = int(round(sel_score))
            
            # Use role-specific threshold coloring (with interpolation)
            name_color = get_role_color(role_key, float(sel_score))

            lines.append(f"""<div style='display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.1); border-radius: 5px;'>
                <span style='font-weight: bold; min-width: 5rem; color: {name_color};'>{pos_label}</span>
                <span style='font-weight: bold; flex-grow: 1; text-align: center; color: {name_color};'>{name}</span>
                <span style='min-width: 4rem; text-align: right; color: {name_color};'>{sel_score_int} pts</span>
            </div>""")

    lines.append(f"""<div style='margin-top: 2rem; padding-top: 1rem; border-top: 2px solid rgba(255,255,255,0.3); text-align: center;'>
        <strong>Team Total: {int(round(team_total))} | Average: {int(round(team_avg))}</strong>
    </div>""")
    lines.append("</div>")

    return "".join(lines)

# Generate teams and display them in tabs
with tab2:
    st.markdown("### Meta Formation (4-2-3-1)")
    
    # Generate teams based on selection mode
    if st.session_state.use_custom_teams and st.session_state.custom_first_xi:
        # Use custom teams
        first_choice = {}
        second_choice = {}
        
        # Convert custom selections to team format
        formation_positions = [
            ("GK", "GK"),
            ("RB", "DL/DR"), ("CB1", "CB"), ("CB2", "CB"), ("LB", "DL/DR"),
            ("DM1", "DM"), ("DM2", "DM"),
            ("AMR", "AML/AMR"), ("AMC", "AMC"), ("AML", "AML/AMR"),
            ("ST", "ST")
        ]
        
        # Map custom selections to team indices
        for i, (pos_label, role) in enumerate(formation_positions):
            if pos_label in st.session_state.custom_first_xi:
                player_name = st.session_state.custom_first_xi[pos_label]
                # Find player index
                player_idx = comprehensive_df[comprehensive_df['Name'] == player_name].index
                if len(player_idx) > 0:
                    first_choice[i] = player_idx[0]
            
            if pos_label in st.session_state.custom_second_xi:
                player_name = st.session_state.custom_second_xi[pos_label]
                # Find player index
                player_idx = comprehensive_df[comprehensive_df['Name'] == player_name].index
                if len(player_idx) > 0:
                    second_choice[i] = player_idx[0]
    else:
        # Use Hungarian algorithm
        all_player_indices = list(range(n_players))
        first_choice = choose_starting_xi(all_player_indices, score_matrix)
        used_player_indices = set(first_choice.values())
        remaining_players = [i for i in all_player_indices if i not in used_player_indices]
        second_choice = choose_starting_xi(remaining_players, score_matrix)

    st.markdown("<br>", unsafe_allow_html=True)
    # Display both teams side by side
    col1, col2 = st.columns(2)

    with col1:
        team_name = "Custom First XI" if st.session_state.use_custom_teams else "First XI"
        first_xi_html = render_xi(first_choice, team_name)
        st.markdown(first_xi_html, unsafe_allow_html=True)

    with col2:
        team_name = "Custom Second XI" if st.session_state.use_custom_teams else "Second XI"
        second_xi_html = render_xi(second_choice, team_name)
        st.markdown(second_xi_html, unsafe_allow_html=True)


