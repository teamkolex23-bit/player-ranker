import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
import unicodedata

# Page config with custom styling
st.set_page_config(
    layout="wide",
    page_title="FM24 Player Ranker",
    page_icon="⚽",
    initial_sidebar_state="expanded"
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
    background: #228B22;
    background-image:
        /* Center circle */
        radial-gradient(circle at 50% 50%, transparent 40px, rgba(255,255,255,0.4) 41px, rgba(255,255,255,0.4) 43px, transparent 44px),
        /* Center line (horizontal) */
        linear-gradient(0deg, transparent calc(50% - 1px), rgba(255,255,255,0.5) calc(50% - 1px), rgba(255,255,255,0.5) calc(50% + 1px), transparent calc(50% + 1px)),
        /* Goal areas (top and bottom) */
        linear-gradient(0deg, 
            rgba(255,255,255,0.3) 0%, 
            rgba(255,255,255,0.3) 2px, 
            transparent 2px, 
            transparent 15%, 
            rgba(255,255,255,0.3) 15%, 
            rgba(255,255,255,0.3) 17px, 
            transparent 17px,
            transparent 83px,
            rgba(255,255,255,0.3) 83px,
            rgba(255,255,255,0.3) 85%,
            transparent 85%,
            transparent calc(100% - 2px),
            rgba(255,255,255,0.3) calc(100% - 2px),
            rgba(255,255,255,0.3) 100%),
        /* Side lines (left and right) */
        linear-gradient(90deg, 
            rgba(255,255,255,0.4) 0%, 
            rgba(255,255,255,0.4) 2px, 
            transparent 2px, 
            transparent calc(100% - 2px), 
            rgba(255,255,255,0.4) calc(100% - 2px), 
            rgba(255,255,255,0.4) 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    font-family: monospace;
    border: 3px solid rgba(255,255,255,0.5);
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

# Header
st.markdown("""
<div class="main-header">
    <h1>⚽ FM24 Player Ranker</h1>
    <p>Advanced Football Manager 2024 Player Analysis & Team Selection</p>
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

def deduplicate_players(df):
    """Deduplicate players by name, keeping best version"""
    if len(df) <= 1:
        return df

    df = df.copy()
    df['_name_key'] = df['Name'].apply(create_name_key)
    df['_transfer_val_numeric'] = df.get('Transfer Value', '').apply(parse_transfer_value)

    df_sorted = df.sort_values(['Score', '_transfer_val_numeric'], ascending=[False, False])
    df_deduped = df_sorted.drop_duplicates(subset=['_name_key'], keep='first')
    df_deduped = df_deduped.drop(columns=['_name_key', '_transfer_val_numeric'])

    duplicates_removed = len(df) - len(df_deduped)
    if duplicates_removed > 0:
        st.success(f"✅ Removed {duplicates_removed} duplicate player(s)")

    return df_deduped.reset_index(drop=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Role selection
    ROLE_OPTIONS = list(WEIGHTS_BY_ROLE.keys())
    role = st.selectbox(
        "🎯 Choose Role to Analyze",
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index("ST") if "ST" in ROLE_OPTIONS else 0,
        help="Select the position you want to rank players for"
    )

    # Analysis info
    st.markdown("### 📈 Analysis Info")
    st.info("""
    **Role Weights**: Each position uses different attribute weightings based on tactical importance.

    **Scoring**: Higher scores indicate better fit for the selected role.

    **Deduplication**: Keeps the best version when duplicate names are found.
    """)

# File Upload Section
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## 📁 Upload Player Data")

st.markdown("""
<div class="info-box">
    <strong>📋 Upload Instructions:</strong><br>
    • Maximum 260 players can be processed<br>
    • Go to <a href="https://fmarenacalc.com" target="_blank">fmarenacalc.com</a> for HTML export guide<br>
    • Multiple files can be uploaded simultaneously<br>
    • Supports both .html and .htm files
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Select your FM24 player HTML files",
    type=["html", "htm"],
    accept_multiple_files=True,
    help="Upload the HTML files exported from Football Manager 2024"
)

st.markdown('</div>', unsafe_allow_html=True)

if not uploaded_files:
    st.stop()

# Processing files with progress indication
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## ⚡ Processing Files")

progress_bar = st.progress(0)
status_text = st.empty()

dfs = []
for i, uploaded in enumerate(uploaded_files):
    status_text.text(f'Processing {uploaded.name}...')
    progress_bar.progress((i + 1) / len(uploaded_files))

    raw = uploaded.read()
    try:
        html_text = raw.decode('utf-8', errors='ignore')
    except Exception:
        html_text = raw.decode('latin-1', errors='ignore')

    df, err = parse_players_from_html(html_text)
    if df is None:
        st.error(f"❌ Parsing failed for {uploaded.name}: {err}")
        continue

    st.success(f"✅ Loaded {len(df)} players from {uploaded.name}")

    df = merge_duplicate_columns(df)
    df = df.reset_index(drop=True)
    dfs.append(df)

status_text.text('Processing complete!')
st.markdown('</div>', unsafe_allow_html=True)

if not dfs:
    st.error("❌ No valid player data parsed from any uploaded file.")
    st.stop()

# Combine all data
df = pd.concat(dfs, ignore_index=True)
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]

if not available_attrs:
    st.error("❌ No matching attribute columns found. Detected columns: " + ", ".join(list(df.columns)))
    st.stop()

# Calculate scores and deduplicate
attrs_df = df[available_attrs].fillna(0).astype(float)
attrs_norm = attrs_df

selected_weights = WEIGHTS_BY_ROLE.get(role, {})
weights = pd.Series({a: float(selected_weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)

scores = attrs_norm.values.dot(weights.values.astype(float))
df['Score'] = scores

df_final = deduplicate_players(df)
df_sorted = df_final.sort_values("Score", ascending=False).reset_index(drop=True)

# Dashboard Overview
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## 📊 Analysis Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 class="blue" style="margin: 0;">👥 Total Players</h3>
        <h2 style="margin: 0.5rem 0;">{len(df_final)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    top_score = df_sorted['Score'].max()
    st.markdown(f"""
    <div class="metric-card">
        <h3 class="green" style="margin: 0;">🏆 Top Score</h3>
        <h2 style="margin: 0.5rem 0;">{top_score:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_score = df_sorted['Score'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h3 class="purple" style="margin: 0;">📈 Average Score</h3>
        <h2 style="margin: 0.5rem 0;">{avg_score:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    unique_positions = df_final['Position'].nunique() if 'Position' in df_final.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3 class="orange" style="margin: 0;">⚽ Positions</h3>
        <h2 style="margin: 0.5rem 0;">{unique_positions}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main Rankings with enhanced display
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"## 🏆 Top Players for {role}")

ranked = df_sorted.copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))

# Enhanced dataframe display
cols_to_show = [c for c in ["Rank", "Name", "Position", "Age", "Transfer Value", "Score"] if c in ranked.columns]

# Add search functionality
search_term = st.text_input("🔍 Search players", placeholder="Enter player name...")
if search_term:
    mask = ranked['Name'].str.contains(search_term, case=False, na=False)
    display_df = ranked[mask]
else:
    display_df = ranked

st.dataframe(
    display_df[cols_to_show + [c for c in available_attrs if c in display_df.columns]],
    use_container_width=True,
    height=400
)

st.markdown('</div>', unsafe_allow_html=True)

# Compact Role Analysis
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## ⚽ Role Analysis Overview")

# Create tabs for better organization
tab1, tab2 = st.tabs(["📊 Top 10 by Role", "📈 Score Distribution"])

with tab1:
    available_attrs_final = [a for a in CANONICAL_ATTRIBUTES if a in df_final.columns]
    attrs_df_final = df_final[available_attrs_final].fillna(0).astype(float)
    attrs_norm_final = attrs_df_final

    roles_per_row = 4
    for i in range(0, len(ROLE_OPTIONS), roles_per_row):
        cols = st.columns(roles_per_row)
        for j, r in enumerate(ROLE_OPTIONS[i:i+roles_per_row]):
            with cols[j]:
                st.markdown(f'<div class="role-header">{r}</div>', unsafe_allow_html=True)

                rw = WEIGHTS_BY_ROLE.get(r, {})
                w = pd.Series({a: float(rw.get(a, 0.0)) for a in available_attrs_final}).reindex(available_attrs_final).fillna(0.0)
                sc = attrs_norm_final.values.dot(w.values.astype(float))

                tmp = df_final.copy()
                tmp["Score"] = sc
                tmp_sorted = tmp.sort_values("Score", ascending=False).head(10).reset_index(drop=True)
                tmp_sorted.insert(0, "Rank", range(1, len(tmp_sorted) + 1))

                display_cols = ["Rank", "Name", "Score"]
                if "Age" in tmp_sorted.columns:
                    display_cols.insert(-1, "Age")

                tiny = tmp_sorted[display_cols].copy()
                tiny["Score"] = tiny["Score"].round(0).astype('Int64')

                st.dataframe(tiny, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("### Score Distribution Analysis")
    score_stats = df_sorted['Score'].describe()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Minimum Score", f"{score_stats['min']:.0f}")
        st.metric("25th Percentile", f"{score_stats['25%']:.0f}")
        st.metric("Median Score", f"{score_stats['50%']:.0f}")

    with col2:
        st.metric("75th Percentile", f"{score_stats['75%']:.0f}")
        st.metric("Maximum Score", f"{score_stats['max']:.0f}")
        st.metric("Standard Deviation", f"{score_stats['std']:.0f}")

st.markdown('</div>', unsafe_allow_html=True)

# Starting XI Section
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## ⚽ Starting XI Generator")

st.markdown("""
<div class="info-box">
    <strong>🎯 Formation Analysis:</strong><br>
    Select your 10 outfield positions below. The app uses the Hungarian algorithm for optimal player-position assignment based on role compatibility scores to maximize overall team strength.
</div>
""", unsafe_allow_html=True)

# Fixed Formation Setup
st.markdown("### ⚙️ Team Formation (4-2-3-1)")
st.markdown("Using a fixed 4-2-3-1 formation with optimal spacing for team display.")

# Fixed formation lines with your desired layout
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

# Precompute role weight vectors
role_weight_vectors = {}
for _, role_key in positions:
    if role_key not in role_weight_vectors: # Avoid re-computing for same role
        rw = WEIGHTS_BY_ROLE.get(role_key, {})
        role_weight_vectors[role_key] = np.array([float(rw.get(a, 0.0)) for a in available_attrs_final], dtype=float)

# Compute score matrix
score_matrix = np.zeros((n_players, n_positions), dtype=float)
for i_idx in range(n_players):
    player_attr_vals = attrs_norm_final.iloc[i_idx].values if len(available_attrs_final) > 0 else np.zeros((len(available_attrs_final),), dtype=float)
    for p_idx, (_, role_key) in enumerate(positions):
        w = role_weight_vectors[role_key]
        score_matrix[i_idx, p_idx] = float(np.dot(player_attr_vals, w))

# Hungarian algorithm assignment
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

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

    # Format as table
    lines = [f"<div class='xi-formation'>"]
    lines.append(f"<h3 style='text-align: center; margin-bottom: 1rem;'>{team_name}</h3>")

    for pos_label, name, sel_score, role_key in rows:
        if role_key == "EMPTY":
            lines.append("<div style='height: 20px;'></div>")  # Empty space
        else:
            sel_score_int = int(round(sel_score))
            lines.append(f"""<div style='display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.1); border-radius: 5px;'>
                <span style='font-weight: bold; min-width: 5rem;'>{pos_label}</span>
                <span style='font-weight: bold; flex-grow: 1; text-align: center;'>{name}</span>
                <span style='min-width: 4rem; text-align: right;'>{sel_score_int} pts</span>
            </div>""")

    lines.append(f"""<div style='margin-top: 2rem; padding-top: 1rem; border-top: 2px solid rgba(255,255,255,0.3); text-align: center;'>
        <strong>Team Total: {int(round(team_total))} | Average: {int(round(team_avg))}</strong>
    </div>""")
    lines.append("</div>")

    return "".join(lines)
# Generate both teams
all_player_indices = list(range(n_players))
first_choice = choose_starting_xi(all_player_indices, score_matrix)
used_player_indices = set(first_choice.values())
remaining_players = [i for i in all_player_indices if i not in used_player_indices]
second_choice = choose_starting_xi(remaining_players, score_matrix)

st.markdown("<br>", unsafe_allow_html=True)
# Display both teams side by side
col1, col2 = st.columns(2)

with col1:
    first_xi_html = render_xi(first_choice, "First XI")
    st.markdown(first_xi_html, unsafe_allow_html=True)

with col2:
    second_xi_html = render_xi(second_choice, "Second XI")
    st.markdown(second_xi_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)





