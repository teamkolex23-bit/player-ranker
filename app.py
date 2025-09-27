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
    # Role selection
    ROLE_OPTIONS = list(WEIGHTS_BY_ROLE.keys())
    role = st.selectbox(
        "Choose Position to display score for",
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index("ST") if "ST" in ROLE_OPTIONS else 0,
        help="Not necessary since I also include a top 10 for every position"
    )

    # Analysis info
    st.markdown("### Analysis Info")
    st.info("""
    **Position Weights Score**: From FMScout (IMO still accurate even if outdated unlike the FM-Arena attribute testing)

    **Scoring**: Higher scores means better fit in that position. Some positions are just naturally inflated for every player.

    **Deduplication**: Keeps the best version of duplicate entries.
    """)

st.markdown("""
<div class="info-box">
    <strong>Upload Instructions:</strong><br>
    • Go to <a href="https://fmarenacalc.com" target="_blank">fmarenacalc.com</a> for HTML export guide<br>
    • Maximum 260 players can be exported per html file, so make sure to narrow your search before each print screen<br>
    • Multiple files can be uploaded simultaneously<br>
    • Supports .html only and not .rtf like the NewGAN mod
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Follow the 'Upload Instructions:' above and upload the HTML files you print screened from FM24 here below",
    type=["html", "htm"],
    accept_multiple_files=True
)


if not uploaded_files:
    st.stop()

# Process files and show summary
dfs = []
file_results = []

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
        continue

    df = merge_duplicate_columns(df)
    df = df.reset_index(drop=True)
    dfs.append(df)
    file_results.append(f"✅ {uploaded.name}: {len(df)} players loaded")

# Complete the progress bar
progress_bar.progress(1.0)
status_text.text('Processing complete!')

if not dfs:
    st.error("❌ No valid player data parsed from any uploaded file.")
    st.stop()

# Show results
for result in file_results:
    st.write(result)

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

# Main Rankings with enhanced display
st.markdown(f"## All players score as a {role}")

ranked = df_sorted.copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))

# Enhanced dataframe display
cols_to_show = [c for c in ["Rank", "Name", "Position", "Age", "Transfer Value", "Score"] if c in ranked.columns]

display_df = ranked

st.dataframe(
    display_df[cols_to_show + [c for c in available_attrs if c in display_df.columns]],
    use_container_width=True,
    height=400
)

# Compact Role Analysis
st.markdown("## Top 10 in each position")

# Role analysis without tabs
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


# Starting XI Section
st.markdown("""
<div class="info-box">
    <strong>Formation Analysis:</strong><br>
    Hungarian algorithm used to create the best starting 11, it also creates a secondary team with 0 overlap in players from the first team. Some ridiculous options occur like a DM being recommended as a ST but it should theoretically be true as long as their hidden attributes aren't terrible.
</div>
""", unsafe_allow_html=True)

# Fixed Formation Setup
st.markdown("### Meta Formation (4-2-3-1)")

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
            
            # Calculate color based on difference from team average
            diff_from_avg = sel_score - team_avg
            
            # Normalize difference to a 0-1 scale (400 points = full intensity)
            intensity = min(abs(diff_from_avg) / 400.0, 1.0)
            
            # Calculate RGB values
            if diff_from_avg >= 0:  # Above average - green
                red = int(255 * (1 - intensity))
                green = 255
                blue = int(255 * (1 - intensity))
            else:  # Below average - red
                red = 255
                green = int(255 * (1 - intensity))
                blue = int(255 * (1 - intensity))
            
            name_color = f"rgb({red}, {green}, {blue})"
            
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
# ---------- START TEAMBUILDER UI (CLIENT-SIDE PITCH) ----------
import json
from streamlit.components.v1 import html as st_html

# Prepare a lightweight JSON payload for the client UI
players_payload = []
for i in range(len(player_names)):
    players_payload.append({
        "id": int(i),
        "name": str(player_names[i]),
        "scores": [float(score_matrix[i, p]) for p in range(score_matrix.shape[1])]
    })

positions_payload = [{"index": idx, "label": label, "role": role}
                     for idx, (label, role) in enumerate(positions)]

payload = {
    "players": players_payload,
    "positions": positions_payload,
    "n_players": len(players_payload),
    "n_positions": len(positions_payload)
}

# HTML + JS component: client-side pitch UI (instant, no reruns)
COMPONENT_HTML = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{ margin:0; font-family: Arial, Helvetica, sans-serif; color:#fff; background:transparent; }}
  .container {{ display:flex; gap:16px; }}
  .panel {{ width:48%; }}
  .controls {{ display:flex; gap:8px; margin-bottom:8px; }}
  .pitch {{
    position: relative;
    background: linear-gradient(#3b8a3b, #2f702f);
    border: 4px solid #0b5a0b;
    height: 620px;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.6);
    overflow: visible;
  }}
  .slot {{
    position: absolute;
    transform: translate(-50%,-50%);
    min-width: 160px;
    max-width: 220px;
    padding: 8px 10px;
    border-radius: 8px;
    text-align: center;
    background: rgba(255,255,255,0.06);
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.12);
    transition: transform .08s ease, box-shadow .08s ease;
    user-select: none;
  }}
  .slot:hover {{ transform: translate(-50%,-50%) scale(1.02); box-shadow: 0 8px 18px rgba(0,0,0,0.45); }}
  .slot .label {{ font-weight:700; display:block; margin-bottom:6px; color:#eaf7ea; }}
  .slot .player {{ font-size:14px; color:#fff; font-weight:600; display:block; }}
  .slot .score {{ font-size:12px; opacity:0.9; margin-top:4px; color:#ddd; }}
  .slot.empty .player {{ color: rgba(255,255,255,0.5); font-weight:400; }}
  .slot.selected {{ background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); box-shadow: 0 8px 24px rgba(0,0,0,0.6); border-color: rgba(255,255,255,0.18); }}
  .candidates {{
    position: absolute;
    left: 100%;
    top: -6px;
    width: 320px;
    max-height: 420px;
    overflow:auto;
    background: rgba(10,10,10,0.9);
    padding:8px;
    border-radius:6px;
    margin-left:12px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.6);
    z-index: 9999;
  }}
  .cand-row {{ display:flex; justify-content:space-between; padding:6px; border-bottom:1px solid rgba(255,255,255,0.04); align-items:center; color:#e6e6e6; }}
  .cand-row button {{ margin-left:8px; background:#1b5f1b; color:#fff; border:none; padding:6px 8px; border-radius:6px; cursor:pointer; }}
  .summary-box {{ padding:8px; border-radius:6px; background: rgba(0,0,0,0.5); margin-top:8px; color:#fff; }}
  .export-btn {{ margin-top:6px; padding:8px 10px; border-radius:6px; background:#8a6c00; border:none; color:#fff; cursor:pointer; }}
  .top-controls {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
  .small-btn {{ padding:6px 8px; border-radius:6px; border:none; cursor:pointer; background:#23527a; color:#fff; }}
</style>
</head>
<body>
<div class="top-controls" style="margin-bottom:6px;">
  <div style="font-weight:700;">Client-side Team Builder — instant UI</div>
  <div>
    <button id="exportCsv" class="export-btn">Export teams as CSV</button>
  </div>
</div>

<div class="container">
  <div class="panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
      <h3 style="margin:0">First XI</h3>
      <div>
        <button id="autopick1" class="small-btn">Autopick</button>
        <button id="clear1" class="small-btn" style="margin-left:6px;background:#6b2b2b;">Clear</button>
      </div>
    </div>
    <div id="pitch1" class="pitch"></div>
    <div class="summary-box" id="summary1"></div>
  </div>

  <div class="panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
      <h3 style="margin:0">Second XI</h3>
      <div>
        <button id="autopick2" class="small-btn">Autopick (exclude 1)</button>
        <button id="clear2" class="small-btn" style="margin-left:6px;background:#6b2b2b;">Clear</button>
      </div>
    </div>
    <div id="pitch2" class="pitch"></div>
    <div class="summary-box" id="summary2"></div>
  </div>
</div>

<div style="margin-top:10px; color:#ddd;">Tip: click a slot to pick the best available player. Click again to see top alternatives and replace/remove.</div>

<script>
const DATA = {json.dumps(payload)};

// A coordinate map (x% left, y% top) — tweak to taste.
const coordMap = {{
  "GK": [50, 88],
  "RB": [80, 68],
  "LB": [20, 68],
  "CB": [50, 64],
  "DL/DR": [80, 68],
  "WBL/WBR": [80, 56],
  "DM": [50, 56],
  "CM": [50, 46],
  "ML/MR": [20, 40],
  "AML/AMR": [80, 28],
  "AMC": [50, 30],
  "ST": [50, 14],
  "DEFAULT": [50,50]
}};

function coordsForLabel(label, role) {{
  if (coordMap[role]) return coordMap[role];
  if (coordMap[label]) return coordMap[label];
  return coordMap["DEFAULT"];
}}

const teams = [Array(DATA.n_positions).fill(null), Array(DATA.n_positions).fill(null)];

function createPitch(containerId, teamIdx) {{
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  container.style.position = "relative";
  DATA.positions.forEach(pos => {{
    const [x, y] = coordsForLabel(pos.label, pos.role);
    const el = document.createElement('div');
    el.className = 'slot empty';
    el.style.left = x + '%';
    el.style.top = y + '%';
    el.dataset.pos = pos.index;
    el.dataset.team = teamIdx;

    const lbl = document.createElement('span');
    lbl.className = 'label';
    lbl.textContent = pos.label;

    const player = document.createElement('span');
    player.className = 'player';
    player.textContent = '---';

    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = '';

    el.appendChild(lbl);
    el.appendChild(player);
    el.appendChild(score);

    container.appendChild(el);

    el.addEventListener('click', (e) => {{
      e.stopPropagation();
      onSlotClick(teamIdx, pos.index, el);
    }});
  }});
  updateSummary(teamIdx);
}}

function getBestAvailable(posIdx, excludeSet) {{
  let best = null;
  let bestScore = -Infinity;
  for (let i=0;i<DATA.players.length;i++) {{
    if (excludeSet.has(i)) continue;
    const s = DATA.players[i].scores[posIdx];
    if (s > bestScore) {{ bestScore = s; best = i; }}
  }}
  return {{ idx: best, score: bestScore }};
}}

function getCandidates(posIdx, excludeSet, topN=30) {{
  const arr = [];
  for (let i=0;i<DATA.players.length;i++) {{
    if (excludeSet.has(i)) continue;
    arr.push([i, DATA.players[i].name, DATA.players[i].scores[posIdx]]);
  }}
  arr.sort((a,b)=> b[2]-a[2]);
  return arr.slice(0, topN);
}}

function onSlotClick(teamIdx, posIdx, el) {{
  // If currently empty -> pick best available
  const otherTeamIdx = 1 - teamIdx;
  const currentTeamSet = new Set(teams[teamIdx].filter(x=>x!==null));
  const excludeForPick = new Set(teams[otherTeamIdx].filter(x=>x!==null));
  // also exclude other positions in same team to avoid duplicates
  teams[teamIdx].forEach(x=>{{ if (x!==null) excludeForPick.add(x); }});

  const curAssigned = teams[teamIdx][posIdx];
  // remove any existing candidate panel
  removeAllCandidatePanels();

  if (curAssigned === null) {{
    const best = getBestAvailable(posIdx, excludeForPick);
    if (best.idx !== null) {{
      teams[teamIdx][posIdx] = best.idx;
      refreshPitches();
    }}
  }} else {{
    // open candidate panel for this slot
    openCandidatePanel(teamIdx, posIdx, el);
  }}
}}

function openCandidatePanel(teamIdx, posIdx, el) {{
  const otherTeamIdx = 1 - teamIdx;
  const excludeForList = new Set(teams[otherTeamIdx].filter(x=>x!==null));
  // allow current assigned player to appear
  const candidates = getCandidates(posIdx, excludeForList, 50);

  const panel = document.createElement('div');
  panel.className = 'candidates';

  candidates.forEach(c => {{
    const row = document.createElement('div');
    row.className = 'cand-row';
    const left = document.createElement('div');
    left.style.flex = '1';
    left.textContent = `${{c[1]}} — ${{Math.round(c[2])}} pts`;

    const right = document.createElement('div');
    const btnReplace = document.createElement('button');
    btnReplace.textContent = 'Replace';
    btnReplace.addEventListener('click', (ev)=>{{ 
      teams[teamIdx][posIdx] = c[0];
      removeAllCandidatePanels();
      refreshPitches();
    }});
    const btnRemove = document.createElement('button');
    btnRemove.style.background='#6b2b2b';
    btnRemove.textContent = 'Remove';
    btnRemove.addEventListener('click', (ev)=>{{ 
      teams[teamIdx][posIdx] = null;
      removeAllCandidatePanels();
      refreshPitches();
    }});
    right.appendChild(btnReplace);
    right.appendChild(btnRemove);

    row.appendChild(left);
    row.appendChild(right);
    panel.appendChild(row);
  }});

  // Close button
  const close = document.createElement('div');
  close.style.textAlign='center';
  close.style.marginTop='6px';
  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Close';
  closeBtn.style.background = '#23527a';
  closeBtn.style.padding = '6px 8px';
  closeBtn.style.borderRadius = '6px';
  closeBtn.style.border = 'none';
  closeBtn.style.color = '#fff';
  closeBtn.addEventListener('click', ()=>{ removeAllCandidatePanels(); });
  close.appendChild(closeBtn);
  panel.appendChild(close);

  // position panel relative to slot; try to avoid overflow
  el.appendChild(panel);
}}

function removeAllCandidatePanels() {{
  document.querySelectorAll('.candidates').forEach(n=>n.remove());
}}

function refreshPitches() {{
  // update all slot UIs and summary
  for (let teamIdx=0; teamIdx<2; teamIdx++) {{
    const container = document.getElementById('pitch'+(teamIdx+1));
    container.querySelectorAll('.slot').forEach(slot => {{
      const posIdx = parseInt(slot.dataset.pos,10);
      const pid = teams[teamIdx][posIdx];
      const playerEl = slot.querySelector('.player');
      const scoreEl = slot.querySelector('.score');
      if (pid === null) {{
        slot.classList.remove('selected');
        slot.classList.add('empty');
        playerEl.textContent = '---';
        scoreEl.textContent = '';
      }} else {{
        slot.classList.add('selected');
        slot.classList.remove('empty');
        playerEl.textContent = DATA.players[pid].name;
        scoreEl.textContent = Math.round(DATA.players[pid].scores[posIdx]) + ' pts';
      }}
    }});
    updateSummary(teamIdx);
  }}
}}

function updateSummary(teamIdx) {{
  const summaryEl = document.getElementById('summary'+(teamIdx+1));
  const assigned = teams[teamIdx].map((pid,posIdx)=> pid===null ? null : DATA.players[pid].scores[posIdx]).filter(x=>x!==null);
  const total = assigned.reduce((a,b)=>a+b,0);
  const avg = assigned.length ? (total/assigned.length) : 0;
  summaryEl.innerHTML = `<strong>Team total:</strong> ${{Math.round(total)}} &nbsp; | &nbsp; <strong>Average:</strong> ${{Math.round(avg)}} &nbsp; | &nbsp; <strong>Players:</strong> ${{assigned.length}}`;
}}

function autopick(teamIdx, excludeOtherTeam=false) {{
  // Greedy autopick: best available for each slot (skips players already picked)
  const otherTeamIdx = 1 - teamIdx;
  const excludeSet = new Set();
  if (excludeOtherTeam) {{
    teams[otherTeamIdx].forEach(x=>{{ if (x!==null) excludeSet.add(x); }});
  }}
  // clear team
  teams[teamIdx] = Array(DATA.n_positions).fill(null);
  // Fill slots greedily
  for (let pos=0; pos<DATA.n_positions; pos++) {{
    // compute best ignoring excludeSet and already-chosen in this team
    const currentlySelected = new Set(teams[teamIdx].filter(x=>x!==null));
    const combinedExclude = new Set([...excludeSet, ...currentlySelected]);
    const best = getBestAvailable(pos, combinedExclude);
    if (best.idx !== null) {{
      teams[teamIdx][pos] = best.idx;
    }}
  }}
  refreshPitches();
}}

function clearTeam(teamIdx) {{
  teams[teamIdx] = Array(DATA.n_positions).fill(null);
  refreshPitches();
}}

function exportCSV() {{
  const rows = [['Team','Slot','Role','Player','Score']];
  for (let teamIdx=0; teamIdx<2; teamIdx++) {{
    for (let pos=0; pos<DATA.n_positions; pos++) {{
      const p = DATA.positions[pos];
      const pid = teams[teamIdx][pos];
      const pname = pid===null ? '' : DATA.players[pid].name;
      const score = pid===null ? '' : Math.round(DATA.players[pid].scores[pos]);
      rows.push([`Team ${{teamIdx+1}}`, p.label, p.role, pname, score]);
    }}
  }}
  const csvContent = rows.map(r => r.map(c => `"${{String(c).replace(/"/g,'""')}}"`).join(',')).join('\\n');
  const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'teams_export.csv';
  a.click();
  URL.revokeObjectURL(url);
}}

// init
createPitch('pitch1', 0);
createPitch('pitch2', 1);
refreshPitches();

// wire controls
document.getElementById('autopick1').addEventListener('click', ()=>{{ autopick(0, false); }});
document.getElementById('clear1').addEventListener('click', ()=>{{ clearTeam(0); }});
document.getElementById('autopick2').addEventListener('click', ()=>{{ autopick(1, true); }});
document.getElementById('clear2').addEventListener('click', ()=>{{ clearTeam(1); }});
document.getElementById('exportCsv').addEventListener('click', ()=>{{ exportCSV(); }});

// close candidate panels when clicking outside
document.addEventListener('click', (e) => {{ removeAllCandidatePanels(); }});
</script>
</body>
</html>
"""

# Render component. Adjust height as you want (800+ if you have tall screens).
st_html(COMPONENT_HTML, height=820, scrolling=True)

# ---------- END TEAMBUILDER UI (CLIENT-SIDE PITCH) ----------
