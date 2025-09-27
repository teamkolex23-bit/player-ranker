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

# Custom CSS for better styling
st.markdown("""
<style>
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
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #1f4e79;
    }
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4169e1;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #e9ecef;
    }
    .role-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .xi-formation {
        background: #1a5d1a;
        background-image:
            linear-gradient(90deg, rgba(255,255,255,0.1) 50%, transparent 50%),
            linear-gradient(rgba(255,255,255,0.1) 50%, transparent 50%);
        background-size: 20px 20px;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        font-family: monospace;
    }
    .stProgress .st-bo {
        background-color: #e8f4fd;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
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

def format_score_with_color(score, percentile_ranks):
    """Color code scores based on percentile ranking"""
    if score >= percentile_ranks[90]:
        return f'<span style="color: #28a745; font-weight: bold;">{score:.0f}</span>'  # Green - Top 10%
    elif score >= percentile_ranks[75]:
        return f'<span style="color: #fd7e14; font-weight: bold;">{score:.0f}</span>'  # Orange - Top 25%
    elif score >= percentile_ranks[50]:
        return f'<span style="color: #6f42c1;">{score:.0f}</span>'  # Purple - Top 50%
    elif score >= percentile_ranks[25]:
        return f'<span style="color: #6c757d;">{score:.0f}</span>'  # Gray - Bottom 50%
    else:
        return f'<span style="color: #dc3545;">{score:.0f}</span>'  # Red - Bottom 25%

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
        <h3 style="color: #1f4e79; margin: 0;">👥 Total Players</h3>
        <h2 style="margin: 0.5rem 0;">{len(df_final)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    top_score = df_sorted['Score'].max()
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #28a745; margin: 0;">🏆 Top Score</h3>
        <h2 style="margin: 0.5rem 0;">{top_score:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_score = df_sorted['Score'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #6f42c1; margin: 0;">📈 Average Score</h3>
        <h2 style="margin: 0.5rem 0;">{avg_score:.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    unique_positions = df_final['Position'].nunique() if 'Position' in df_final.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #fd7e14; margin: 0;">⚽ Positions</h3>
        <h2 style="margin: 0.5rem 0;">{unique_positions}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main Rankings with enhanced display
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"## 🏆 Top Players for {role}")

ranked = df_sorted.copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))

# Calculate percentile ranks for color coding
percentile_ranks = df_sorted['Score'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
percentile_ranks = {int(k*100): v for k, v in percentile_ranks.items()}

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
    Uses Hungarian algorithm for optimal player-position assignment based on role compatibility scores.
    Players are assigned to maximize overall team strength while avoiding duplicates.
</div>
""", unsafe_allow_html=True)

# Formation setup
positions = [
    ("GK", "GK"),
    ("RB", "DL/DR"),
    ("CB1", "CB"),
    ("CB2", "CB"),
    ("LB", "DL/DR"),
    ("DM1", "DM"),
    ("DM2", "DM"),
    ("AMR", "AML/AMR"),
    ("AMC", "AMC"),
    ("AML", "AML/AMR"),
    ("ST", "ST"),
]

n_players = len(df_final)
n_positions = len(positions)
player_names = df_final["Name"].astype(str).tolist()

# Precompute role weight vectors
role_weight_vectors = {}
for _, role_key in positions:
    rw = WEIGHTS_BY_ROLE.get(role_key, {})
    role_weight_vectors[role_key] = np.array([float(rw.get(a, 0.0)) for a in available_attrs_final], dtype=float)

# Compute score matrix
score_matrix = np.zeros((n_players, n_positions), dtype=float)
for i_idx in range(n_players):
    player_attr_vals = attrs_norm_final.iloc[i_idx].values if len(available_attrs_final) > 0 else np.zeros((len(available_attrs_final),), dtype=float)
    for p_idx, (_, role_key) in enumerate(positions):
        w = role_weight_vectors[role_key]
        score_matrix[i_idx, p_idx] = float(np.dot(player_attr_vals, w))

# Find best non-ST role for each player
all_role_keys = list(WEIGHTS_BY_ROLE.keys())
all_role_vectors = {rk: np.array([float(WEIGHTS_BY_ROLE[rk].get(a, 0.0)) for a in available_attrs_final], dtype=float) for rk in all_role_keys}

player_best_role = []
for i_idx in range(n_players):
    player_attr_vals = attrs_norm_final.iloc[i_idx].values if len(available_attrs_final) > 0 else np.zeros((len(available_attrs_final),), dtype=float)
    best_score = -1e9
    best_role = None

    for rk, vec in all_role_vectors.items():
        if rk == "ST":
            continue
        sc = float(np.dot(player_attr_vals, vec))
        if sc > best_score:
            best_score = sc
            best_role = rk

    player_best_role.append((best_role, best_score))

# Hungarian algorithm assignment
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

def choose_starting_xi(available_player_indices):
    avail = list(available_player_indices)
    m = max(len(avail), n_positions)
    cost = np.zeros((m, n_positions), dtype=float)

    if len(avail) > 0:
        cost[:len(avail), :] = -score_matrix[avail, :]

    if linear_sum_assignment is None:
        # Greedy fallback
        chosen = {}
        used_players = set()

        for p_idx in range(n_positions):
            best_p = None
            best_sc = -1e9

            for i_idx in avail:
                if i_idx in used_players:
                    continue
                sc = float(score_matrix[int(i_idx), p_idx])
                if sc > best_sc:
                    best_sc = sc
                    best_p = int(i_idx)

            if best_p is not None:
                chosen[p_idx] = int(best_p)
                used_players.add(best_p)

        return chosen
    else:
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen = {}

        for r, c in zip(row_ind, col_ind):
            if r < len(avail) and c < n_positions:
                chosen[c] = int(avail[r])

        return chosen

def render_xi(chosen_map, team_name="Team"):
    rows = []
    sel_scores = []

    for pos_idx, (pos_label, role_key) in enumerate(positions):
        if pos_idx in chosen_map:
            p_idx = int(chosen_map[pos_idx])
            name = str(player_names[p_idx]) if p_idx is not None else ""
            sel_score = float(score_matrix[p_idx, pos_idx]) if p_idx is not None else 0.0
            best_role, best_score = player_best_role[p_idx] if p_idx is not None else ("", 0.0)
            rows.append((pos_label, name, sel_score, best_role, best_score, p_idx))
            sel_scores.append(sel_score)
        else:
            rows.append((pos_label, "", 0.0, "", 0.0, None))

    team_total = float(sum([r[2] for r in rows if r[5] is not None]))
    placed_scores = [r[2] for r in rows if r[5] is not None]
    team_avg = float(np.mean(placed_scores)) if placed_scores else 0.0

    def color_for_diff(diff, current_max_score=400.0):
        cap = current_max_score
        diff = max(-cap, min(cap, diff))

        if diff > 0:
            ratio = diff / cap
            r = int(255 * (1 - ratio))
            g = 255
            b = int(255 * (1 - ratio))
        elif diff < 0:
            ratio = -diff / cap
            r = 255
            g = int(255 * (1 - ratio))
            b = int(255 * (1 - ratio))
        else:
            r = g = b = 255

        return f"rgb({r},{g},{b})"

    # Format as table
    lines = [f"<div class='xi-formation'>"]
    lines.append(f"<h3 style='text-align: center; margin-bottom: 1rem;'>{team_name}</h3>")

    group_breaks = {"GK": "🥅 GOALKEEPER", "RB": "🛡️ DEFENSE", "DM1": "⚙️ MIDFIELD", "AMR": "⚡ ATTACK"}

    for pos_label, name, sel_score, best_role, best_score, p_idx in rows:
        if pos_label in group_breaks:
            lines.append(f"<div style='margin: 1rem 0; font-weight: bold; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 0.5rem;'>{group_breaks[pos_label]}</div>")

        if name:
            diff = sel_score - team_avg
            color = color_for_diff(diff)
            sel_score_int = int(round(float(sel_score)))
            best_score_int = int(round(float(best_score)))

            lines.append(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.1); border-radius: 5px;'>
                <span style='font-weight: bold; min-width: 3rem;'>{pos_label}</span>
                <span style='color:{color}; font-weight: bold; flex-grow: 1; text-align: center; text-shadow: 1px 1px 3px #000;'>{name}</span>
                <span style='min-width: 4rem; text-align: right;'>{sel_score_int} pts</span>
            </div>
            """)
        else:
            lines.append(f"<div style='padding: 0.5rem; opacity: 0.5;'>{pos_label}: No player assigned</div>")

    lines.append(f"""
    <div style='margin-top: 2rem; padding-top: 1rem; border-top: 2px solid rgba(255,255,255,0.3); text-align: center;'>
        <strong>Team Total: {int(round(team_total))} | Average: {int(round(team_avg))}</strong>
    </div>
    """)
    lines.append("</div>")

    return "".join(lines), team_total

# Generate both teams
all_player_indices = list(range(n_players))
first_choice = choose_starting_xi(all_player_indices)
used_player_indices = set(first_choice.values())
remaining_players = [i for i in all_player_indices if i not in used_player_indices]
second_choice = choose_starting_xi(remaining_players)

# Display both teams side by side
col1, col2 = st.columns(2)

with col1:
    first_lines, first_total = render_xi(first_choice, "First XI")
    st.markdown(first_lines, unsafe_allow_html=True)

with col2:
    second_lines, second_total = render_xi(second_choice, "Second XI")
    st.markdown(second_lines, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Download Section
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("## 📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📊 Download Full Rankings (CSV)",
        csv_bytes,
        file_name=f"players_ranked_{role}_{len(df_final)}_players.csv",
        mime="text/csv"
    )

with col2:
    # Create summary report
    summary_data = {
        'Analysis_Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
        'Role_Analyzed': [role],
        'Total_Players': [len(df_final)],
        'Top_Player': [df_sorted.iloc[0]['Name']],
        'Top_Score': [df_sorted.iloc[0]['Score']],
        'Average_Score': [df_sorted['Score'].mean()],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📋 Download Analysis Summary",
        summary_csv,
        file_name=f"analysis_summary_{role}.csv",
        mime="text/csv"
    )

st.markdown('</div>', unsafe_allow_html=True)
