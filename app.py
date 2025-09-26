# app.py
# Full Streamlit app — replace your old app.py with this file.
# Features:
# - Robust parser for pipe-delimited RTF tables with abbreviated headers (e.g. "Dri", "Fin", "Pac")
# - Exact weight presets and exact role names: GK, DRL, DC, WBRL, DM, MRL, MC, AMRL, AMC, SC
# - Upload RTF, parse automatically, choose role, view & download ranked CSV
#
# Requirements:
# pip install streamlit pandas numpy striprtf

import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from striprtf.striprtf import rtf_to_text

st.set_page_config(layout="wide", page_title="Auto Player Ranker (Exact Weights)")
st.title("Auto Player Ranker — upload .rtf, pick a role, get ranked players")
st.markdown("Upload your FM-style `.rtf` export (pipe `|` table). The app auto-parses abbreviations (e.g. `Dri`→Dribbling) and ranks all players using the exact weights from your chart.")

# -------------------------
# Exact weights (transcribed)
# -------------------------
# Canonical attributes used in weights
CANONICAL_ATTRIBUTES = [
    "Corners","Crossing","Dribbling","Finishing","First Touch","Free Kick Taking","Heading",
    "Long Shots","Long Throws","Marking","Passing","Penalty Taking","Tackling","Technique",
    "Aggression","Anticipation","Bravery","Composure","Concentration","Decisions","Determination",
    "Flair","Leadership","Off The Ball","Positioning","Teamwork","Vision","Work Rate",
    "Acceleration","Agility","Balance","Jumping Reach","Natural Fitness","Pace","Stamina","Strength",
    "Weaker Foot","Aerial Reach","Command of Area","Communication","Eccentricity","Handling","Kicking",
    "One on Ones","Punching (Tendency)","Reflexes","Rushing Out (Tendency)","Throwing"
]

WEIGHTS_BY_ROLE = {
    "GK": {
        "Corners":0,"Crossing":0,"Dribbling":0,"Finishing":0,"First Touch":0,"Free Kick Taking":0,"Heading":1,
        "Long Shots":0,"Long Throws":0,"Marking":0,"Passing":0,"Penalty Taking":0,"Tackling":0,"Technique":1,
        "Aggression":0,"Anticipation":3,"Bravery":6,"Composure":2,"Concentration":6,"Decisions":10,"Determination":0,
        "Flair":0,"Leadership":2,"Off The Ball":0,"Positioning":5,"Teamwork":2,"Vision":1,"Work Rate":1,
        "Acceleration":6,"Agility":8,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":3,"Stamina":1,"Strength":4,
        "Weaker Foot":3,"Aerial Reach":6,"Command of Area":6,"Communication":5,"Eccentricity":0,"Handling":8,"Kicking":5,
        "One on Ones":4,"Punching (Tendency)":0,"Reflexes":8,"Rushing Out (Tendency)":0,"Throwing":3
    },
    "DRL": {
        "Corners":1,"Crossing":2,"Dribbling":2,"Finishing":1,"First Touch":3,"Free Kick Taking":1,"Heading":2,
        "Long Shots":1,"Long Throws":1,"Marking":3,"Passing":2,"Penalty Taking":1,"Tackling":4,"Technique":2,
        "Aggression":0,"Anticipation":3,"Bravery":2,"Composure":2,"Concentration":4,"Decisions":7,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":1,"Positioning":4,"Teamwork":2,"Vision":2,"Work Rate":2,
        "Acceleration":7,"Agility":6,"Balance":2,"Jumping Reach":2,"Natural Fitness":0,"Pace":5,"Stamina":6,"Strength":4,
        "Weaker Foot":4,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "DC": {
        "Corners":1,"Crossing":1,"Dribbling":1,"Finishing":1,"First Touch":2,"Free Kick Taking":1,"Heading":5,
        "Long Shots":1,"Long Throws":1,"Marking":8,"Passing":2,"Penalty Taking":1,"Tackling":5,"Technique":1,
        "Aggression":0,"Anticipation":5,"Bravery":2,"Composure":2,"Concentration":4,"Decisions":10,"Determination":0,
        "Flair":0,"Leadership":2,"Off The Ball":1,"Positioning":8,"Teamwork":1,"Vision":1,"Work Rate":2,
        "Acceleration":6,"Agility":6,"Balance":2,"Jumping Reach":6,"Natural Fitness":0,"Pace":5,"Stamina":3,"Strength":6,
        "Weaker Foot":4.5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "WBRL": {
        "Corners":1,"Crossing":3,"Dribbling":2,"Finishing":1,"First Touch":3,"Free Kick Taking":1,"Heading":1,
        "Long Shots":1,"Long Throws":1,"Marking":2,"Passing":3,"Penalty Taking":1,"Tackling":3,"Technique":3,
        "Aggression":0,"Anticipation":3,"Bravery":1,"Composure":2,"Concentration":3,"Decisions":5,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":2,"Positioning":3,"Teamwork":2,"Vision":1,"Work Rate":2,
        "Acceleration":8,"Agility":5,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":6,"Stamina":7,"Strength":4,
        "Weaker Foot":4,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "DM": {
        "Corners":1,"Crossing":1,"Dribbling":2,"Finishing":2,"First Touch":4,"Free Kick Taking":1,"Heading":1,
        "Long Shots":3,"Long Throws":1,"Marking":3,"Passing":4,"Penalty Taking":1,"Tackling":7,"Technique":3,
        "Aggression":0,"Anticipation":5,"Bravery":1,"Composure":2,"Concentration":3,"Decisions":8,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":2,"Positioning":5,"Teamwork":2,"Vision":4,"Work Rate":4,
        "Acceleration":6,"Agility":6,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":4,"Stamina":4,"Strength":5,
        "Weaker Foot":5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "MRL": {
        "Corners":1,"Crossing":5,"Dribbling":3,"Finishing":2,"First Touch":4,"Free Kick Taking":1,"Heading":1,
        "Long Shots":2,"Long Throws":1,"Marking":1,"Passing":3,"Penalty Taking":1,"Tackling":2,"Technique":4,
        "Aggression":0,"Anticipation":3,"Bravery":1,"Composure":3,"Concentration":2,"Decisions":5,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":3,"Positioning":1,"Teamwork":2,"Vision":3,"Work Rate":3,
        "Acceleration":8,"Agility":6,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":6,"Stamina":5,"Strength":3,
        "Weaker Foot":5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "MC": {
        "Corners":1,"Crossing":1,"Dribbling":2,"Finishing":2,"First Touch":6,"Free Kick Taking":1,"Heading":1,
        "Long Shots":3,"Long Throws":1,"Marking":3,"Passing":6,"Penalty Taking":1,"Tackling":3,"Technique":4,
        "Aggression":0,"Anticipation":3,"Bravery":1,"Composure":3,"Concentration":2,"Decisions":7,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":2,"Positioning":3,"Teamwork":2,"Vision":6,"Work Rate":3,
        "Acceleration":6,"Agility":6,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":5,"Stamina":6,"Strength":4,
        "Weaker Foot":5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "AMRL": {
        "Corners":1,"Crossing":5,"Dribbling":5,"Finishing":2,"First Touch":5,"Free Kick Taking":1,"Heading":1,
        "Long Shots":2,"Long Throws":1,"Marking":1,"Passing":2,"Penalty Taking":1,"Tackling":2,"Technique":4,
        "Aggression":0,"Anticipation":3,"Bravery":1,"Composure":3,"Concentration":2,"Decisions":5,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":3,"Positioning":1,"Teamwork":2,"Vision":3,"Work Rate":3,
        "Acceleration":10,"Agility":6,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":10,"Stamina":7,"Strength":3,
        "Weaker Foot":6,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "AMC": {
        "Corners":1,"Crossing":1,"Dribbling":3,"Finishing":3,"First Touch":5,"Free Kick Taking":1,"Heading":1,
        "Long Shots":3,"Long Throws":1,"Marking":1,"Passing":4,"Penalty Taking":1,"Tackling":2,"Technique":5,
        "Aggression":0,"Anticipation":3,"Bravery":1,"Composure":3,"Concentration":2,"Decisions":6,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":2,"Positioning":2,"Teamwork":2,"Vision":6,"Work Rate":3,
        "Acceleration":9,"Agility":6,"Balance":2,"Jumping Reach":1,"Natural Fitness":0,"Pace":7,"Stamina":6,"Strength":3,
        "Weaker Foot":5.5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    },
    "SC": {
        "Corners":1,"Crossing":2,"Dribbling":5,"Finishing":8,"First Touch":6,"Free Kick Taking":1,"Heading":6,
        "Long Shots":2,"Long Throws":1,"Marking":1,"Passing":2,"Penalty Taking":1,"Tackling":1,"Technique":4,
        "Aggression":0,"Anticipation":5,"Bravery":1,"Composure":6,"Concentration":2,"Decisions":5,"Determination":0,
        "Flair":0,"Leadership":1,"Off The Ball":6,"Positioning":2,"Teamwork":1,"Vision":2,"Work Rate":2,
        "Acceleration":10,"Agility":6,"Balance":2,"Jumping Reach":5,"Natural Fitness":0,"Pace":7,"Stamina":6,"Strength":6,
        "Weaker Foot":7.5,"Aerial Reach":0,"Command of Area":0,"Communication":0,"Eccentricity":0,"Handling":0,"Kicking":0,
        "One on Ones":0,"Punching (Tendency)":0,"Reflexes":0,"Rushing Out (Tendency)":0,"Throwing":0
    }
}

# -------------------------
# Abbreviation mapping for parsing headers
# -------------------------
ABBR_MAP = {
    "Name":"Name","Position":"Position","Inf":"Inf","Age":"Age","Transfer Value":"Transfer Value",
    "Cor":"Corners","Cro":"Crossing","Dri":"Dribbling","Fin":"Finishing","Fir":"First Touch","Fre":"Free Kick Taking",
    "Hea":"Heading","Lon":"Long Shots","L Th":"Long Throws","LTh":"Long Throws","L_Th":"Long Throws",
    "Mar":"Marking","Pas":"Passing","Pen":"Penalty Taking","Tck":"Tackling","Tec":"Technique","Agg":"Aggression",
    "Ant":"Anticipation","Bra":"Bravery","Cmp":"Composure","Cnt":"Concentration","Dec":"Decisions","Det":"Determination",
    "Fla":"Flair","Ldr":"Leadership","OtB":"Off The Ball","Pos":"Positioning","Tea":"Teamwork","Vis":"Vision","Wor":"Work Rate",
    "Acc":"Acceleration","Agi":"Agility","Bal":"Balance","Jum":"Jumping Reach","Nat":"Natural Fitness","Pac":"Pace","Sta":"Stamina","Str":"Strength",
    "Weaker Foot":"Weaker Foot","Aer":"Aerial Reach","Cmd":"Command of Area","Com":"Communication","Ecc":"Eccentricity","Han":"Handling","Kic":"Kicking",
    "1v1":"One on Ones","1v1s":"One on Ones","Pun":"Punching (Tendency)","Ref":"Reflexes","TRO":"Rushing Out (Tendency)","Tro":"Rushing Out (Tendency)","Thr":"Throwing",
    # sometimes header had '.' or trimmed forms — keep safe fallbacks
    "Cor ":"Corners"
}

# -------------------------
# Parser: robust for pipe '|' tables with abbreviations
# -------------------------
def parse_players_from_text(text):
    """
    Parse a pipe-delimited table in the text.
    Returns (df, error_message_or_None).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = None
    # find the header line: must contain '|' and 'Name' and at least one common attribute code
    for ln in lines[:400]:
        if '|' in ln and 'Name' in ln and any(tok in ln for tok in ("Fin","Dri","Pas","Pac","Acc","Hea","Tck")):
            header = ln
            break
    if header is None:
        for ln in lines[:400]:
            if '|' in ln and 'Name' in ln:
                header = ln
                break
    if header is None:
        return None, "Could not find header/attribute line automatically."

    hdr_parts = [p.strip() for p in header.split('|')]
    # map header parts to canonical names
    canonical = []
    for h in hdr_parts:
        if not h:
            canonical.append("")  # keep placeholder for alignment
            continue
        mapped = ABBR_MAP.get(h, None)
        if mapped is None:
            # try cleaned variants
            key = h.replace('.', '').replace('_',' ').strip()
            mapped = ABBR_MAP.get(key, None)
        canonical.append(mapped if mapped is not None else h)

    # collect aligned rows
    start_idx = lines.index(header)
    rows = []
    for ln in lines[start_idx+1: start_idx+5000]:
        if '|' not in ln:
            continue
        parts = [p.strip() for p in ln.split('|')]
        # skip separator lines that are only dashes/underscores
        if all((not p) or set(p) <= set('- _.') for p in parts):
            continue
        # require exact alignment otherwise skip (robust strategy)
        if len(parts) != len(hdr_parts):
            continue
        row = {}
        for col_name, val in zip(canonical, parts):
            if col_name == "":
                continue
            row[col_name] = val
        # require a Name and that it's not the header repeated
        name = row.get("Name", "").strip()
        if not name or name.lower() == "name":
            continue
        rows.append(row)

    if not rows:
        return None, "Found header but no data rows detected (alignment mismatch)."

    df = pd.DataFrame(rows)

    # convert numeric-like columns to floats where appropriate
    for c in df.columns:
        if c in ("Name", "Position", "Transfer Value"):
            continue
        # extract first numeric token if present
        df[c] = df[c].astype(str).str.extract(r'(-?\d+(\.\d+)?)')[0].astype(float)

    return df, None

# -------------------------
# UI: file upload and main flow
# -------------------------
uploaded = st.file_uploader("Upload your `.rtf` file", type=["rtf"])

if not uploaded:
    st.info("Upload an RTF file (FM-style pipe `|` table). After successful parsing you'll be able to choose a role and rank players.")
    st.stop()

# try to extract text from RTF safely
raw = uploaded.read()
try:
    text = rtf_to_text(raw.decode('utf-8', errors='ignore'))
except Exception:
    try:
        text = rtf_to_text(raw)
    except Exception as e:
        st.error(f"Failed to convert RTF to text: {e}")
        st.stop()

st.subheader("Extracted text preview (first 1000 chars)")
st.code(text[:1000])

with st.spinner("Parsing players and attributes..."):
    df, err = parse_players_from_text(text)

if df is None:
    st.error("Automatic parsing failed: " + str(err))
    st.info("If automatic parsing failed: try a different RTF export format, or export CSV instead. You can also paste the RTF here and I can adjust the parser.")
    st.stop()

st.success(f"Detected {len(df)} players and {len([c for c in df.columns if c!='Name'])} columns/attributes.")
st.dataframe(df.head(10))

# Role selection (exact role names from image)
ROLE_OPTIONS = ["GK","DRL","DC","WBRL","DM","MRL","MC","AMRL","AMC","SC"]
role = st.selectbox("Choose role to rank for", ROLE_OPTIONS, index=ROLE_OPTIONS.index("SC"))

# build weight vector for selected role
selected_weights = WEIGHTS_BY_ROLE.get(role.upper(), {})
# determine which attributes from the weights exist in parsed df
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]

if not available_attrs:
    st.error("No matching attribute columns found in your RTF data. The parser found these columns: " + ", ".join(list(df.columns)[:60]))
    st.stop()

# build the weight series aligned to available_attrs
weights = pd.Series({a: float(selected_weights.get(a, 0.0)) for a in available_attrs})
weights = weights.reindex(available_attrs).fillna(0.0)

st.subheader("Exact weights used for ranking (from image)")
st.dataframe(weights.rename("Weight").to_frame())

# normalization option
normalize = st.checkbox("Normalize attribute values (divide by maximum possible)", value=True)
if normalize:
    max_val = st.number_input("Assumed max attribute value (e.g. 20)", value=20.0, min_value=1.0)
    attrs_df = df[available_attrs].fillna(0).astype(float) / float(max_val)
else:
    attrs_df = df[available_attrs].fillna(0).astype(float)

# compute scores
w_vec = weights.values.astype(float)
scores = attrs_df.values.dot(w_vec)
df_out = df.copy()
df_out["Score"] = scores
df_out_sorted = df_out.sort_values("Score", ascending=False).reset_index(drop=True)

st.subheader(f"Top players for role: {role}")
cols_to_show = ["Name","Position","Score"] + available_attrs
# limit columns shown in the table for readability (but allow full download)
st.dataframe(df_out_sorted[cols_to_show].head(100))

# show download button with full CSV
csv_bytes = df_out_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download ranked CSV (full)", csv_bytes, file_name=f"players_ranked_{role}.csv")

st.success("Done — you can upload different RTF files and select other roles. If parser fails on a different RTF, save that RTF and paste it here and I will help adapt the parser.")
