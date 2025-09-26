# app.py
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from striprtf.striprtf import rtf_to_text

st.set_page_config(layout="wide", page_title="Auto Player Ranker (Exact Chart Weights)")

st.title("Auto Player Ranker — upload .rtf, pick a role, get ranked players")
st.markdown("This version uses the exact weights from your weight-chart image and the exact role names: GK, DRL, DC, WBRL, DM, MRL, MC, AMRL, AMC, SC.")

uploaded = st.file_uploader("Upload your `.rtf` file", type=["rtf"])

# Canonical attribute list (must match names the parser will try to extract)
ATTRIBUTES = [
    "Name","Corners","Crossing","Dribbling","Finishing","First Touch","Free Kick Taking",
    "Heading","Long Shots","Long Throws","Marking","Passing","Penalty Taking","Tackling","Technique",
    "Aggression","Anticipation","Bravery","Composure","Concentration","Decisions","Determination",
    "Flair","Leadership","Off The Ball","Positioning","Teamwork","Vision","Work Rate",
    "Acceleration","Agility","Balance","Jumping Reach","Natural Fitness","Pace","Stamina","Strength",
    "Weaker Foot","Aerial Reach","Command of Area","Communication","Eccentricity","Handling","Kicking",
    "One on Ones","Passing","Punching (Tendency)","Reflexes","Rushing Out (Tendency)","Throwing"
]

# Exact weights transcribed from the provided image.
# Each dictionary maps attribute -> weight for that role (0 if not relevant).
WEIGHTS_BY_ROLE = {
    "GK": {
        # Technical
        "Corners":0.0,"Crossing":0.0,"Dribbling":0.0,"Finishing":0.0,"First Touch":0.0,"Free Kick Taking":0.0,
        "Heading":1.0,"Long Shots":0.0,"Long Throws":0.0,"Marking":0.0,"Passing":0.0,"Penalty Taking":0.0,"Tackling":0.0,"Technique":1.0,
        # Mental
        "Aggression":0.0,"Anticipation":3.0,"Bravery":6.0,"Composure":2.0,"Concentration":6.0,"Decisions":10.0,"Determination":0.0,
        "Flair":0.0,"Leadership":2.0,"Off The Ball":0.0,"Positioning":5.0,"Teamwork":2.0,"Vision":1.0,"Work Rate":1.0,
        # Physical
        "Acceleration":6.0,"Agility":8.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":3.0,"Stamina":1.0,"Strength":4.0,
        "Weaker Foot":3.0,
        # Goalkeeping specific
        "Aerial Reach":6.0,"Command of Area":6.0,"Communication":5.0,"Eccentricity":0.0,"Handling":8.0,"Kicking":5.0,
        "One on Ones":4.0,"Punching (Tendency)":0.0,"Reflexes":8.0,"Rushing Out (Tendency)":0.0,"Throwing":3.0
    },
    "DRL": {
        # Values from image (DRL column)
        "Corners":1.0,"Crossing":2.0,"Dribbling":2.0,"Finishing":1.0,"First Touch":3.0,"Free Kick Taking":1.0,
        "Heading":2.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":3.0,"Passing":2.0,"Penalty Taking":1.0,"Tackling":4.0,"Technique":2.0,
        "Aggression":0.0,"Anticipation":3.0,"Bravery":2.0,"Composure":2.0,"Concentration":4.0,"Decisions":7.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":1.0,"Positioning":4.0,"Teamwork":2.0,"Vision":2.0,"Work Rate":2.0,
        "Acceleration":7.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":2.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":6.0,"Strength":4.0,
        "Weaker Foot":4.0,
        # GK extras are 0
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "DC": {
        "Corners":1.0,"Crossing":1.0,"Dribbling":1.0,"Finishing":1.0,"First Touch":2.0,"Free Kick Taking":1.0,
        "Heading":5.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":8.0,"Passing":2.0,"Penalty Taking":1.0,"Tackling":5.0,"Technique":1.0,
        "Aggression":0.0,"Anticipation":5.0,"Bravery":2.0,"Composure":2.0,"Concentration":4.0,"Decisions":10.0,"Determination":0.0,
        "Flair":0.0,"Leadership":2.0,"Off The Ball":1.0,"Positioning":8.0,"Teamwork":1.0,"Vision":1.0,"Work Rate":2.0,
        "Acceleration":6.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":6.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":3.0,"Strength":6.0,
        "Weaker Foot":4.5,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "WBRL": {
        "Corners":1.0,"Crossing":3.0,"Dribbling":2.0,"Finishing":1.0,"First Touch":3.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":2.0,"Passing":3.0,"Penalty Taking":1.0,"Tackling":3.0,"Technique":3.0,
        "Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":2.0,"Concentration":3.0,"Decisions":5.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,"Positioning":3.0,"Teamwork":2.0,"Vision":1.0,"Work Rate":2.0,
        "Acceleration":8.0,"Agility":5.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":6.0,"Stamina":7.0,"Strength":4.0,
        "Weaker Foot":4.0,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "DM": {
        "Corners":1.0,"Crossing":1.0,"Dribbling":2.0,"Finishing":2.0,"First Touch":4.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":3.0,"Passing":4.0,"Penalty Taking":1.0,"Tackling":7.0,"Technique":3.0,
        "Aggression":0.0,"Anticipation":5.0,"Bravery":1.0,"Composure":2.0,"Concentration":3.0,"Decisions":8.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,"Positioning":5.0,"Teamwork":2.0,"Vision":4.0,"Work Rate":4.0,
        "Acceleration":6.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":4.0,"Stamina":4.0,"Strength":5.0,
        "Weaker Foot":5.0,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "MRL": {
        "Corners":1.0,"Crossing":5.0,"Dribbling":3.0,"Finishing":2.0,"First Touch":4.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":3.0,"Penalty Taking":1.0,"Tackling":2.0,"Technique":4.0,
        "Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,"Concentration":2.0,"Decisions":5.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":3.0,"Positioning":1.0,"Teamwork":2.0,"Vision":3.0,"Work Rate":3.0,
        "Acceleration":8.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":6.0,"Stamina":5.0,"Strength":3.0,
        "Weaker Foot":5.0,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "MC": {
        "Corners":1.0,"Crossing":1.0,"Dribbling":2.0,"Finishing":2.0,"First Touch":6.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":3.0,"Passing":6.0,"Penalty Taking":1.0,"Tackling":3.0,"Technique":4.0,
        "Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,"Concentration":2.0,"Decisions":7.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,"Positioning":3.0,"Teamwork":2.0,"Vision":6.0,"Work Rate":3.0,
        "Acceleration":6.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":6.0,"Strength":4.0,
        "Weaker Foot":5.0,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "AMRL": {
        "Corners":1.0,"Crossing":5.0,"Dribbling":5.0,"Finishing":2.0,"First Touch":5.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":2.0,"Penalty Taking":1.0,"Tackling":2.0,"Technique":4.0,
        "Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,"Concentration":2.0,"Decisions":5.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":3.0,"Positioning":1.0,"Teamwork":2.0,"Vision":3.0,"Work Rate":3.0,
        "Acceleration":10.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":10.0,"Stamina":7.0,"Strength":3.0,
        "Weaker Foot":6.0,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "AMC": {
        "Corners":1.0,"Crossing":1.0,"Dribbling":3.0,"Finishing":3.0,"First Touch":5.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":1.0,"Passing":4.0,"Penalty Taking":1.0,"Tackling":2.0,"Technique":5.0,
        "Aggression":0.0,"Anticipipation":0.0, # small typo guard: not used
        "Anticipation":3.0,"Bravery":1.0,"Composure":3.0,"Concentration":2.0,"Decisions":6.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,"Positioning":2.0,"Teamwork":2.0,"Vision":6.0,"Work Rate":3.0,
        "Acceleration":9.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":7.0,"Stamina":6.0,"Strength":3.0,
        "Weaker Foot":5.5,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "SC": {
        "Corners":1.0,"Crossing":2.0,"Dribbling":5.0,"Finishing":8.0,"First Touch":6.0,"Free Kick Taking":1.0,
        "Heading":6.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":2.0,"Penalty Taking":1.0,"Tackling":1.0,"Technique":4.0,
        "Aggression":0.0,"Anticipation":5.0,"Bravery":1.0,"Composure":6.0,"Concentration":2.0,"Decisions":5.0,"Determination":0.0,
        "Flair":0.0,"Leadership":1.0,"Off The Ball":6.0,"Positioning":2.0,"Teamwork":1.0,"Vision":2.0,"Work Rate":2.0,
        "Acceleration":10.0,"Agility":6.0,"Balance":2.0,"Jumping Reach":5.0,"Natural Fitness":0.0,"Pace":7.0,"Stamina":6.0,"Strength":6.0,
        "Weaker Foot":7.5,
        "Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,"Handling":0.0,"Kicking":0.0,
        "One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,"Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
}

# Parser function (same heuristic approach as before)
def parse_players_from_text(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    header_cols = None
    attr_set = set([a.lower() for a in ATTRIBUTES if a != "Name"])
    for i, ln in enumerate(lines):
        parts = re.split(r"\t| {2,}|,|\s{2,}", ln)
        match_count = sum(1 for p in parts if p.strip().lower() in attr_set)
        if match_count >= 6:
            header_idx = i
            header_cols = [p.strip() for p in parts if p.strip()]
            break
    if header_idx is None:
        for i, ln in enumerate(lines[:40]):
            if "Finishing" in ln or "Dribbling" in ln or "Passing" in ln:
                parts = re.split(r"\t| {2,}|,|\s{2,}", ln)
                header_idx = i
                header_cols = [p.strip() for p in parts if p.strip()]
                break
    if header_idx is None:
        return None, "Could not find header/attribute line automatically."

    header_lower = [h.lower() for h in header_cols]
    cols = []
    for a in ATTRIBUTES:
        if a.lower() in header_lower:
            cols.append(a)
    if "Name" not in cols:
        cols = ["Name"] + cols

    rows = []
    for ln in lines[header_idx+1:]:
        parts = re.split(r"\t| {2,}|,|\s{2,}", ln)
        parts = [p for p in parts if p != ""]
        numeric_tokens = [p for p in parts if re.match(r'^-?\d+(\.\d+)?$', p)]
        if len(numeric_tokens) >= max(3, len(cols)-1):
            n_attr = len(cols)-1
            last_parts = parts[-n_attr:]
            maybe_attrs = []
            valid_attr_count = 0
            for p in last_parts:
                if re.match(r'^-?\d+(\.\d+)?$', p):
                    maybe_attrs.append(float(p))
                    valid_attr_count += 1
                else:
                    maybe_attrs.append(np.nan)
            if valid_attr_count >= max(1, n_attr//3):
                name_parts = parts[:-n_attr] or [parts[0]]
                name = " ".join(name_parts)
                row = {"Name": name}
                for i, col in enumerate(cols[1:]):
                    row[col] = maybe_attrs[i] if i < len(maybe_attrs) else np.nan
                rows.append(row)
            else:
                continue
        else:
            continue
    if not rows:
        return None, "Found header but no data rows detected."

    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    for c in df.columns:
        if c != "Name":
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df, None

if uploaded:
    raw = uploaded.read()
    try:
        text = rtf_to_text(raw.decode('utf-8', errors='ignore'))
    except Exception:
        text = rtf_to_text(raw)
    st.subheader("Extracted text preview (first 1000 chars)")
    st.code(text[:1000])

    with st.spinner("Parsing players and attributes..."):
        df, err = parse_players_from_text(text)
    if df is None:
        st.error(f"Automatic parsing failed: {err}")
        st.info("If automatic parsing failed, try a different RTF export or upload a CSV.")
        st.stop()

    st.success(f"Detected {len(df)} players and {len([c for c in df.columns if c!='Name'])} attributes.")
    st.dataframe(df.head(10))

    role_options = ["GK","DRL","DC","WBRL","DM","MRL","MC","AMRL","AMC","SC"]
    role = st.selectbox("Choose role to rank for (exact roles from image)", role_options, index=9)

    presets = WEIGHTS_BY_ROLE
    selected_weights = presets.get(role.upper(), {})
    available_attrs = [c for c in ATTRIBUTES if c in df.columns and c != "Name"]
    if not available_attrs:
        st.error("No matching attribute columns found in your RTF data. Parser may have missed the header.")
        st.stop()

    weights = {a: float(selected_weights.get(a, 0.0)) for a in available_attrs}

    st.subheader("Exact weights used for ranking (from image)")
    w_df = pd.DataFrame.from_dict(weights, orient='index', columns=["Weight"])
    st.dataframe(w_df.style.format("{:.2f}"))

    normalize = st.checkbox("Normalize attribute values (divide by max seen)", value=True)
    if normalize:
        max_val = st.number_input("Assumed max attribute value (e.g. 20)", value=20.0)
        attrs_df = df[available_attrs].fillna(0).astype(float) / float(max_val)
    else:
        attrs_df = df[available_attrs].fillna(0).astype(float)

    w_vec = np.array([weights[a] for a in available_attrs], dtype=float)
    scores = attrs_df.values.dot(w_vec)
    df_out = df.copy()
    df_out["Score"] = scores
    df_out_sorted = df_out.sort_values("Score", ascending=False).reset_index(drop=True)

    st.subheader(f"Top players for role: {role}")
    st.dataframe(df_out_sorted[["Name","Score"] + available_attrs].head(50))

    st.download_button("Download ranked CSV", df_out_sorted.to_csv(index=False).encode('utf-8'),
                       file_name=f"players_ranked_{role}.csv")
