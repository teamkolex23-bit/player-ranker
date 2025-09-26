import re
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(layout="wide", page_title="Auto Player Ranker (HTML)")
st.title("Auto Player Ranker — upload an HTML table and pick a role to rank")
st.markdown("Upload a single HTML file containing a single players table (header row + data rows). Columns must match common Football Manager abbreviations (e.g. `Name`, `Position`, `Age`, `Transfer Value`, `Pac`, `Acc`, `Dri`, ...).")

CANONICAL_ATTRIBUTES = [
    "Corners","Crossing","Dribbling","Finishing","First Touch","Free Kick Taking","Heading",
    "Long Shots","Long Throws","Marking","Passing","Penalty Taking","Tackling","Technique",
    "Aggression","Anticipation","Bravery","Composure","Concentration","Decisions","Determination",
    "Flair","Leadership","Off The Ball","Positioning","Teamwork","Vision","Work Rate",
    "Acceleration","Agility","Balance","Jumping Reach","Natural Fitness","Pace","Stamina","Strength",
    "Weaker Foot","Aerial Reach","Command of Area","Communication","Eccentricity","Handling","Kicking",
    "One on Ones","Punching (Tendency)","Reflexes","Rushing Out (Tendency)","Throwing","Long Shots"
]

WEIGHTS_BY_ROLE = {
    "GK":{
        "Decisions":10,"Agility":8,"Reflexes":8,"Handling":8,"Concentration":6,"Bravery":6,
        "Acceleration":6,"Command of Area":6,"Aerial Reach":6,"Positioning":5,"Kicking":5,
        "Communication":5,"Strength":4,"One on Ones":4,"Pace":3
    },
    "DL/DR":{
        "Acceleration":7,"Decisions":7,"Agility":6,"Stamina":6,"Pace":5,"Concentration":4,
        "Strength":4,"Positioning":4,"Tackling":4,"First Touch":3,"Anticipation":3,"Marking":3,
        "Composure":2,"Bravery":2,"Balance":2
    },
    "WBL/WBR":{
        "Acceleration":8,"Stamina":7,"Pace":6,"Agility":5,"Decisions":5,"Strength":4,
        "Concentration":3,"Technique":3,"First Touch":3,"Positioning":3,"Anticipation":3,
        "Tackling":3,"Passing":3,"Crossing":3,"Composure":2
    },
    "ML/MR":{
        "Acceleration":8,"Agility":6,"Pace":6,"Stamina":5,"Decisions":5,"Crossing":5,
        "Technique":4,"First Touch":4,"Composure":3,"Strength":3,"Work Rate":3,
        "Anticipation":3,"Vision":3,"Passing":3,"Dribbling":3
    },
    "AML/AMR":{
        "Pace":10,"Acceleration":10,"Stamina":7,"Agility":6,"First Touch":5,"Decisions":5,
        "Dribbling":5,"Crossing":5,"Technique":4,"Composure":3,"Strength":3,"Work Rate":3,
        "Anticipation":3,"Vision":3,"Concentration":2
    },
    "CB":{
        "Decisions":10,"Positioning":8,"Marking":8,"Agility":6,"Jumping Reach":6,"Strength":6,
        "Acceleration":6,"Pace":5,"Anticipation":5,"Tackling":5,"Heading":5,"Concentration":4,
        "Stamina":3,"Composure":2,"Bravery":2
    },
    "DM":{
        "Decisions":8,"Tackling":7,"Agility":6,"Acceleration":6,"Strength":5,"Positioning":5,
        "Anticipation":5,"Pace":4,"Stamina":4,"Work Rate":4,"First Touch":4,"Vision":4,
        "Passing":4,"Concentration":3,"Technique":3
    },
    "CM":{
        "Decisions":7,"Agility":6,"Stamina":6,"Acceleration":6,"First Touch":6,"Vision":6,
        "Passing":6,"Pace":5,"Strength":4,"Technique":4,"Composure":3,"Work Rate":3,
        "Positioning":3,"Anticipation":3,"Tackling":3
    },
    "AMC":{
        "Acceleration":9,"Pace":7,"Agility":6,"Stamina":6,"Decisions":6,"Vision":6,
        "Technique":5,"First Touch":5,"Passing":4,"Composure":3,"Strength":3,"Work Rate":3,
        "Anticipation":3,"Off The Ball":3,"Long Shots":3
    },
    "ST":{
        "Acceleration":10,"Finishing":8,"Pace":7,"Composure":6,"Agility":6,"Stamina":6,
        "Strength":6,"First Touch":6,"Off The Ball":6,"Heading":6,"Jumping Reach":5,
        "Decisions":5,"Anticipation":5,"Dribbling":5,"Technique":4
    }
}

ABBR_MAP = {
    "Name":"Name","Position":"Position","Inf":"Inf","Age":"Age","Transfer Value":"Transfer Value",
    "Cor":"Corners","Cro":"Crossing","Dri":"Dribbling","Fin":"Finishing","Fir":"First Touch","Fre":"Free Kick Taking",
    "Hea":"Heading","Lon":"Long Shots","L Th":"Long Throws","LTh":"Long Throws","Mar":"Marking","Pas":"Passing","Pen":"Penalty Taking",
    "Tck":"Tackling","Tec":"Technique","Agg":"Aggression","Ant":"Anticipation","Bra":"Bravery","Cmp":"Composure","Cnt":"Concentration",
    "Dec":"Decisions","Det":"Determination","Fla":"Flair","Ldr":"Leadership","OtB":"Off The Ball","Pos":"Positioning","Tea":"Teamwork","Vis":"Vision","Wor":"Work Rate",
    "Acc":"Acceleration","Agi":"Agility","Bal":"Balance","Jum":"Jumping Reach","Nat":"Natural Fitness","Pac":"Pace","Sta":"Stamina","Str":"Strength",
    "Weaker Foot":"Weaker Foot","Aer":"Aerial Reach","Cmd":"Command of Area","Com":"Communication","Ecc":"Eccentricity","Han":"Handling","Kic":"Kicking",
    "1v1":"One on Ones","Pun":"Punching (Tendency)","Ref":"Reflexes","TRO":"Rushing Out (Tendency)","Thr":"Throwing"
}


def parse_players_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, "No <table> found in HTML."

    # header
    header_row = table.find("tr")
    if header_row is None:
        return None, "No rows in table."
    ths = header_row.find_all(["th","td"])  # first row may use td
    header_cells = [th.get_text(strip=True) for th in ths]
    canonical = [ABBR_MAP.get(h, h) for h in header_cells]

    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
        if not cols or all(not c for c in cols):
            continue
        if len(cols) < len(canonical):
            cols += [""]*(len(canonical)-len(cols))
        cols = cols[:len(canonical)]
        row = {col_name: val for col_name,val in zip(canonical, cols) if col_name}
        name = (row.get("Name") or "").strip()
        if not name or name.lower() == "name":
            continue
        rows.append(row)

    if not rows:
        return None, "No data rows parsed from HTML table."

    df = pd.DataFrame(rows)

    # coerce numeric columns where appropriate
    for c in df.columns:
        if c in ("Name","Position","Transfer Value","Inf"):
            continue
        # extract first numeric token
        df[c] = df[c].astype(str).str.extract(r'(-?\d+(?:\.\d+)?)')[0]
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Age convert if exists
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors='coerce').astype('Int64')

    return df, None


uploaded = st.file_uploader("Upload your players HTML file", type=["html","htm"]) 
if not uploaded:
    st.info("Upload an HTML file (export from FM as 'export to clipboard' -> save as .html). This app only accepts HTML tables.")
    st.stop()

raw = uploaded.read()
try:
    html_text = raw.decode('utf-8', errors='ignore')
except Exception:
    html_text = raw.decode('latin-1', errors='ignore')

df, err = parse_players_from_html(html_text)
if df is None:
    st.error(f"Parsing failed: {err}")
    st.stop()

st.success(f"Parsed {len(df)} players.")

ROLE_OPTIONS = list(WEIGHTS_BY_ROLE.keys())
role = st.selectbox("Choose role to rank for", ROLE_OPTIONS, index=ROLE_OPTIONS.index("ST") if "ST" in ROLE_OPTIONS else 0)

selected_weights = WEIGHTS_BY_ROLE.get(role, {})

# choose attributes that are both in canonical list and in parsed columns
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]
if not available_attrs:
    st.error("No matching attribute columns found in the uploaded table. Detected columns: " + ", ".join(list(df.columns)))
    st.stop()

normalize = st.checkbox("Normalize attribute values (divide by max)", value=True)
max_val = 20.0
if normalize:
    max_val = st.number_input("Assumed max attribute value (e.g. 20)", value=20.0, min_value=1.0)

attrs_df = df[available_attrs].fillna(0).astype(float)
if normalize:
    attrs_norm = attrs_df / float(max_val)
else:
    attrs_norm = attrs_df

weights = pd.Series({a: float(selected_weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)

scores = attrs_norm.values.dot(weights.values.astype(float))

df_out = df.copy()
df_out["Score"] = scores

df_out_sorted = df_out.sort_values("Score", ascending=False).reset_index(drop=True)

cols_to_show = [c for c in ["Name","Position","Age","Transfer Value","Score"] if c in df_out_sorted.columns]
# show top 200 by score
st.subheader(f"Top players for role: {role} (sorted by Score)")
st.dataframe(df_out_sorted[cols_to_show + [c for c in available_attrs if c in df_out_sorted.columns]].head(200))

csv_bytes = df_out_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download ranked CSV (full)", csv_bytes, file_name=f"players_ranked_{role}.csv")

st.markdown("---")
with st.expander("About normalization"):
    st.write("Normalization divides attribute values by an assumed maximum (e.g. 20). This turns raw attribute scores into a 0..1 range so weights act proportionally across attributes. You can change the assumed max if your data uses a different scale (e.g. 10 or 100). Changing it rescales the influence of attributes relative to their weights.")
