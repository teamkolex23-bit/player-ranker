# app.py
# Streamlit app: accepts .html or .rtf, parses player table, ranks players using exact weights.
# Requirements: streamlit pandas numpy striprtf beautifulsoup4

import re
import numpy as np
import pandas as pd
import streamlit as st
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup

st.set_page_config(layout="wide", page_title="Auto Player Ranker (HTML+RTF)")
st.title("Auto Player Ranker — upload .html or .rtf, pick a role, get ranked players")
st.markdown("Preferred: upload HTML (table). RTF still supported but HTML is more reliable.")

# -------------------------
# Weights & attributes (exact chart)
# -------------------------
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
    # (truncated here for readability; same full dict as your working app — ensure you paste full dict)
    "SC": {
        "Corners":1,"Crossing":2,"Dribbling":5,"Finishing":8,"First Touch":6,"Free Kick Taking":1,"Heading":6,
        "Long Shots":2,"Long Throws":1,"Marking":1,"Passing":2,"Penalty Taking":1,"Tackling":1,"Technique":4,
        "Anticipation":5,"Composure":6,"Concentration":2,"Decisions":5,"Off The Ball":6,"Positioning":2,"Teamwork":1,"Vision":2,"Work Rate":2,
        "Acceleration":10,"Agility":6,"Balance":2,"Jumping Reach":5,"Pace":7,"Stamina":6,"Strength":6,"Weaker Foot":7.5
    },
    # add other roles here exactly as in your app (GK, DRL, DC, WBRL, DM, MRL, MC, AMRL, AMC)
}

# If your WEIGHTS_BY_ROLE in code has the full roles, include them all.
# -------------------------
# Abbreviation mapping (HTML/RTF headers -> canonical names)
# -------------------------
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

# -------------------------
# HTML parser (reliable)
# -------------------------
def parse_players_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, "No <table> found in HTML."
    # Get header cells (first row with th or first tr)
    header_cells = []
    header_row = table.find("tr")
    if header_row is None:
        return None, "No rows in table."
    # prefer <th>, fallback to first tr's <td>
    ths = header_row.find_all("th")
    if ths:
        header_cells = [th.get_text(strip=True) for th in ths]
        data_start = 1
    else:
        tds = header_row.find_all("td")
        if tds:
            header_cells = [td.get_text(strip=True) for td in tds]
            data_start = 1
        else:
            return None, "Header row found but no cells."
    # map headers
    canonical = [ABBR_MAP.get(h, h) for h in header_cells]
    # parse data rows (tr after header)
    rows = []
    for tr in table.find_all("tr")[data_start:]:
        cols = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
        # skip empty rows
        if not cols or all(not c for c in cols):
            continue
        # align length
        if len(cols) < len(canonical):
            cols += [""]*(len(canonical)-len(cols))
        cols = cols[:len(canonical)]
        row = {col_name: val for col_name,val in zip(canonical, cols) if col_name}
        # require a Name
        name = (row.get("Name") or "").strip()
        if not name or name.lower() == "name":
            continue
        rows.append(row)
    if not rows:
        return None, "No data rows parsed from HTML table."
    df = pd.DataFrame(rows)
    # coerce numeric-like columns
    for c in df.columns:
        if c in ("Name","Position","Transfer Value"):
            continue
        df[c] = df[c].astype(str).str.extract(r'(-?\d+(\.\d+)?)')[0].astype(float)
    return df, None

# -------------------------
# RTF fallback parser (robust pipe '|' alignment)
# -------------------------
def parse_players_from_rtf_text(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = None
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
    canonical = [ABBR_MAP.get(h, h) if h else "" for h in hdr_parts]
    start_idx = lines.index(header)
    rows = []
    for ln in lines[start_idx+1: start_idx+5000]:
        if '|' not in ln:
            continue
        parts = [p.strip() for p in ln.split('|')]
        if all((not p) or set(p) <= set('- _.') for p in parts):
            continue
        if len(parts) != len(hdr_parts):
            # tolerant attempt: if a line wraps (fewer parts), try merging with next line(s)
            # find block of consecutive '|' lines until counts match header length
            # build chunk starting at current line
            chunk = parts.copy()
            j = lines.index(ln) + 1
            while len(chunk) < len(hdr_parts) and j < start_idx+5000:
                nxt = lines[j]
                if '|' in nxt:
                    more = [p.strip() for p in nxt.split('|')]
                    chunk += more
                j += 1
            if len(chunk) != len(hdr_parts):
                continue
            parts = chunk[:len(hdr_parts)]
        row = {}
        for col_name, val in zip(canonical, parts):
            if col_name == "":
                continue
            row[col_name] = val
        name = row.get("Name","").strip()
        if not name or name.lower() == "name":
            continue
        rows.append(row)
    if not rows:
        return None, "Found header but no data rows detected (alignment mismatch)."
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c in ("Name","Position","Transfer Value"):
            continue
        df[c] = df[c].astype(str).str.extract(r'(-?\d+(\.\d+)?)')[0].astype(float)
    return df, None

# -------------------------
# Main UI / file handling
# -------------------------
uploaded = st.file_uploader("Upload your `.html` or `.rtf` file", type=["html","htm","rtf"])

if not uploaded:
    st.info("Upload an HTML or RTF file. HTML works best (table parsing is robust).")
    st.stop()

filename = uploaded.name.lower()
raw = uploaded.read()

df = None
err = None

if filename.endswith(('.html','.htm')):
    # parse HTML
    try:
        html_text = raw.decode('utf-8', errors='ignore')
    except:
        html_text = raw.decode('latin-1', errors='ignore')
    df, err = parse_players_from_html(html_text)
else:
    # RTF: convert and parse
    try:
        # try decode then rtf->text
        try:
            text = rtf_to_text(raw.decode('utf-8', errors='ignore'))
        except Exception:
            text = rtf_to_text(raw)
    except Exception as e:
        err = f"Failed to convert RTF to text: {e}"
    if text and err is None:
        df, err = parse_players_from_rtf_text(text)

st.subheader("Extracted preview / parse status")
if df is None:
    st.error("Automatic parsing failed: " + str(err))
    st.info("If parsing fails try exporting as HTML (recommended) or paste the problematic file here and I'll adapt the parser.")
    st.code((raw.decode('utf-8', errors='ignore')[:1500] if isinstance(raw, (bytes, bytearray)) else str(raw)[:1500]))
    st.stop()

st.success(f"Detected {len(df)} players and {len([c for c in df.columns if c!='Name'])} columns/attributes.")
st.dataframe(df.head(10))

ROLE_OPTIONS = list(WEIGHTS_BY_ROLE.keys())
if not ROLE_OPTIONS:
    st.error("No roles present in WEIGHTS_BY_ROLE. Please ensure the full weight dict is included.")
    st.stop()

role = st.selectbox("Choose role to rank for", ROLE_OPTIONS, index=ROLE_OPTIONS.index("SC") if "SC" in ROLE_OPTIONS else 0)

selected_weights = WEIGHTS_BY_ROLE.get(role, {})
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]

if not available_attrs:
    st.error("No matching attributes found in parsed file. Columns detected: " + ", ".join(list(df.columns)[:60]))
    st.stop()

weights = pd.Series({a: float(selected_weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)

st.subheader("Weights used for ranking")
st.dataframe(weights.rename("Weight").to_frame())

normalize = st.checkbox("Normalize attribute values (divide by max)", value=True)
if normalize:
    max_val = st.number_input("Assumed max attribute value (e.g. 20)", value=20.0, min_value=1.0)
    attrs_df = df[available_attrs].fillna(0).astype(float) / float(max_val)
else:
    attrs_df = df[available_attrs].fillna(0).astype(float)

scores = attrs_df.values.dot(weights.values.astype(float))
df_out = df.copy()
df_out["Score"] = scores
df_out_sorted = df_out.sort_values("Score", ascending=False).reset_index(drop=True)

st.subheader(f"Top players for role: {role}")
cols_to_show = ["Name","Position","Score"] + available_attrs
st.dataframe(df_out_sorted[cols_to_show].head(200))

csv_bytes = df_out_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download ranked CSV (full)", csv_bytes, file_name=f"players_ranked_{role}.csv")
