import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(layout="wide", page_title="FM24 Player Ranker")
st.title("FM24 Player Ranker")

CANONICAL_ATTRIBUTES = [
    "Corners", "Crossing", "Dribbling", "Finishing", "First Touch", "Free Kick Taking", "Heading",
    "Long Shots", "Long Throws", "Marking", "Passing", "Penalty Taking", "Tackling", "Technique",
    "Aggression", "Anticipation", "Bravery", "Composure", "Concentration", "Decisions", "Determination",
    "Flair", "Leadership", "Off The Ball", "Positioning", "Teamwork", "Vision", "Work Rate",
    "Acceleration", "Agility", "Balance", "Jumping Reach", "Natural Fitness", "Pace", "Stamina", "Strength",
    "Weaker Foot", "Aerial Reach", "Command of Area", "Communication", "Eccentricity", "Handling", "Kicking",
    "One on Ones", "Punching (Tendency)", "Reflexes", "Rushing Out (Tendency)", "Throwing"
]

WEIGHTS_BY_ROLE = {
    "GK":{
        "Corners":0.0,"Crossing":0.0,"Dribbling":0.0,"Finishing":0.0,"First Touch":0.0,"Free Kick Taking":0.0,
        "Heading":1.0,"Long Shots":0.0,"Long Throws":0.0,"Marking":0.0,"Passing":0.0,"Penalty Taking":0.0,
        "Tackling":0.0,"Technique":1.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":6.0,"Composure":2.0,
        "Concentration":6.0,"Decisions":10.0,"Determination":0.0,"Flair":0.0,"Leadership":2.0,"Off The Ball":0.0,
        "Positioning":5.0,"Teamwork":2.0,"Vision":1.0,"Work Rate":1.0,"Acceleration":6.0,"Agility":8.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":3.0,"Stamina":1.0,"Strength":4.0,
        "Weaker Foot":3.0,"Aerial Reach":6.0,"Command of Area":6.0,"Communication":5.0,"Eccentricity":0.0,
        "Handling":8.0,"Kicking":5.0,"One on Ones":4.0,"Punching (Tendency)":0.0,"Reflexes":8.0,
        "Rushing Out (Tendency)":0.0,"Throwing":3.0
    },
    "DL/DR":{
        "Corners":1.0,"Crossing":2.0,"Dribbling":1.0,"Finishing":1.0,"First Touch":3.0,"Free Kick Taking":1.0,
        "Heading":2.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":3.0,"Passing":2.0,"Penalty Taking":1.0,
        "Tackling":4.0,"Technique":2.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":2.0,"Composure":2.0,
        "Concentration":4.0,"Decisions":7.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":1.0,
        "Positioning":4.0,"Teamwork":2.0,"Vision":2.0,"Work Rate":2.0,"Acceleration":7.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":2.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":6.0,"Strength":4.0,
        "Weaker Foot":4.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "CB":{
        "Corners":1.0,"Crossing":1.0,"Dribbling":1.0,"Finishing":1.0,"First Touch":2.0,"Free Kick Taking":1.0,
        "Heading":5.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":8.0,"Passing":2.0,"Penalty Taking":1.0,
        "Tackling":5.0,"Technique":1.0,"Aggression":0.0,"Anticipation":5.0,"Bravery":2.0,"Composure":2.0,
        "Concentration":4.0,"Decisions":10.0,"Determination":0.0,"Flair":0.0,"Leadership":2.0,"Off The Ball":1.0,
        "Positioning":8.0,"Teamwork":1.0,"Vision":1.0,"Work Rate":2.0,"Acceleration":6.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":6.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":3.0,"Strength":6.0,
        "Weaker Foot":4.5,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "WBL/WBR":{
        "Corners":1.0,"Crossing":3.0,"Dribbling":2.0,"Finishing":1.0,"First Touch":3.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":1.0,"Long Throws":1.0,"Marking":2.0,"Passing":3.0,"Penalty Taking":1.0,
        "Tackling":3.0,"Technique":3.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":2.0,
        "Concentration":3.0,"Decisions":5.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,
        "Positioning":3.0,"Teamwork":2.0,"Vision":2.0,"Work Rate":2.0,"Acceleration":8.0,"Agility":5.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":6.0,"Stamina":7.0,"Strength":4.0,
        "Weaker Foot":4.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "DM":{
        "Corners":1.0,"Crossing":1.0,"Dribbling":2.0,"Finishing":2.0,"First Touch":4.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":3.0,"Passing":4.0,"Penalty Taking":1.0,
        "Tackling":7.0,"Technique":3.0,"Aggression":0.0,"Anticipation":5.0,"Bravery":1.0,"Composure":2.0,
        "Concentration":3.0,"Decisions":8.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":1.0,
        "Positioning":5.0,"Teamwork":2.0,"Vision":4.0,"Work Rate":4.0,"Acceleration":6.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":4.0,"Stamina":4.0,"Strength":5.0,
        "Weaker Foot":5.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "ML/MR":{
        "Corners":1.0,"Crossing":5.0,"Dribbling":3.0,"Finishing":2.0,"First Touch":4.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":3.0,"Penalty Taking":1.0,
        "Tackling":2.0,"Technique":4.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":2.0,
        "Concentration":2.0,"Decisions":5.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,
        "Positioning":1.0,"Teamwork":2.0,"Vision":3.0,"Work Rate":3.0,"Acceleration":8.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":6.0,"Stamina":5.0,"Strength":3.0,
        "Weaker Foot":5.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "CM":{
        "Corners":1.0,"Crossing":1.0,"Dribbling":2.0,"Finishing":2.0,"First Touch":6.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":3.0,"Passing":6.0,"Penalty Taking":1.0,
        "Tackling":3.0,"Technique":4.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,
        "Concentration":2.0,"Decisions":7.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":3.0,
        "Positioning":3.0,"Teamwork":2.0,"Vision":6.0,"Work Rate":3.0,"Acceleration":6.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":5.0,"Stamina":6.0,"Strength":4.0,
        "Weaker Foot":6.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "AML/AMR":{
        "Corners":1.0,"Crossing":5.0,"Dribbling":5.0,"Finishing":2.0,"First Touch":5.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":2.0,"Penalty Taking":1.0,
        "Tackling":2.0,"Technique":4.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,
        "Concentration":2.0,"Decisions":5.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":2.0,
        "Positioning":1.0,"Teamwork":2.0,"Vision":3.0,"Work Rate":3.0,"Acceleration":10.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":10.0,"Stamina":7.0,"Strength":3.0,
        "Weaker Foot":5.5,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "AMC":{
        "Corners":1.0,"Crossing":1.0,"Dribbling":3.0,"Finishing":3.0,"First Touch":5.0,"Free Kick Taking":1.0,
        "Heading":1.0,"Long Shots":3.0,"Long Throws":1.0,"Marking":1.0,"Passing":4.0,"Penalty Taking":1.0,
        "Tackling":2.0,"Technique":5.0,"Aggression":0.0,"Anticipation":3.0,"Bravery":1.0,"Composure":3.0,
        "Concentration":2.0,"Decisions":6.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":3.0,
        "Positioning":2.0,"Teamwork":2.0,"Vision":6.0,"Work Rate":3.0,"Acceleration":9.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":1.0,"Natural Fitness":0.0,"Pace":7.0,"Stamina":6.0,"Strength":3.0,
        "Weaker Foot":7.0,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
    },
    "ST":{
        "Corners":1.0,"Crossing":2.0,"Dribbling":5.0,"Finishing":8.0,"First Touch":6.0,"Free Kick Taking":1.0,
        "Heading":6.0,"Long Shots":2.0,"Long Throws":1.0,"Marking":1.0,"Passing":2.0,"Penalty Taking":1.0,
        "Tackling":1.0,"Technique":4.0,"Aggression":0.0,"Anticipation":5.0,"Bravery":1.0,"Composure":6.0,
        "Concentration":2.0,"Decisions":5.0,"Determination":0.0,"Flair":0.0,"Leadership":1.0,"Off The Ball":6.0,
        "Positioning":2.0,"Teamwork":1.0,"Vision":2.0,"Work Rate":2.0,"Acceleration":10.0,"Agility":6.0,
        "Balance":2.0,"Jumping Reach":5.0,"Natural Fitness":0.0,"Pace":7.0,"Stamina":6.0,"Strength":6.0,
        "Weaker Foot":7.5,"Aerial Reach":0.0,"Command of Area":0.0,"Communication":0.0,"Eccentricity":0.0,
        "Handling":0.0,"Kicking":0.0,"One on Ones":0.0,"Punching (Tendency)":0.0,"Reflexes":0.0,
        "Rushing Out (Tendency)":0.0,"Throwing":0.0
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


def parse_players_from_html(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if table is None:
        return None, "No <table> found in HTML."

    header_row = table.find("tr")
    if header_row is None:
        return None, "No rows in table."
    ths = header_row.find_all(["th", "td"])  # first row may use td
    header_cells = [th.get_text(strip=True) for th in ths]
    canonical = [ABBR_MAP.get(h, h) for h in header_cells]

    rows = []
    for tr in table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"]) ]
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

# Uploader with hover help (info)
uploader_help = (
    "Go to https://fmarenacalc.com, follow the exact instructions there, upload the html file on our website since their weight table is inaccurate"
)
file_label = "Upload your players HTML file"
uploaded = st.file_uploader(file_label, type=["html","htm"], help=uploader_help)
# small hoverable hint for drag-and-drop (rendered as HTML)
st.markdown(
    '<div style="font-size:12px"> <span title="Maximum of 260 players can be loaded so limit your search in FM to narrow it down. I personally segment it into age groups and personality/media handling style but you can also block certain regions or divisions, like only having top 5 from the 2nd division up etc">🛈 Drag & drop tips</span></div>',
    unsafe_allow_html=True
)
if not uploaded:
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

# merge duplicate columns and reset index for simple indexing
df = merge_duplicate_columns(df)
df = df.reset_index(drop=True)

# determine available attributes present in the upload
available_attrs = [a for a in CANONICAL_ATTRIBUTES if a in df.columns]
if not available_attrs:
    st.error("No matching attribute columns found in the uploaded table. Detected columns: " + ", ".join(list(df.columns)))
    st.stop()

# normalization control with hoverable help directly on the checkbox
norm_help = (
    "Normalization divides attribute values by an assumed maximum (e.g. 20), "
    "turning raw attribute scores into a 0..1 range so weights act proportionally. "
    "If your attributes use a different top value (e.g. 10), change the assumed max to rescale attributes."
)
normalize = st.checkbox("Normalize attribute values (divide by max) 🛈", value=True, help=norm_help)
max_val = 20.0
if normalize:
    max_val = st.number_input("Assumed max attribute value (e.g. 20)", value=20.0, min_value=1.0)

attrs_df = df[available_attrs].fillna(0).astype(float)
attrs_norm = attrs_df / float(max_val) if normalize else attrs_df

# compute role scores for the currently selected role (single select box)
ROLE_OPTIONS = list(WEIGHTS_BY_ROLE.keys())
role = st.selectbox("Choose role to rank for", ROLE_OPTIONS, index=ROLE_OPTIONS.index("ST") if "ST" in ROLE_OPTIONS else 0)
selected_weights = WEIGHTS_BY_ROLE.get(role, {})
weights = pd.Series({a: float(selected_weights.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)
current_scores = attrs_norm.values.dot(weights.values.astype(float))

# create main ranked dataframe
df_out = df.copy()
df_out["Score"] = current_scores
df_out_sorted = df_out.sort_values("Score", ascending=False).reset_index(drop=True)
# ranking starts at 1
ranked = df_out_sorted.copy()
ranked.insert(0, "Rank", range(1, len(ranked) + 1))

# show selected-role top list
cols_to_show = [c for c in ["Rank","Name","Position","Age","Transfer Value","Score"] if c in ranked.columns]
st.subheader(f"Top players for role: {role} (sorted by Score)")
st.dataframe(ranked[cols_to_show + [c for c in available_attrs if c in ranked.columns]].head(200))

# additional compact top-10 per role
st.markdown("---")
st.subheader("Top 10 — every role (compact)")
per_row = 4
roles = ROLE_OPTIONS
for i in range(0, len(roles), per_row):
    cols = st.columns(per_row)
    for j, r in enumerate(roles[i:i+per_row]):
        with cols[j]:
            rw = WEIGHTS_BY_ROLE.get(r, {})
            w = pd.Series({a: float(rw.get(a, 0.0)) for a in available_attrs}).reindex(available_attrs).fillna(0.0)
            sc = attrs_norm.values.dot(w.values.astype(float))
            tmp = df.copy()
            tmp["Score"] = sc
            tmp_sorted = tmp.sort_values("Score", ascending=False).reset_index(drop=True).head(10)
            tmp_sorted = tmp_sorted.reset_index(drop=True)
            tmp_sorted.insert(0, "Rank", range(1, len(tmp_sorted) + 1))
            tiny = tmp_sorted[[c for c in ["Rank","Name","Age","Transfer Value","Score"] if c in tmp_sorted.columns]].copy()
            tiny["Score"] = tiny["Score"].round(0).astype('Int64')
            st.markdown(f"**{r}**")
            st.table(tiny)

# Starting XI selection (optimal assignment per formation)
st.markdown("---")
st.subheader("Best Starting XIs")

# formation mapping: position label -> role key
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

n_players = len(df)
n_positions = len(positions)
player_names = df["Name"].astype(str).tolist()

# precompute role weight vectors aligned with available_attrs
role_weight_vectors = {}
for _, role_key in positions:
    rw = WEIGHTS_BY_ROLE.get(role_key, {})
    role_weight_vectors[role_key] = np.array([float(rw.get(a, 0.0)) for a in available_attrs], dtype=float)

# compute score matrix players x positions
score_matrix = np.zeros((n_players, n_positions), dtype=float)
for i_idx in range(n_players):
    player_attr_vals = attrs_norm.iloc[i_idx].values if len(available_attrs) > 0 else np.zeros((len(available_attrs),), dtype=float)
    for p_idx, (_, role_key) in enumerate(positions):
        w = role_weight_vectors[role_key]
        score_matrix[i_idx, p_idx] = float(np.dot(player_attr_vals, w))

# helper: find best role for each player across all defined roles
all_role_keys = list(WEIGHTS_BY_ROLE.keys())
all_role_vectors = {rk: np.array([float(WEIGHTS_BY_ROLE[rk].get(a, 0.0)) for a in available_attrs], dtype=float) for rk in all_role_keys}
player_best_role = []
for i_idx in range(n_players):
    player_attr_vals = attrs_norm.iloc[i_idx].values if len(available_attrs) > 0 else np.zeros((len(available_attrs),), dtype=float)
    best_score = -1e9
    best_role = None
    for rk, vec in all_role_vectors.items():
        sc = float(np.dot(player_attr_vals, vec))
        if sc > best_score:
            best_score = sc
            best_role = rk
    player_best_role.append((best_role, best_score))

# assignment helper using Hungarian (scipy)
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def choose_starting_xi(available_player_indices):
    m = max(len(available_player_indices), n_positions)
    cost = np.zeros((m, n_positions), dtype=float)
    # cost as negative scores for minimization
    if len(available_player_indices) > 0:
        cost[:len(available_player_indices), :] = -score_matrix[available_player_indices, :]
    # default zeros for dummy rows
    if linear_sum_assignment is None:
        # fallback greedy
        chosen = {}
        used_players = set()
        for p_idx in range(n_positions):
            best_p = None
            best_sc = -1e9
            for i_idx in available_player_indices:
                if i_idx in used_players:
                    continue
                sc = score_matrix[i_idx, p_idx]
                if sc > best_sc:
                    best_sc = sc
                    best_p = i_idx
            if best_p is not None:
                chosen[p_idx] = best_p
                used_players.add(best_p)
        return chosen
    else:
        row_ind, col_ind = linear_sum_assignment(cost)
        chosen = {}
        for r, c in zip(row_ind, col_ind):
            if r < len(available_player_indices) and c < n_positions:
                chosen[c] = available_player_indices[r]
        return chosen

# first XI
all_player_indices = list(range(n_players))
first_choice = choose_starting_xi(all_player_indices)

# compute display for a chosen XI
def render_xi(chosen_map):
    rows = []
    sel_scores = []
    for pos_idx, (pos_label, role_key) in enumerate(positions):
        if pos_idx in chosen_map:
            p_idx = chosen_map[pos_idx]
            name = player_names[p_idx]
            sel_score = score_matrix[p_idx, pos_idx]
            best_role, best_score = player_best_role[p_idx]
            rows.append((pos_label, name, sel_score, best_role, best_score, p_idx))
            sel_scores.append(sel_score)
        else:
            rows.append((pos_label, "", 0.0, "", 0.0, None))
    team_total = float(sum([r[2] for r in rows if r[5] is not None]))
    placed_scores = [r[2] for r in rows if r[5] is not None]
    team_avg = float(np.mean(placed_scores)) if placed_scores else 0.0

    def color_for_diff(diff):
        cap = 400.0
        ratio = max(-1.0, min(1.0, diff / cap))
        if ratio > 0:
            g = int(55 + 200 * ratio)
            return f"rgb(0,{g},0)"
        elif ratio < 0:
            r = int(55 + 200 * abs(ratio))
            return f"rgb({r},0,0)"
        else:
            return "#000000"

    lines = []
    for pos_label, name, sel_score, best_role, best_score, p_idx in rows:
        if name:
            diff = sel_score - team_avg
            color = color_for_diff(diff)
            name_html = f"<span style='color:{color}; font-weight:600'>{st.markdown.__self__ if False else name}</span>"
            # st.markdown.__self__ trick avoided; use safe name
            name_html = f"<span style='color:{color}; font-weight:600'>{name}</span>"
            lines.append(f"{pos_label} | {name_html} | {int(round(sel_score))} | {best_role} | {int(round(best_score))}")
        else:
            lines.append(f"{pos_label} |  |  |  | ")
            return "\n".join(lines), team_total

first_lines, first_total = render_xi(first_choice)

# second XI (exclude players used in first)
used_player_indices = set(first_choice.values())
remaining_players = [i for i in all_player_indices if i not in used_player_indices]
second_choice = choose_starting_xi(remaining_players)
# if returned indices refer to available_player_indices ordering, ensure mapping uses those original indices
# our function already maps to original indices when using scipy; when using greedy it also returns original indices.
second_lines, second_total = render_xi(second_choice)

st.markdown("**First Starting XI**")
st.markdown(first_lines, unsafe_allow_html=True)
st.markdown(f"**Team total score = {int(round(first_total))}**")

st.markdown("---")
st.markdown("**Second Starting XI**")
st.markdown(second_lines, unsafe_allow_html=True)
st.markdown(f"**Team total score = {int(round(second_total))}**")

# final download
csv_bytes = df_out_sorted.to_csv(index=False).encode("utf-8")
st.download_button("Download ranked CSV (full)", csv_bytes, file_name=f"players_ranked_{role}.csv")


