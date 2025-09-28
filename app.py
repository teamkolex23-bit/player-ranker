import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import io
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Football Manager Player Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .formation-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .player-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
    }
    
    .score-excellent {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .score-poor {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .score-average {
        color: #666666;
        font-weight: bold;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2e8b57;
    }
    
    .stSelectbox > div > div {
        background-color: white;
    }
    
    .stFileUploader > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Position weights dictionary
POSITION_WEIGHTS = {
    'GK': {
        'Aerial Reach': 6.0, 'Command of Area': 6.0, 'Communication': 5.0, 'Eccentricity': 0.0,
        'First Touch': 1.0, 'Handling': 8.0, 'Kicking': 5.0, 'One on Ones': 4.0, 'Passing': 3.0,
        'Punching (Tendency)': 0.0, 'Reflexes': 8.0, 'Rushing Out (Tendency)': 0.0, 'Throwing': 3.0,
        'Anticipation': 3.0, 'Bravery': 6.0, 'Composure': 2.0, 'Concentration': 6.0, 'Decisions': 10.0,
        'Leadership': 2.0, 'Positioning': 5.0, 'Acceleration': 6.0, 'Agility': 8.0, 'Balance': 2.0,
        'Jumping Reach': 1.0, 'Pace': 3.0, 'Stamina': 1.0, 'Strength': 4.0, 'Weaker Foot': 3.0
    },
    'DL/DR': {
        'Corners': 0.0, 'Crossing': 2.0, 'Dribbling': 1.0, 'Finishing': 1.0, 'First Touch': 3.0,
        'Free Kick Taking': 1.0, 'Heading': 2.0, 'Long Shots': 1.0, 'Long Throws': 1.0, 'Marking': 3.0,
        'Passing': 2.0, 'Penalty Taking': 1.0, 'Tackling': 4.0, 'Technique': 1.0, 'Anticipation': 3.0,
        'Bravery': 2.0, 'Composure': 2.0, 'Concentration': 4.0, 'Decisions': 7.0, 'Leadership': 1.0,
        'Off The Ball': 1.0, 'Positioning': 4.0, 'Teamwork': 2.0, 'Vision': 2.0, 'Work Rate': 1.0,
        'Acceleration': 6.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 2.0, 'Pace': 3.0,
        'Stamina': 6.0, 'Strength': 4.0, 'Weaker Foot': 4.0
    },
    'CB': {
        'Corners': 0.0, 'Crossing': 1.0, 'Dribbling': 1.0, 'Finishing': 1.0, 'First Touch': 2.0,
        'Free Kick Taking': 1.0, 'Heading': 5.0, 'Long Shots': 1.0, 'Long Throws': 1.0, 'Marking': 8.0,
        'Passing': 2.0, 'Penalty Taking': 1.0, 'Tackling': 5.0, 'Technique': 1.0, 'Anticipation': 5.0,
        'Bravery': 2.0, 'Composure': 2.0, 'Concentration': 4.0, 'Decisions': 10.0, 'Leadership': 2.0,
        'Off The Ball': 1.0, 'Positioning': 8.0, 'Teamwork': 1.0, 'Vision': 1.0, 'Work Rate': 2.0,
        'Acceleration': 6.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 6.0, 'Pace': 5.0,
        'Stamina': 3.0, 'Strength': 6.0, 'Weaker Foot': 4.0
    },
    'WBL/WBR': {
        'Corners': 0.0, 'Crossing': 3.0, 'Dribbling': 2.0, 'Finishing': 2.0, 'First Touch': 3.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 3.0, 'Long Throws': 1.0, 'Marking': 3.0,
        'Passing': 4.0, 'Penalty Taking': 1.0, 'Tackling': 7.0, 'Technique': 3.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 2.0, 'Concentration': 3.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 2.0, 'Positioning': 5.0, 'Teamwork': 2.0, 'Vision': 4.0, 'Work Rate': 2.0,
        'Acceleration': 8.0, 'Agility': 5.0, 'Balance': 2.0, 'Jumping Reach': 1.0, 'Pace': 6.0,
        'Stamina': 7.0, 'Strength': 5.0, 'Weaker Foot': 5.0
    },
    'DM': {
        'Corners': 0.0, 'Crossing': 1.0, 'Dribbling': 3.0, 'Finishing': 2.0, 'First Touch': 4.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 2.0, 'Long Throws': 1.0, 'Marking': 1.0,
        'Passing': 3.0, 'Penalty Taking': 1.0, 'Tackling': 2.0, 'Technique': 3.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 2.0, 'Concentration': 2.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 2.0, 'Positioning': 1.0, 'Teamwork': 2.0, 'Vision': 3.0, 'Work Rate': 3.0,
        'Acceleration': 6.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 1.0, 'Pace': 6.0,
        'Stamina': 5.0, 'Strength': 3.0, 'Weaker Foot': 5.0
    },
    'ML/MR': {
        'Corners': 0.0, 'Crossing': 5.0, 'Dribbling': 2.0, 'Finishing': 2.0, 'First Touch': 4.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 3.0, 'Long Throws': 1.0, 'Marking': 1.0,
        'Passing': 6.0, 'Penalty Taking': 1.0, 'Tackling': 3.0, 'Technique': 4.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 2.0, 'Concentration': 2.0, 'Decisions': 7.0, 'Leadership': 1.0,
        'Off The Ball': 3.0, 'Positioning': 3.0, 'Teamwork': 2.0, 'Vision': 6.0, 'Work Rate': 3.0,
        'Acceleration': 8.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 1.0, 'Pace': 6.0,
        'Stamina': 6.0, 'Strength': 4.0, 'Weaker Foot': 5.0
    },
    'MC': {
        'Corners': 0.0, 'Crossing': 1.0, 'Dribbling': 5.0, 'Finishing': 2.0, 'First Touch': 6.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 3.0, 'Long Throws': 1.0, 'Marking': 3.0,
        'Passing': 2.0, 'Penalty Taking': 1.0, 'Tackling': 2.0, 'Technique': 4.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 3.0, 'Concentration': 2.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 2.0, 'Positioning': 1.0, 'Teamwork': 2.0, 'Vision': 3.0, 'Work Rate': 3.0,
        'Acceleration': 6.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 1.0, 'Pace': 5.0,
        'Stamina': 7.0, 'Strength': 3.0, 'Weaker Foot': 6.0
    },
    'AML/AMR': {
        'Corners': 0.0, 'Crossing': 5.0, 'Dribbling': 5.0, 'Finishing': 3.0, 'First Touch': 5.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 2.0, 'Long Throws': 1.0, 'Marking': 1.0,
        'Passing': 4.0, 'Penalty Taking': 1.0, 'Tackling': 2.0, 'Technique': 5.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 3.0, 'Concentration': 2.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 3.0, 'Positioning': 1.0, 'Teamwork': 2.0, 'Vision': 6.0, 'Work Rate': 3.0,
        'Acceleration': 10.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 1.0, 'Pace': 10.0,
        'Stamina': 7.0, 'Strength': 3.0, 'Weaker Foot': 5.0
    },
    'AMC': {
        'Corners': 0.0, 'Crossing': 1.0, 'Dribbling': 3.0, 'Finishing': 8.0, 'First Touch': 6.0,
        'Free Kick Taking': 1.0, 'Heading': 1.0, 'Long Shots': 2.0, 'Long Throws': 1.0, 'Marking': 1.0,
        'Passing': 2.0, 'Penalty Taking': 1.0, 'Tackling': 2.0, 'Technique': 4.0, 'Anticipation': 3.0,
        'Bravery': 1.0, 'Composure': 6.0, 'Concentration': 2.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 6.0, 'Positioning': 2.0, 'Teamwork': 1.0, 'Vision': 2.0, 'Work Rate': 2.0,
        'Acceleration': 9.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 5.0, 'Pace': 7.0,
        'Stamina': 6.0, 'Strength': 6.0, 'Weaker Foot': 7.0
    },
    'ST': {
        'Corners': 0.0, 'Crossing': 2.0, 'Dribbling': 5.0, 'Finishing': 8.0, 'First Touch': 6.0,
        'Free Kick Taking': 1.0, 'Heading': 6.0, 'Long Shots': 2.0, 'Long Throws': 1.0, 'Marking': 1.0,
        'Passing': 2.0, 'Penalty Taking': 1.0, 'Tackling': 1.0, 'Technique': 4.0, 'Anticipation': 5.0,
        'Bravery': 1.0, 'Composure': 6.0, 'Concentration': 2.0, 'Decisions': 5.0, 'Leadership': 1.0,
        'Off The Ball': 6.0, 'Positioning': 2.0, 'Teamwork': 1.0, 'Vision': 2.0, 'Work Rate': 2.0,
        'Acceleration': 10.0, 'Agility': 6.0, 'Balance': 2.0, 'Jumping Reach': 5.0, 'Pace': 7.0,
        'Stamina': 6.0, 'Strength': 6.0, 'Weaker Foot': 7.0
    }
}

def parse_html_file(html_content):
    """Parse HTML file and extract player data"""
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return pd.DataFrame()
    
    # Extract headers
    headers = []
    header_row = table.find('tr')
    for th in header_row.find_all('th'):
        headers.append(th.get_text(strip=True))
    
    # Extract data rows
    data = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        row_data = []
        for td in row.find_all('td'):
            row_data.append(td.get_text(strip=True))
        if len(row_data) == len(headers):
            data.append(row_data)
    
    df = pd.DataFrame(data, columns=headers)
    return df

def clean_and_process_data(df):
    """Clean and process the player data"""
    if df.empty:
        return df
    
    # Convert numeric columns
    numeric_columns = ['Age'] + [col for col in df.columns if col not in ['Name', 'Position', 'Inf', 'Transfer Value']]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean transfer value column
    if 'Transfer Value' in df.columns:
        df['Transfer Value Cleaned'] = df['Transfer Value'].str.replace('€', '').str.replace('K', '000').str.replace('M', '000000')
        df['Transfer Value Cleaned'] = df['Transfer Value Cleaned'].str.replace(' - ', '-')
        
        # Handle ranges by taking the average
        def parse_transfer_value(value):
            if pd.isna(value) or value == '':
                return 0
            try:
                if '-' in str(value):
                    parts = str(value).split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                else:
                    return float(value)
            except:
                return 0
        
        df['Transfer Value Numeric'] = df['Transfer Value Cleaned'].apply(parse_transfer_value)
    
    # Remove duplicates based on name, keeping highest transfer value
    if 'Transfer Value Numeric' in df.columns:
        df = df.sort_values('Transfer Value Numeric', ascending=False).drop_duplicates('Name', keep='first')
    
    return df

def calculate_position_score(player_data, position):
    """Calculate player score for a specific position"""
    if position not in POSITION_WEIGHTS:
        return 0
    
    weights = POSITION_WEIGHTS[position]
    total_score = 0
    total_weight = 0
    
    for attribute, weight in weights.items():
        if attribute in player_data and not pd.isna(player_data[attribute]) and weight > 0:
            total_score += player_data[attribute] * weight
            total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0

def calculate_all_position_scores(df):
    """Calculate scores for all positions for all players"""
    positions = list(POSITION_WEIGHTS.keys())
    
    for position in positions:
        df[f'{position}_Score'] = df.apply(lambda row: calculate_position_score(row, position), axis=1)
    
    return df

def get_color_class(score, team_avg, threshold=400):
    """Get color class based on score relative to team average"""
    diff = score - team_avg
    
    if diff >= threshold:
        return "score-excellent"
    elif diff <= -threshold:
        return "score-poor"
    else:
        return "score-average"

def create_formation_display(team_data, formation_name="Starting XI"):
    """Create visual formation display"""
    st.markdown(f"### {formation_name}")
    
    # Formation positions
    formation_positions = [
        "GK", "", "RB", "CB", "CB", "LB", "",
        "DM", "DM", "",
        "AMR", "AMC", "AML", "",
        "ST"
    ]
    
    # Create formation layout
    formation_html = f"""
    <div class="formation-container">
        <h3 style="text-align: center; color: white; margin-bottom: 2rem;">{formation_name}</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; max-width: 600px; margin: 0 auto;">
    """
    
    for i, pos in enumerate(formation_positions):
        if pos == "":
            formation_html += '<div style="height: 60px;"></div>'
        else:
            if pos in team_data:
                player = team_data[pos]
                score = player.get('score', 0)
                name = player.get('name', 'Unknown')
                age = player.get('age', 'N/A')
                
                # Color based on score
                color_class = get_color_class(score, team_data.get('team_avg', 0))
                
                formation_html += f"""
                <div class="player-card">
                    <div style="font-weight: bold; font-size: 0.9rem;">{name}</div>
                    <div style="font-size: 0.8rem; color: #666;">{pos} | Age: {age}</div>
                    <div class="{color_class}">{score:.1f}</div>
                </div>
                """
            else:
                formation_html += f"""
                <div class="player-card" style="opacity: 0.3;">
                    <div style="font-weight: bold;">{pos}</div>
                    <div>No Player</div>
                </div>
                """
    
    formation_html += """
        </div>
    </div>
    """
    
    st.markdown(formation_html, unsafe_allow_html=True)

def solve_team_selection(df, positions_needed):
    """Use Hungarian algorithm to solve optimal team selection"""
    # Create cost matrix
    players = df.index.tolist()
    n_players = len(players)
    n_positions = len(positions_needed)
    
    # Create cost matrix (negative scores because we want to maximize)
    cost_matrix = np.zeros((n_players, n_positions))
    
    for i, player_idx in enumerate(players):
        for j, position in enumerate(positions_needed):
            score_col = f'{position}_Score'
            if score_col in df.columns:
                cost_matrix[i, j] = -df.loc[player_idx, score_col]
            else:
                cost_matrix[i, j] = 0
    
    # Solve using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Create team selection
    team = {}
    used_players = set()
    
    for i, j in zip(row_indices, col_indices):
        if i < len(players) and j < len(positions_needed):
            player_idx = players[i]
            position = positions_needed[j]
            
            if player_idx not in used_players:
                team[position] = {
                    'name': df.loc[player_idx, 'Name'],
                    'age': df.loc[player_idx, 'Age'],
                    'score': df.loc[player_idx, f'{position}_Score'],
                    'player_idx': player_idx
                }
                used_players.add(player_idx)
    
    return team, used_players

def main():
    st.markdown('<h1 class="main-header">⚽ Football Manager Player Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("📁 Upload HTML Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose HTML files from Football Manager",
        type=['html'],
        accept_multiple_files=True,
        help="Upload the HTML files exported from Football Manager"
    )
    
    if uploaded_files:
        # Process all uploaded files
        all_players = []
        
        for uploaded_file in uploaded_files:
            content = uploaded_file.read().decode('utf-8')
            df = parse_html_file(content)
            if not df.empty:
                df = clean_and_process_data(df)
                all_players.append(df)
        
        if all_players:
            # Combine all dataframes
            combined_df = pd.concat(all_players, ignore_index=True)
            combined_df = clean_and_process_data(combined_df)
            
            # Calculate position scores
            combined_df = calculate_all_position_scores(combined_df)
            
            st.success(f"✅ Successfully loaded {len(combined_df)} unique players from {len(uploaded_files)} files")
            
            # Display player statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Players", len(combined_df))
            
            with col2:
                avg_age = combined_df['Age'].mean()
                st.metric("Average Age", f"{avg_age:.1f}")
            
            with col3:
                total_value = combined_df['Transfer Value Numeric'].sum()
                st.metric("Total Squad Value", f"€{total_value/1000000:.1f}M")
            
            with col4:
                top_value = combined_df['Transfer Value Numeric'].max()
                st.metric("Highest Value Player", f"€{top_value/1000000:.1f}M")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Player Rankings", "⚽ Starting XI", "🔄 Second XI"])
            
            with tab1:
                st.header("Player Rankings by Position")
                
                # Position selector
                selected_position = st.selectbox(
                    "Select Position to View Rankings",
                    list(POSITION_WEIGHTS.keys())
                )
                
                if selected_position:
                    score_col = f'{selected_position}_Score'
                    if score_col in combined_df.columns:
                        # Sort by position score
                        ranked_players = combined_df.sort_values(score_col, ascending=False).head(20)
                        
                        # Create ranking table
                        display_cols = ['Name', 'Age', 'Transfer Value'] + [f'{pos}_Score' for pos in POSITION_WEIGHTS.keys()]
                        available_cols = [col for col in display_cols if col in ranked_players.columns]
                        
                        st.dataframe(
                            ranked_players[available_cols].head(20),
                            use_container_width=True,
                            height=600
                        )
            
            with tab2:
                st.header("Optimal Starting XI")
                
                # Define formation positions
                formation_positions = ['GK', 'DL/DR', 'CB', 'CB', 'WBL/WBR', 'DM', 'DM', 'ML/MR', 'MC', 'AML/AMR', 'AMC', 'ST']
                
                # Solve for starting XI
                starting_team, used_players = solve_team_selection(combined_df, formation_positions)
                
                # Calculate team average
                if starting_team:
                    team_scores = [player['score'] for player in starting_team.values()]
                    team_avg = np.mean(team_scores)
                    starting_team['team_avg'] = team_avg
                    
                    # Display formation
                    create_formation_display(starting_team, "Starting XI")
                    
                    # Display team statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Team Average Score", f"{team_avg:.1f}")
                    with col2:
                        st.metric("Total Team Value", f"€{sum(combined_df.loc[player['player_idx'], 'Transfer Value Numeric'] for player in starting_team.values() if 'player_idx' in player)/1000000:.1f}M")
                    with col3:
                        st.metric("Average Age", f"{np.mean([player['age'] for player in starting_team.values()]):.1f}")
            
            with tab3:
                st.header("Second XI (Excluding Starting XI Players)")
                
                if 'used_players' in locals():
                    # Filter out used players
                    available_players = combined_df[~combined_df.index.isin(used_players)]
                    
                    if len(available_players) > 0:
                        # Solve for second XI
                        second_team, _ = solve_team_selection(available_players, formation_positions)
                        
                        if second_team:
                            # Calculate team average
                            team_scores = [player['score'] for player in second_team.values()]
                            team_avg = np.mean(team_scores)
                            second_team['team_avg'] = team_avg
                            
                            # Display formation
                            create_formation_display(second_team, "Second XI")
                            
                            # Display team statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Team Average Score", f"{team_avg:.1f}")
                            with col2:
                                st.metric("Total Team Value", f"€{sum(available_players.loc[player['player_idx'], 'Transfer Value Numeric'] for player in second_team.values() if 'player_idx' in player)/1000000:.1f}M")
                            with col3:
                                st.metric("Average Age", f"{np.mean([player['age'] for player in second_team.values()]):.1f}")
                    else:
                        st.warning("No available players for Second XI after selecting Starting XI")
                else:
                    st.info("Please generate Starting XI first")
    
    else:
        st.info("👆 Please upload HTML files from Football Manager to get started!")
        
        # Show example of what the app does
        st.markdown("""
        ### What this app does:
        
        1. **📁 Upload HTML Files**: Upload exported player data from Football Manager
        2. **📊 Analyze Players**: Calculate position-specific scores using weighted attributes
        3. **⚽ Build Teams**: Use Hungarian algorithm to find optimal team selections
        4. **🎨 Visual Display**: Beautiful formation displays with color-coded performance
        5. **🔄 Multiple Teams**: Create both Starting XI and Second XI
        
        ### Features:
        - **Weighted Scoring**: Each position has specific attribute weights
        - **No Duplicates**: Automatically handles duplicate players (keeps highest value)
        - **Color Coding**: Green for excellent, red for poor performance vs team average
        - **Optimal Selection**: Hungarian algorithm ensures best possible team
        - **Beautiful UI**: Modern, responsive design
        """)

if __name__ == "__main__":
    main()
