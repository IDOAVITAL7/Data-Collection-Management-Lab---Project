import streamlit as st
import pandas as pd
import json
import ast
import openai
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Any, Optional

# --- Configuration & Setup ---

client = openai.OpenAI(api_key="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")

SCHEMA_METADATA = {
    'city': {'type': 'string'},
    'state': {'type': 'string'}, 
    'location': {'type': 'string'}, 
    'ratings': {'type': 'double', 'min': 0.0, 'max': 5.0, 'mean': 4.48},
    'property_number_of_reviews': {'type': 'int', 'mean': 34.7},
    'City_Crime_Rate_Per_100K': {'type': 'double', 'mean': 647.88},
    'Total_Fatalities': {'type': 'bigint', 'mean': 31.54},
    'Median_Income': {'type': 'double', 'mean': 86350.11},
    'Disability_Rate': {'type': 'double', 'mean': 30.76},
    'Median_AQI': {'type': 'double', 'mean': 41.97},
    'amenities_count': {'type': 'int', 'mean': 111.58},
    'price': {'type': 'double'}
}

# --- Paths ---
BASE_DIR = "/Workspace/Users/avital.ido@campus.technion.ac.il/app_data"
DATA_DIR = os.path.join(BASE_DIR, "data")
MAP_FILE = os.path.join(BASE_DIR, "location_map.json")

# --- Helper Functions ---

def safe_parse(value: Any, default: Any = None) -> Any:
    if value is None or value == "": return default
    if not isinstance(value, str): return value
    try: return json.loads(value)
    except:
        try: return ast.literal_eval(value)
        except: return default

def get_score_color(score: float) -> str:
    try: score = float(score)
    except: return "#808080"
    score = max(0, min(10, score))
    if score < 5: return "#ff4b4b"
    elif score < 7.5: return "#ffa500"
    else: return "#21c354"

def get_gradient_color(score_val: float) -> str:
    """
    Calculates a color from Red (0) to Yellow (50) to Green (100).
    Returns a hex string.
    """
    try:
        val = float(score_val)
    except:
        return "#e0e0e0" # Gray for N/A
    
    val = max(0, min(100, val)) # Clamp 0-100
    
    # Red: (255, 75, 75) -> #ff4b4b
    # Yellow: (255, 193, 7) -> #ffc107
    # Green: (33, 195, 84) -> #21c354
    
    if val <= 50:
        # Interpolate Red -> Yellow
        ratio = val / 50.0
        r = int(255 + (255 - 255) * ratio) # 255 -> 255
        g = int(75 + (193 - 75) * ratio)   # 75 -> 193
        b = int(75 + (7 - 75) * ratio)     # 75 -> 7
    else:
        # Interpolate Yellow -> Green
        ratio = (val - 50) / 50.0
        r = int(255 + (33 - 255) * ratio)  # 255 -> 33
        g = int(193 + (195 - 193) * ratio) # 193 -> 195
        b = int(7 + (84 - 7) * ratio)      # 7 -> 84
        
    return f"#{r:02x}{g:02x}{b:02x}"
# --- Optimized Data Loading ---

@st.cache_data
def load_location_map() -> Dict[str, str]:
    """
    Loads the JSON mapping that links cities/states to their filename code.
    Example: {"miami": "fl", "florida": "fl"}
    """
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE, 'r') as f:
            return json.load(f)
        
    fallback_path = "/Workspace/Users/avital.ido@campus.technion.ac.il/unique_locations_list.json"
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {str(item).lower(): str(item).lower() for item in data}
                
                return data
        except Exception:
            return {} 
            
    return {}


    

def load_data_for_state(state_code: str):
    """
    Loads ONLY the specific parquet file for the requested state.
    This is extremely fast compared to loading the whole dataset.
    """
    file_path = os.path.join(DATA_DIR, f"{state_code}.parquet")
    
    if not os.path.exists(file_path):
        st.error(f"Data file for state '{state_code}' not found.")
        return pd.DataFrame()
        
    try:
        df = pd.read_parquet(file_path)
        
        # Deduplicate immediately upon loading
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'])
        else:
            cols_to_check = [c for c in ['name', 'city', 'price'] if c in df.columns]
            if cols_to_check:
                df = df.drop_duplicates(subset=cols_to_check)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Logic ---

def find_similar_properties(property_data: pd.DataFrame, target_record: Dict, top_k: int = 50):
    top_k = min(top_k, len(property_data))
    # FIXED: Uncommented and fixed logic to ensure it returns a DataFrame
    if property_data.empty: return property_data
    
    numeric_cols = [col for col, meta in SCHEMA_METADATA.items() if meta.get('type') in ['double', 'int', 'bigint', 'float'] and col in property_data.columns]
    
    df_numeric = property_data[numeric_cols].copy().fillna(0)
    target_vector = []
    valid_cols = []
    
    for col in numeric_cols:
        val = target_record.get(col)
        if val is not None and val != "":
            try: target_vector.append(float(val)); valid_cols.append(col)
            except: pass
            
    if not valid_cols: 
        return property_data.head(top_k)

    X = df_numeric[valid_cols]
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        target_vector_scaled = scaler.transform([target_vector])
        
        k = min(len(X), top_k)
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_scaled)
        
        dists, idxs = nn.kneighbors(target_vector_scaled)
        
        sim_props = property_data.iloc[idxs[0]].copy()
        sim_props['match_score'] = 1 / (1 + dists[0])
        
        return sim_props
    except: 
        return property_data.head(top_k)

def update_best_match(df, row, decision, selections=None):
    if selections is None: selections = []
    
    if decision == "like":
        # Convert row to dictionary
        prop_data = row.to_dict() if hasattr(row, 'to_dict') else row
        
        # Check if property URL already exists in current selections
        # This prevents the same property from appearing twice in Favorites
        existing_urls = {item.get('url') for item in selections}
        
        if prop_data.get('url') not in existing_urls:
            selections.append(prop_data)
            
    # Remove the current row from the dataframe (Standard Swipe Logic)
    if len(df) > 0:
        return df.iloc[1:].reset_index(drop=True), selections
        
    return df, selections

def load_filtered_data(max_price: float, location_filter: str, required_columns: list):
    """
    Loads data and removes duplicates immediately to prevent same property appearing twice.
    """
    data_path = "/dbfs/FileStore/Users/avital.ido@campus.technion.ac.il/eliezer_data"
    try:
        
        # filters = [('price', '<=', float(max_price))]
        filters = [
                [('price', '<=', float(max_price)), ('city', '==', location_filter)],
                [('price', '<=', float(max_price)), ('state', '==', location_filter)],
                [('price', '<=', float(max_price)), ('county', '==', location_filter)]
            ]
        df = pd.read_parquet(data_path, columns=required_columns, filters=filters, engine='pyarrow')
        
        # Drop Duplicates Logic
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'])
        else:
            cols_to_check = [c for c in ['name', 'city', 'price'] if c in df.columns]
            if cols_to_check:
                df = df.drop_duplicates(subset=cols_to_check)
        
        return df
    except Exception as e:
        st.error(f"Error loading filtered data: {e}")
        return pd.DataFrame()


def find_best_match(user_location: str, max_price: float, requirements: str):
    # 1. Resolve State File
    location_map = load_location_map()
    search_term = user_location.lower().strip()
    
    # Try to find the state code from the map
    state_code = location_map.get(search_term)
    
    if not state_code:
        # Fallback: check if the input is a direct match to keys
        # If not found, we can't load specific data.
        required_cols = list(SCHEMA_METADATA.keys()) + ['name', 'listing_name', 'image', 'url', 'seller_info', 'guests', 'highlights', 'description', 'reviews', 'images', 'pets_allowed']
        return load_filtered_data(max_price, user_location, required_cols)

    # 2. Load SPECIFIC State Data
    df = load_data_for_state(state_code)
    
    if df.empty: return pd.DataFrame()
    
    # 3. Filter by Price
    df = df[df['price'] <= float(max_price)]
    
    # 4. Filter by specific City (if user entered a city, not just the state)
    if search_term != state_code: # means user entered a specific city
         df = df[
            df['city'].astype(str).str.lower().str.contains(search_term, na=False) | 
            df['state'].astype(str).str.lower().str.contains(search_term, na=False)
        ]
    
    if 'selections' in st.session_state and st.session_state.selections:
        # Get list of URLs currently in favorites
        liked_urls = [item.get('url') for item in st.session_state.selections]
        # Keep only rows where the URL is NOT in the liked list
        if 'url' in df.columns:
            df = df[~df['url'].isin(liked_urls)]

    if df.empty: return pd.DataFrame()

    # 5. LLM Logic
    system_prompt = f"""
    You are a highly intelligent data matching assistant for an Airbnb search engine.
    Your task is to translate user requirements into a structured JSON record that follows the provided schema.

    RULES FOR VALUE ASSIGNMENT:
    1. UNDERSTAND CONTEXT: Analyze the user's free text. If they ask for "safety", reduce 'City_Crime_Rate_Per_100K'. If they want "luxury", increase 'Median_Income' and 'amenities_count'.
    2. NUMERIC FIELDS:
       - If the user provides a requirement that maps to a numeric field, set a realistic value based on the [min, max, mean, std] provided.
       - IMPORTANT: If the user DOES NOT mention or imply anything about a numeric field, you MUST set its value to exactly the 'mean' provided in the schema.
    3. STRING FIELDS:
       - If information is provided or can be inferred (like City/State from location), fill it.
       - If NO information is provided for a string field, set it to an empty string "".
    4. CONSISTENCY: Ensure all keys from the schema are present in the output.

    Schema Metadata:
    {list(SCHEMA_METADATA.keys())}

    Return ONLY a valid JSON object.
    """
    
    try:
        res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Loc: {user_location}, Max: {max_price}, Req: {requirements}"}], response_format={"type": "json_object"}, temperature=0.7)
        target = json.loads(res.choices[0].message.content)
        target['price'] = float(max_price)
    except: target = {'price': float(max_price)}
    
    return find_similar_properties(df, target)

# --- CALLBACK FUNCTION ---

def handle_swipe(decision):
    if st.session_state.current_df is None or len(st.session_state.current_df) == 0:
        return

    current_row = st.session_state.current_df.iloc[0]
    
    new_df, new_selections = update_best_match(
        st.session_state.current_df, 
        current_row, 
        decision, 
        st.session_state.selections
    )
    
    st.session_state.current_df = new_df
    st.session_state.selections = new_selections
    st.session_state.gallery_idx = 0

# --- UI ---

def main():
    st.set_page_config(page_title="Airbnb Matcher", page_icon="🏡", layout="wide")
    
    if 'page' not in st.session_state: st.session_state.page = 'form'
    if 'current_df' not in st.session_state: st.session_state.current_df = None
    if 'selections' not in st.session_state: st.session_state.selections = []
    if 'gallery_idx' not in st.session_state: st.session_state.gallery_idx = 0
    if 'last_prop_id' not in st.session_state: st.session_state.last_prop_id = None
        
    st.markdown("""<style>
        .block-container { padding-top: 2rem; }
        .header-container { text-align: center; margin-bottom: 30px; }
        .header-title { 
            font-size: 4rem; 
            font-weight: 800; 
            background: linear-gradient(90deg, #FF5A5F 0%, #FF385C 100%);
            -webkit-background_clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            cursor: pointer;
            text-align: center;
        }
        .header-subtitle { font-size: 1.2rem; color: #717171; font-weight: 500; margin-top: -10px; text-align: center; }
        .main-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .score-badge { font-size: 2em; font-weight: bold; padding: 10px; border-radius: 12px; color: white; text-align: center; display: block; }
        .highlight-item { background: #f8f9fa; border-left: 4px solid #FF5A5F; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
        .reviews-container { max-height: 250px; overflow-y: auto; padding: 10px; border: 1px solid #eee; border-radius: 8px; margin-top: 10px; }
        .review-bubble { background: #f7f7f7; padding: 12px; border-radius: 12px; margin-bottom: 10px; font-style: italic; border-left: 3px solid #ddd; font-size: 0.9em; color: #444; }
        .custom-img { width: 100%; border-radius: 10px; object-fit: cover; height: 400px; display: block; margin-left: auto; margin-right: auto; }
        div.stButton > button.title-btn { background: none; border: none; font-size: 4rem; font-weight: 800; color: #FF5A5F; width: 100%; }
        .about-card { padding: 20px; border-radius: 10px; background: #f9f9f9; margin-bottom: 15px; border-left: 5px solid #FF5A5F; }
        button[data-testid="baseButton-secondary"] { border-color: #ff4b4b !important; color: #ff4b4b !important; font-weight: bold; }
        button[data-testid="baseButton-secondary"]:hover { background-color: #ff4b4b !important; color: white !important; }
        button[data-testid="baseButton-primary"] { background-color: #21c354 !important; border-color: #21c354 !important; font-weight: bold; }
        button[data-testid="baseButton-primary"]:hover { background-color: #1a9c43 !important; border-color: #1a9c43 !important; }

        .score-container {
            display: flex;
            flex-direction: row;
            width: 100%;          /* Force container to take full width */
            gap: 5px;             /* Small gap to fit all 10 items */
            padding: 5px 0;
            margin-bottom: 15px;
            /* overflow-x is removed so items shrink to fit instead of scroll */
        }
        .score-container::-webkit-scrollbar {
            display: none; /* Hide scrollbar for Chrome/Safari */
        }
        .score-card {
            flex: 1;              /* Grow to fill space equally */
            min-width: 0;         /* Allow item to shrink below content size if needed */
            text-align: center;
            padding: 6px 2px;
            border-radius: 6px;
            color: #333;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        
        .score-val { 
            font-weight: 800; 
            font-size: 0.9em; 
            line-height: 1.1;
        }
        
        .score-label { 
            font-size: 0.55em;    /* Slightly smaller to prevent text overflow */
            line-height: 1.1; 
            font-weight: 600; 
            text-transform: uppercase; 
            margin-top: 2px;
            white-space: nowrap;  /* Keep text on one line */
            overflow: hidden;     /* Hide overflow */
            text-overflow: ellipsis; /* Add ... if text is too long */
            width: 100%;
        }
    </style>""", unsafe_allow_html=True)
    
    # Header
    _, col_center, _ = st.columns([1, 6, 1])
    with col_center:
        if st.button("🏡 AirBNB Matcher", key="home_btn", use_container_width=True):
            st.session_state.selections = []
            st.session_state.current_df = None
            st.session_state.page = 'form'
            st.rerun()
        st.markdown('<p class="header-subtitle">Developed by Ido Avital & Eliezer Mashihov</p>', unsafe_allow_html=True)

    # PAGE 1: FORM
    if st.session_state.page == 'form':
        left_col, center_col, right_col = st.columns([1, 2, 1])
        with center_col:
            st.markdown("<h3 style='text-align: center;'>Find your perfect stay</h3><br>", unsafe_allow_html=True)
            
            # Load locations from map
            location_map = load_location_map()
            available_locs = sorted(list(location_map.keys()))
            
            if available_locs:
                loc_display = [l.title() for l in available_locs]
                selected_loc_disp = st.selectbox("Select Target Location", options=loc_display)
                loc = selected_loc_disp.lower() 
            else:
                loc = st.text_input("Target Location").lower()
            
            price = st.number_input("Maximum Price per Night ($)", min_value=50, value=1000, step=10)
            req = st.text_area("Describe your dream vacation", height=145, placeholder="e.g. Quiet apartment near the beach...")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Start Search", use_container_width=True):
                loading_msg = "🔍 Scanning Airbnb ecosystem... (This may take up to 90s)"
                with st.spinner(loading_msg):
                    res = find_best_match(loc, price, req)
                    if not res.empty:
                        st.session_state.current_df = res
                        st.session_state.page = 'swipe'
                        st.rerun()
                    else: st.error("No matches found. Please try different criteria.")
            
            st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
            if st.button("ℹ️ About This App", use_container_width=True):
                st.session_state.page = 'about'
                st.rerun()

    # PAGE 4: ABOUT
    elif st.session_state.page == 'about':
        st.subheader("ℹ️ About AirBNB Matcher")
        st.markdown("""
        <div class="about-card"><h3>🎯 The Mission</h3><p>Finding the perfect Airbnb can be overwhelming. This tool uses <b>AI and Big Data</b> to match you with properties that fit your specific needs—not just your budget.</p></div>
        <div class="about-card"><h3>⚙️ How It Works</h3><ul><li><b>Smart Filtering:</b> We scan partitioned data for maximum speed.</li><li><b>AI Analysis:</b> GPT-4 understands your text description.</li><li><b>Similarity Matching:</b> KNN algorithms find properties that match your profile.</li></ul></div>
        <div class="about-card"><h3>👨‍💻 Developers</h3><p>Built by <b>Ido Avital and Eliezer Mashihov</b>.</p></div>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Back to Search", use_container_width=True):
            st.session_state.page = 'form'
            st.rerun()

    # PAGE 2: SWIPE
    elif st.session_state.page == 'swipe':
        if st.session_state.current_df is None or len(st.session_state.current_df) == 0:
            st.success("🎉 Reviewed all!"); st.session_state.page = 'results'; st.rerun(); return

        row = st.session_state.current_df.iloc[0]
        listing_name = row.get('name', row.get('listing_name', 'Listing'))
        parts = listing_name.split(' · ', 2)
        name = parts[0]
        rating = parts[1]
        short_description = parts[2]

        title = row.get('listing_title', name)

        pets_allowed = row.get('pets_allowed', 'No')
        if pets_allowed != 'true':
            pets_allowed = 'No Pets Allowed'
        else:
            pets_allowed = 'Pets Allowed'
            
        df_len = len(st.session_state.current_df)
        prop_key = f"{df_len}_{name[:10]}" 
        
        if st.session_state.last_prop_id != name:
            st.session_state.gallery_idx = 0
            st.session_state.last_prop_id = name
        
        main_img = row.get('image', None)
        extra_imgs = safe_parse(row.get('images'), [])
        gallery_images = (extra_imgs if isinstance(extra_imgs, list) else [])
        if not gallery_images: gallery_images = ["https://via.placeholder.com/800x600?text=No+Image"]
        
        idx = st.session_state.gallery_idx
        if idx >= len(gallery_images): idx = 0
        
        with st.container():
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.markdown(f"<h2>{title}</h2>", unsafe_allow_html=True)
            st.markdown(f"📍 {row.get('city')}, {row.get('state')}<hr>", unsafe_allow_html=True)
            
            score_columns = [
                ('Overall', 'overall_quality_score'),
                ('Location', 'location_convenience_score'),
                ('Host', 'host_excellence_score'),
                ('Value', 'value_perception_score'),
                ('Comfort', 'comfort_index_score'),
                ('Family', 'family_suitability_score'),
                ('Business', 'business_ready_score'),
                ('Social', 'social_experience_score'),
                ('Wellness', 'relaxation_wellness_score'),
                ('Local', 'local_immersion_score')
            ]
            
            # Construct the HTML for the score bar
            scores_html = '<div class="score-container">'
            for label, col_name in score_columns:
                try:
                    # raw_val = row.get(col_name, 0)
                    # val = float(raw_val) if raw_val is not None else 0.0
                    val = 3.4
                except: val = 0
                
                bg_color = get_gradient_color(val)
                scores_html += f'<div class="score-card" style="background-color: {bg_color};">'
                scores_html += f'<div class="score-val">{val:.0f}</div>'
                scores_html += f'<div class="score-label">{label}</div>'
                scores_html += '</div>'
            scores_html += '</div>'
            st.markdown(scores_html, unsafe_allow_html=True)

            c_img, c_stats = st.columns([1.5, 1])
            with c_img:
                st.markdown(f'<img src="{gallery_images[idx]}" class="custom-img">', unsafe_allow_html=True)
                if len(gallery_images) > 1:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    c_spacer_l, c_prev, c_text, c_next, c_spacer_r = st.columns([3, 1, 3, 1, 3])
                    with c_prev:
                        if st.button("⬅️", key=f"prev_{prop_key}", use_container_width=True): st.session_state.gallery_idx = (idx - 1) % len(gallery_images); st.rerun()
                    with c_text:
                        st.markdown(f"<div style='text-align:center;color:#555;font-weight:500;padding-top:8px;'>{idx + 1} / {len(gallery_images)}</div>", unsafe_allow_html=True)
                    with c_next:
                        if st.button("➡️", key=f"next_{prop_key}", use_container_width=True): st.session_state.gallery_idx = (idx + 1) % len(gallery_images); st.rerun()

            with c_stats:
                score = float(row.get('risk_score_raw', 0) or 0)*10
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="score-badge" style="background:{get_score_color(score)}">{score:.1f}/10</div>', unsafe_allow_html=True)
                st.write(f"")
                st.write(f"**💰 Price:** ${row.get('price')}")
                st.write(f"**🏠 Arrangement:** {short_description}")
                st.write(f"**👥 Guests:** {row.get('guests', 'N/A')}")
                st.write(f"**⭐ Rating:** {row.get('ratings')} ({row.get('property_number_of_reviews')} reviews)")
                st.write(f"**🐕 Pets:** {pets_allowed}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.link_button("🏠 View in Airbnb", row.get('url', '#'))
                s = safe_parse(row.get('seller_info'), {})
                st.link_button("👤 Host details", s.get('url', '#') if isinstance(s, dict) else '#')

            hls = safe_parse(row.get('highlights'), [])
            if hls and isinstance(hls, list):
                st.markdown("#### ✨ Highlights")
                ch = st.columns(min(3, len(hls)))
                for i, h in enumerate(hls[:3]):
                    if isinstance(h, dict):
                        with ch[i]: st.markdown(f'<div class="highlight-item"><b>{h.get("name")}</b><br><small>{h.get("value")}</small></div>', unsafe_allow_html=True)

            with st.expander("📝 Description"): st.write(row.get('description', 'No desc.'))
            revs = safe_parse(row.get('reviews'), [])
            st.markdown(f"#### 💬 Reviews ({len(revs) if isinstance(revs, list) else 0})")
            if isinstance(revs, list) and revs:
                rh = "".join([f'<div class="review-bubble">"{r}"</div>' for r in revs])
                st.markdown(f'<div class="reviews-container">{rh}</div>', unsafe_allow_html=True)
            else: st.info("No text reviews.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        ca1, ca2, ca3 = st.columns([1, 2, 1])
        with ca1:
            st.button("❌ Pass", on_click=handle_swipe, args=("dislike",), use_container_width=True, key=f"pass_btn_{prop_key}", type="secondary")
        with ca2:
            if st.button("📋 Favorites", use_container_width=True): st.session_state.page = 'results'; st.rerun()
        with ca3:
            st.button("❤️ Like", on_click=handle_swipe, args=("like",), use_container_width=True, type="primary", key=f"like_btn_{prop_key}")

    # PAGE 3: RESULTS
    elif st.session_state.page == 'results':
        st.subheader("🎉 Your Favorites")
        if not st.session_state.selections: st.info("No likes yet.")
        else:
            for i, item in enumerate(st.session_state.selections):
                with st.expander(f"{i+1}. {item.get('name', 'Prop')} - ${item.get('price')}", expanded=True):
                    cr1, cr2 = st.columns([1, 3])
                    with cr1: 
                        img = item.get('image')
                        if img: st.markdown(f'<img src="{img}" class="custom-img" style="height: 150px;">', unsafe_allow_html=True)
                    with cr2:
                        st.write(f"Location: {item.get('city')}")
                        st.link_button("Book", item.get('url', '#'))
        
        st.markdown("---")
        cb1, cb2 = st.columns(2)
        with cb1: 
            if st.button("⬅️ Back"): st.session_state.page = 'swipe'; st.rerun()
        with cb2:
            if st.button("🔄 New Search"): 
                st.session_state.selections = []; st.session_state.current_df = None; st.session_state.page = 'form'; st.rerun()

if __name__ == "__main__":
    main()
