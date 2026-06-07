import streamlit as st
import pandas as pd
import io
import re
import datetime
import sqlite3
import torch
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="Perfetti Van Melle | SKU Management System", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        .stButton>button { border-radius: 20px; font-weight: 700; }
        .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E3E8F2; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
        .pvm-header { color: #15247A; font-weight: 900; letter-spacing: -0.02em; }
    </style>
""", unsafe_allow_html=True)

CURRENT_USER = "regional.analyst"

# ==========================================
# 2. Advanced NLP & Domain Dictionaries
# ==========================================
COUNTRY_DICTIONARIES = {
    "Vietnam": {
        "replacements": {
            "straw": "strawberry", "pfruit": "passionfruit", "chiaki": "chia kiwi",
            "px": "pouch", "bags": "pouch", "bag": "pouch", "pouch": "pouch",
            "rect": "rectangular", "llp": "lollipop"
        },
        "inferred_attributes": [
            (r"45\s*(px|pouch|bag|b)\b", "small pouch"),
            (r"24\s*(px|pouch|bag|b)\b", "medium pouch")
        ]
    }
}

GLOBAL_ABBREV = {
    "al": "alpenliebe", "alp": "alpenliebe", "cc": "chupa chups", 
    "mt": "mentos", "mts": "mentos", "gl": "golia", "bb": "big babol"
}

def parse_specs(desc: str) -> dict:
    if not isinstance(desc, str): return {}
    out = {}
    weight = re.search(r'(\d+(?:[.,]\d+)?)\s*g\b', desc, re.I)
    if weight: out["weight"] = weight.group(1) + "g"
    pcs = re.search(r'(\d+)\s*(?:pcs?|u)\b', desc, re.I)
    if pcs: out["pieces"] = pcs.group(1) + "pcs"
    pack = re.search(r'(\d+)\s*(box|bag|pouch|tin|stick|disp|db|px)', desc, re.I)
    if pack: out["pack"] = pack.group(1) + " " + pack.group(2).lower()
    return out

def advanced_normalize(text: str, country: str = "General") -> str:
    if not isinstance(text, str): return ""
    t = f" {text.lower()} "
    t = t.replace("&", " and ")
    
    for k, v in GLOBAL_ABBREV.items():
        t = re.sub(rf'\b{k}\b', v, t)
        
    config = COUNTRY_DICTIONARIES.get(country, {"replacements": {}, "inferred_attributes": []})
    for k, v in config["replacements"].items():
        t = re.sub(rf'\b{k}\b', v, t)
            
    for regex, structural_tag in config["inferred_attributes"]:
        if re.search(regex, t):
            t += f" {structural_tag} "
            
    t = re.sub(r'[()\/.,\-\[\]]', ' ', t)
    # Strip explicit sizes for deep semantic matching
    t = re.sub(r'\d+(\.\d+)?\s*(g|kg|px|pcs|pc|box|bag|bags|stick|sticks|tin)\b', ' ', t)
    return " ".join(t.split())

@st.cache_resource
def get_ml_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_col(df, keywords):
    """Helper to dynamically locate column names based on keywords."""
    return next((c for c in df.columns if any(k in str(c).lower() for k in keywords)), df.columns[0])

# ==========================================
# 3. State Management Configuration
# ==========================================
def seed_initial_state():
    if 'audit_trail' not in st.session_state:
        st.session_state.audit_trail = [
            {"at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "by": "system", "action": "Application initialized", "detail": "Awaiting data import pipeline", "method": "system", "note": ""}
        ]
    # Initialize empty dataframes to await the 3-step upload
    if 'mothers' not in st.session_state:
        st.session_state.mothers = pd.DataFrame(columns=['id', 'code', 'desc', 'brand', 'packType', 'weight'])
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=['child_code', 'child_desc', 'mother_code'])
    if 'children' not in st.session_state:
        st.session_state.children = pd.DataFrame(columns=['id', 'country', 'code', 'desc', 'mother_id', 'status', 'confidence', 'method', 'note'])

seed_initial_state()

def log_event(action, detail, method, note=""):
    st.session_state.audit_trail.insert(0, {
        "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "by": CURRENT_USER, "action": action, "detail": detail, "method": method, "note": note
    })

# ==========================================
# 4. Sidebar Layout Mapping
# ==========================================
st.sidebar.markdown("""
    <div style='padding: 10px 0px;'>
        <h2 style='color: #15247A; margin-bottom: 0px; font-weight:900;'>Perfetti Van Melle</h2>
        <small style='color: #00AEEF; font-weight: 800; text-transform: uppercase;'>SKU Management System</small>
    </div>
""", unsafe_allow_html=True)

tabs = {
    "dashboard": "📋 Dashboard Overview",
    "import": "📥 Data Ingestion Pipeline",
    "workbench": "🔄 Mapping Workbench",
    "unmapped": "📋 Unmapped Queue",
    "conflicts": "⚠️ Conflict Detection",
    "registry": "🗄️ Master Registry",
    "audit": "📜 Audit Ledger"
}
selected_tab = st.sidebar.radio("Navigation Menu", list(tabs.keys()), format_func=lambda x: tabs[x])

st.sidebar.markdown(f"""
    <div style='margin-top: 40px; padding: 12px; background: #E6F7FE; border-radius: 8px;'>
        <span style='font-size: 11px; color: #15247A; font-weight: 800; text-transform: uppercase;'>Session Analyst</span><br/>
        <strong style='color: #1C2433;'>{CURRENT_USER}</strong>
    </div>
""", unsafe_allow_html=True)

mothers_df = st.session_state.mothers
children_df = st.session_state.children

# ==========================================
# 5. Core Views Module Implementation
# ==========================================

if selected_tab == "dashboard":
    st.markdown("<h1 class='pvm-header'>Mapping Overview</h1>", unsafe_allow_html=True)
    st.markdown("Consolidated metrics of SKU mapping across operating regions.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><h5>Total Unprocessed</h5><h2>{len(children_df)}</h2></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><h5 style='color: #1E9E6A;'>Confirmed Links</h5><h2>{len(children_df[children_df['status'].isin(['confirmed', 'auto_confirmed'])])}</h2></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><h5 style='color: #00AEEF;'>Awaiting Review</h5><h2>{len(children_df[children_df['status']=='suggested'])}</h2></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><h5 style='color: #F5A623;'>Unmapped Flags</h5><h2>{len(children_df[children_df['status']=='unmapped'])}</h2></div>", unsafe_allow_html=True)
    
    if len(children_df) == 0:
        st.info("No data loaded. Please proceed to the Data Ingestion Pipeline to upload your matrices.")

elif selected_tab == "import":
    st.header("Data Ingestion Pipeline")
    st.markdown("Upload your structural datasets. The system will process history auto-carries and run ML fuzzy embeddings on new SKUs.")
    
    # The 3-Step Upload Interface
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Master Mother SKUs**")
        mother_file = st.file_uploader("Target Canonical Registry", type=["xlsx", "csv"], key="m")
    with col2:
        st.markdown("**2. Historical Mappings**")
        hist_file = st.file_uploader("Previous Child->Mother Links", type=["xlsx", "csv"], key="h")
    with col3:
        st.markdown("**3. New Unmapped SKUs**")
        child_file = st.file_uploader("Raw P&L Extraction", type=["xlsx", "csv"], key="c")
        
    if mother_file and hist_file and child_file:
        if st.button("🚀 Execute History-Aware ML Pipeline", type="primary"):
            with st.spinner("Analyzing history and executing deep learning vector calculations..."):
                try:
                    read_file = lambda f: pd.read_excel(f) if f.name.endswith('.xlsx') else pd.read_csv(f)
                    df_mother = read_file(mother_file)
                    df_hist = read_file(hist_file)
                    df_child = read_file(child_file)
                    
                    # Discover Columns
                    m_code = get_col(df_mother, ['code', 'id'])
                    m_desc = get_col(df_mother, ['desc', 'name'])
                    
                    h_c_code = get_col(df_hist, ['child', 'code'])
                    h_c_desc = get_col(df_hist, ['child', 'desc'])
                    h_m_code = get_col(df_hist, ['mother', 'target'])
                    
                    c_code = get_col(df_child, ['code', 'id'])
                    c_desc = get_col(df_child, ['desc', 'name'])
                    c_country = get_col(df_child, ['country', 'market']) if any(k in str(c).lower() for c in df_child.columns for k in ['country', 'market']) else None

                    # Force ID Columns to String to prevent merge runtime errors
                    df_mother[m_code] = df_mother[m_code].astype(str).str.strip()
                    df_hist[h_c_code] = df_hist[h_c_code].astype(str).str.strip()
                    df_hist[h_m_code] = df_hist[h_m_code].astype(str).str.strip()
                    df_child[c_code] = df_child[c_code].astype(str).str.strip()
                    
                    # Ensure Mother ID system
                    if 'id' not in df_mother.columns:
                        df_mother['id'] = df_mother[m_code] # Use code as ID if explicit ID is missing
                    
                    # --- Build Engine History Dictionary ---
                    history_dict = dict(zip(df_hist[h_c_code], df_hist[h_m_code]))
                    
                    # --- Text Normalization & Embeddings ---
                    st.toast("Generating semantic embeddings...", icon="🧠")
                    model = get_ml_model()
                    
                    mother_clean = df_mother[m_desc].apply(lambda x: advanced_normalize(str(x))).tolist()
                    hist_clean = df_hist[h_c_desc].apply(lambda x: advanced_normalize(str(x))).tolist()
                    child_clean = df_child[c_desc].apply(lambda x: advanced_normalize(str(x))).tolist()
                    
                    mother_embs = model.encode(mother_clean, convert_to_tensor=True)
                    hist_embs = model.encode(hist_clean, convert_to_tensor=True)
                    child_embs = model.encode(child_clean, convert_to_tensor=True)
                    
                    # --- Triaging & Matching Engine ---
                    results = []
                    for i, row in df_child.iterrows():
                        current_code = str(row[c_code])
                        current_emb = child_embs[i]
                        country_val = str(row[c_country]) if c_country else "Unknown"
                        desc_val = str(row[c_desc])
                        
                        # TIER 1: Exact History Match
                        if current_code in history_dict:
                            target_mother_code = history_dict[current_code]
                            # Find matching mother ID
                            m_match = df_mother[df_mother[m_code] == target_mother_code]
                            m_id = m_match['id'].iloc[0] if not m_match.empty else None
                            
                            results.append({
                                "id": f"c_{i}", "country": country_val, "code": current_code, "desc": desc_val,
                                "mother_id": m_id, "status": "auto_confirmed", "confidence": 1.0, "method": "history", "note": "Auto-carried from history"
                            })
                            continue
                            
                        # TIER 2 & 3: Semantic ML Match
                        sim_mother = util.cos_sim(current_emb, mother_embs)[0]
                        best_m_score, best_m_idx = torch.max(sim_mother, dim=0)
                        
                        sim_hist = util.cos_sim(current_emb, hist_embs)[0]
                        best_h_score, best_h_idx = torch.max(sim_hist, dim=0)
                        
                        best_mother_id = None
                        status = "unmapped"
                        score = 0.0
                        method = ""
                        
                        if best_h_score > best_m_score and best_h_score > 0.45:
                            # Matched via historical synonym
                            best_hist_mother_code = df_hist.iloc[best_h_idx.item()][h_m_code]
                            m_match = df_mother[df_mother[m_code] == best_hist_mother_code]
                            best_mother_id = m_match['id'].iloc[0] if not m_match.empty else None
                            score = round(best_h_score.item(), 3)
                            method = "fuzzy_via_history"
                            status = "suggested" if score < 0.85 else "auto_confirmed"
                            
                        elif best_m_score > 0.45:
                            # Matched direct to mother registry
                            best_mother_id = df_mother.iloc[best_m_idx.item()]['id']
                            score = round(best_m_score.item(), 3)
                            method = "fuzzy_direct"
                            status = "suggested" if score < 0.85 else "auto_confirmed"
                            
                        results.append({
                            "id": f"c_{i}", "country": country_val, "code": current_code, "desc": desc_val,
                            "mother_id": best_mother_id, "status": status, "confidence": score, "method": method, "note": ""
                        })

                    # --- Save State ---
                    if 'brand' not in df_mother.columns: df_mother['brand'] = "Unknown"
                    st.session_state.mothers = df_mother
                    st.session_state.history = df_hist
                    st.session_state.children = pd.DataFrame(results)
                    
                    log_event("Batch ML Pipeline Executed", f"Processed {len(results)} new records.", "system")
                    st.success("Pipeline Execution Complete! Navigate to Dashboard or Workbench to review.")
                    
                except Exception as e:
                    st.error(f"Processing error: {e}")

elif selected_tab == "workbench":
    st.header("Mapping Workbench Engine")
    st.markdown("Verify the algorithmic suggestions against parsed metadata properties.")
    
    f_status = st.selectbox("Filter Current Queue State Matrix", ["suggested", "auto_confirmed", "confirmed", "unmapped", "all"])
    search_query = st.text_input("Search code indexes or keyword descriptions...")
    
    view_df = children_df if f_status == "all" else children_df[children_df['status'] == f_status]
    if search_query:
        view_df = view_df[view_df['desc'].str.contains(search_query, case=False) | view_df['code'].str.contains(search_query)]
        
    for idx, c_row in view_df.iterrows():
        specs = parse_specs(c_row['desc'])
        current_mother = mothers_df[mothers_df['id'] == str(c_row['mother_id'])] if pd.notnull(c_row['mother_id']) else None
        
        with st.expander(f"🌐 {c_row['country']} | Code ID: {c_row['code']} — {c_row['desc']}"):
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown("**Parsed Spec Values:**")
                st.json(specs)
                if current_mother is not None and not current_mother.empty:
                    st.info(f"🔒 **Assigned Parent Node:** {current_mother.iloc[0]['desc']} ({current_mother.iloc[0]['code']})")
                    st.caption(f"Assigned via: {c_row['method']} | Confidence: {c_row['confidence']*100}%")
                else:
                    st.warning("⚠️ Status: Unmapped")
                    
            with cols[1]:
                # If they want to override or map an unmapped one, we search the Mothers DB
                if st.button("Confirm Current Assign", key=f"conf_btn_{idx}", disabled=c_row['status'] in ['confirmed', 'auto_confirmed']):
                    st.session_state.children.at[idx, 'status'] = 'confirmed'
                    st.rerun()
                
                st.markdown("**Search Registry for Override:**")
                search_override = st.text_input("Search Mothers...", key=f"search_{idx}")
                if search_override:
                    opts = mothers_df[mothers_df['desc'].str.contains(search_override, case=False) | mothers_df['code'].str.contains(search_override)]
                    for _, opt_m in opts.head(3).iterrows():
                        if st.button(f"Link -> {opt_m['desc']} ({opt_m['code']})", key=f"link_{idx}_{opt_m['id']}"):
                            st.session_state.children.at[idx, 'mother_id'] = opt_m['id']
                            st.session_state.children.at[idx, 'status'] = 'confirmed'
                            st.session_state.children.at[idx, 'method'] = 'manual_override'
                            st.rerun()

elif selected_tab == "unmapped":
    st.header("Unmapped Workspace Queue")
    st.markdown("Items here yielded low operational correlation parameters and require validation.")
    
    unmapped_items = children_df[children_df['status'] == 'unmapped']
    if unmapped_items.empty:
        st.success("Unmapped ledger tracking clear! Everything resolved. 🎉")
    else:
        uc1, uc2 = st.columns([2, 3])
        with uc1:
            selected_child_idx = st.radio(
                "Select Child Record Targets", 
                unmapped_items.index, 
                format_func=lambda x: f"{unmapped_items.at[x, 'country']} | {unmapped_items.at[x, 'desc']}"
            )
        with uc2:
            c_data = children_df.loc[selected_child_idx]
            st.metric("Source SKU System Reference Code", c_data['code'])
            st.text_area("Original String Asset", c_data['desc'], disabled=True)
            
            target_mother = st.selectbox(
                "Assign To Target Mother", 
                mothers_df['id'].tolist(), 
                format_func=lambda x: f"{mothers_df[mothers_df['id']==x].iloc[0]['desc']} ({mothers_df[mothers_df['id']==x].iloc[0]['code']})"
            )
            audit_note = st.text_input("Enter Audit Trail Rationale Description (Required)")
            
            if st.button("Execute Binding Override Link", type="primary", disabled=not bool(audit_note.strip())):
                st.session_state.children.at[selected_child_idx, 'mother_id'] = str(target_mother)
                st.session_state.children.at[selected_child_idx, 'status'] = 'confirmed'
                st.session_state.children.at[selected_child_idx, 'method'] = 'manual'
                st.session_state.children.at[selected_child_idx, 'note'] = audit_note
                log_event("Manual Binding", f"Linked raw index code {c_data['code']}", "manual", audit_note)
                st.rerun()

elif selected_tab == "conflicts":
    st.header("Conflict Matrix Detector")
    st.markdown("Identifies duplicate code registrations.")
    
    if len(children_df) > 0:
        conn = sqlite3.connect(':memory:')
        children_df.astype(str).to_sql('children', conn, index=False)
        query = """
            SELECT code, count(distinct mother_id) as links 
            FROM children 
            WHERE mother_id IS NOT NULL AND mother_id != 'None'
            GROUP BY code HAVING links > 1
        """
        conflicts_found = pd.read_sql_query(query, conn)
        conn.close()
        
        if conflicts_found.empty:
            st.success("Zero transactional duplicate blocks detected.")
        else:
            st.error(f"Alert: Multi-Mother Code Overlaps Found!")
            st.dataframe(conflicts_found, use_container_width=True)
    else:
        st.info("Awaiting data.")

elif selected_tab == "registry":
    st.header("Master SKU Registry Matrix Index")
    if len(mothers_df) > 0:
        updated_registry = st.data_editor(mothers_df, num_rows="dynamic", use_container_width=True)
        if st.button("Commit Master Catalog Adjustments"):
            st.session_state.mothers = updated_registry
            st.success("Records updated successfully.")
    else:
        st.info("Awaiting data.")

elif selected_tab == "audit":
    st.header("Audit Analytics Ledger")
    st.dataframe(pd.DataFrame(st.session_state.audit_trail), use_container_width=True)
    
    confirmed_links = children_df[children_df['status'].isin(['confirmed', 'auto_confirmed'])].copy()
    if not confirmed_links.empty:
        confirmed_links['mother_id'] = confirmed_links['mother_id'].astype(str).str.strip()
        mothers_clean_df = mothers_df.copy()
        mothers_clean_df['id'] = mothers_clean_df['id'].astype(str).str.strip()
        
        export_matrix = confirmed_links.merge(mothers_clean_df, left_on='mother_id', right_on='id', suffixes=('_child', '_canonical'))
        
        xl_io = io.BytesIO()
        with pd.ExcelWriter(xl_io, engine='openpyxl') as wr:
            export_matrix.to_excel(wr, index=False, sheet_name='PVM_Canonical_Mappings')
            
        st.download_button(
            label="Download Complete Canonical Mapping File (Excel)",
            data=xl_io.getvalue(),
            file_name="PVM_Canonical_Export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
