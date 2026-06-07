import streamlit as st
import pandas as pd
import io
import re
import datetime
import sqlite3
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="Perfetti Van Melle | SKU Management System", layout="wide")

# Inject PVM Brand theme via custom HTML/CSS injections
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
# 2. Country-Aware Advanced Matching Engine
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
    },
    "Indonesia": {
        "replacements": {
            "sberry": "strawberry", "tf": "toffee", "cc": "chupa chups", 
            "bb": "big babol", "straw": "strawberry"
        },
        "inferred_attributes": [
            (r"6x16x12g", "surprise box")
        ]
    }
}

GLOBAL_ABBREV = {
    "al": "alpenliebe", "alp": "alpenliebe", "cc": "chupa chups", 
    "mt": "mentos", "mts": "mentos", "gl": "golia"
}

def parse_specs(desc: str) -> dict:
    """Extract structural attributes out of free-text SKU strings."""
    if not desc: return {}
    out = {}
    weight = re.search(r'(\d+(?:[.,]\d+)?)\s*g\b', desc, re.I)
    if weight: out["weight"] = weight.group(1) + "g"
    pcs = re.search(r'(\d+)\s*(?:pcs?|u)\b', desc, re.I)
    if pcs: out["pieces"] = pcs.group(1) + "pcs"
    pack = re.search(r'(\d+)\s*(box|bag|pouch|tin|stick|disp|db|px)', desc, re.I)
    if pack: out["pack"] = pack.group(1) + " " + pack.group(2).lower()
    return out

def advanced_normalize(text: str, country: str) -> str:
    """Applies regional rules followed by semantic token standardization."""
    if not isinstance(text, str): return ""
    t = f" {text.lower()} "
    
    # Clean symbol variations safely before word splitting
    t = t.replace("&", " and ")
    
    # Apply global brand standardization mappings
    for k, v in GLOBAL_ABBREV.items():
        t = re.sub(rf'\b{k}\b', v, t)
        
    # Apply targeted country-specific rule dictionaries if available
    config = COUNTRY_DICTIONARIES.get(country, {"replacements": {}, "inferred_attributes": []})
    for k, v in config["replacements"].items():
        t = re.sub(rf'\b{k}\b', v, t)
            
    # Inject explicit package configuration tags using logic matching patterns (e.g., 45Px -> small pouch)
    for regex, structural_tag in config["inferred_attributes"]:
        if re.search(regex, t):
            t += f" {structural_tag} "
            
    t = re.sub(r'[()\/.,\-\[\]]', ' ', t)
    return " ".join(t.split())

@st.cache_resource
def get_ml_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def run_hybrid_matching(child_desc: str, country: str, mothers_df: pd.DataFrame, top_k=5) -> list:
    """Combines Token Fuzzy Matching with Deep Semantic Embeddings for precise ranking."""
    if mothers_df.empty: return []
    model = get_ml_model()
    
    processed_child = advanced_normalize(child_desc, country)
    processed_mothers = [advanced_normalize(m, country) for m in mothers_df['desc'].tolist()]
    
    # Compute Context Embeddings
    child_vector = model.encode(processed_child, convert_to_tensor=True)
    mother_vectors = model.encode(processed_mothers, convert_to_tensor=True)
    semantic_scores = util.cos_sim(child_vector, mother_vectors)[0].tolist()
    
    candidates = []
    for idx, (_, m_row) in enumerate(mothers_df.iterrows()):
        token_fuzzy = fuzz.token_set_ratio(processed_child, processed_mothers[idx]) / 100.0
        final_score = (0.6 * semantic_scores[idx]) + (0.4 * token_fuzzy)
        
        candidates.append({
            "id": m_row['id'], "code": m_row['code'], "desc": m_row['desc'],
            "brand": m_row['brand'], "score": round(max(0.0, min(1.0, final_score)), 3)
        })
        
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]

# ==========================================
# 3. State Management Configuration
# ==========================================
def seed_initial_state():
    if 'audit_trail' not in st.session_state:
        st.session_state.audit_trail = [
            {"at": "2026-06-07 08:00", "by": "seed.import", "action": "Seed dataset imported", "detail": "3 child SKUs, 4 mother SKUs initialized", "method": "manual", "note": ""}
        ]
    if 'mothers' not in st.session_state:
        st.session_state.mothers = pd.DataFrame([
            {"id": "m1001", "code": "872963", "desc": "AL 2chew Chia Seeds Kiwi & Passion Medium Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "220.5g", "season": "", "promo": False},
            {"id": "m1002", "code": "872964", "desc": "AL 2chew Strawberry and Grape Medium Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "220.5g", "season": "", "promo": False},
            {"id": "m1003", "code": "872961", "desc": "AL 2chew Strawberry and Grape Small Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "84g", "season": "", "promo": False},
            {"id": "m1004", "code": "100212", "desc": "CC Surprise Box", "brand": "Chupa Chups", "packType": "Box", "weight": "12g", "season": "", "promo": False}
        ])
    if 'children' not in st.session_state:
        st.session_state.children = pd.DataFrame([
            {"id": "c1", "country": "Vietnam", "code": "872963", "desc": "AL 2Chew ChiaKi&PFruit 24Px 220.5g(63pc)", "mother_id": "m1001", "status": "confirmed", "confidence": 1.0, "method": "manual", "note": ""},
            {"id": "c2", "country": "Vietnam", "code": "873524", "desc": "AL 2Chew Straw&Grape 45Px19pcsx4.5g", "mother_id": None, "status": "suggested", "confidence": 0.0, "method": None, "note": ""},
            {"id": "c3", "country": "Indonesia", "code": "100212", "desc": "CC Surprise 6x16x12g Changing Faces", "mother_id": None, "status": "unmapped", "confidence": 0.0, "method": None, "note": ""}
        ])

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
    "import": "📥 Import Portal Export",
    "workbench": "🔄 Mapping Workbench",
    "unmapped": "📋 Unmapped Queue",
    "conflicts": "⚠️ Conflict Detection",
    "registry": "🗄️ Mother SKU Registry",
    "audit": "📜 Audit Trail"
}
selected_tab = st.sidebar.radio("Navigation Menu", list(tabs.keys()), format_func=lambda x: tabs[x])

st.sidebar.markdown(f"""
    <div style='margin-top: 40px; padding: 12px; background: #E6F7FE; border-radius: 8px;'>
        <span style='font-size: 11px; color: #15247A; font-weight: 800; text-transform: uppercase;'>Regional Session User</span><br/>
        <strong style='color: #1C2433;'>{CURRENT_USER}</strong>
    </div>
""", unsafe_allow_html=True)

# Shared Memory Dataframes
mothers_df = st.session_state.mothers
children_df = st.session_state.children

# ==========================================
# 5. Application Workspace Page Router Module
# ==========================================

if selected_tab == "dashboard":
    st.markdown("<h1 class='pvm-header'>Mapping Overview</h1>", unsafe_allow_html=True)
    st.markdown("Consolidated metrics of SKU mapping across operating regions.")
    
    # Top Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><h5>Total Records</h5><h2>{len(children_df)}</h2></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><h5 style='color: #1E9E6A;'>Confirmed Mappings</h5><h2>{len(children_df[children_df['status']=='confirmed'])}</h2></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><h5 style='color: #00AEEF;'>Awaiting Review</h5><h2>{len(children_df[children_df['status']=='suggested'])}</h2></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><h5 style='color: #F5A623;'>Unmapped Flags</h5><h2>{len(children_df[children_df['status']=='unmapped'])}</h2></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Global Matrix Completeness Rate")
        total = len(children_df)
        confirmed_count = len(children_df[children_df['status'] == 'confirmed'])
        pct = (confirmed_count / total) * 100 if total > 0 else 0
        st.progress(pct / 100, text=f"{round(pct, 1)}% Confirmed Matrix Rows Ledgered")
        
    with c2:
        st.subheader("Consolidated Aggregations by Brand")
        if not children_df[children_df['status'] == 'confirmed'].empty:
            # Force mapping code columns to string before joining to prevent runtime failures
            df_c_fixed = children_df[children_df['status'] == 'confirmed'].copy()
            df_c_fixed['mother_id'] = df_c_fixed['mother_id'].astype(str).str.strip()
            df_m_fixed = mothers_df.copy()
            df_m_fixed['id'] = df_m_fixed['id'].astype(str).str.strip()
            
            merged = df_c_fixed.merge(df_m_fixed, left_on='mother_id', right_on='id')
            st.bar_chart(merged['brand'].value_counts())
        else:
            st.caption("No confirmed entities available to plot breakdown aggregates.")

elif selected_tab == "import":
    st.header("Import Portal Data")
    st.markdown("Upload transactional datasets extracted from the centralization platform dashboard.")
    
    uploaded_file = st.file_uploader("Upload Excel Workspace Source Document", type=["xlsx", "csv"])
    if uploaded_file:
        if st.button("Initialize Processing Protocol Pipeline", type="primary"):
            try:
                # Type safe casting file buffer parsing logic
                if uploaded_file.name.endswith('.csv'):
                    raw_uploaded_df = pd.read_csv(uploaded_file)
                else:
                    raw_uploaded_df = pd.read_excel(uploaded_file)
                
                # Align string schema signatures
                for c in raw_uploaded_df.columns:
                    raw_uploaded_df[c] = raw_uploaded_df[c].astype(str).str.strip()
                
                st.success(f"Successfully processed {len(raw_uploaded_df)} target source data records.")
            except Exception as e:
                st.error(f"Inference exception parsing error encountered: {e}")

elif selected_tab == "workbench":
    st.header("Mapping Workbench Engine")
    st.markdown("Verify high-confidence computer recommendations alongside matching spec profiles.")
    
    f_status = st.selectbox("Filter Current Queue State Matrix", ["suggested", "confirmed", "unmapped", "all"])
    search_query = st.text_input("Search code indexes or keyword context descriptions...")
    
    view_df = children_df if f_status == "all" else children_df[children_df['status'] == f_status]
    if search_query:
        view_df = view_df[view_df['desc'].str.contains(search_query, case=False) | view_df['code'].str.contains(search_query)]
        
    for idx, c_row in view_df.iterrows():
        specs = parse_specs(c_row['desc'])
        current_mother = mothers_df[mothers_df['id'] == str(c_row['mother_id'])] if c_row['mother_id'] else None
        
        with st.expander(f"🌐 {c_row['country']} | Code ID: {c_row['code']} — {c_row['desc']}"):
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown("**Parsed Spec Values:**")
                st.json(specs)
                if current_mother is not None and not current_mother.empty:
                    st.info(f"🔒 **Assigned Parent Node:** {current_mother.iloc[0]['desc']} ({current_mother.iloc[0]['code']})")
                else:
                    st.warning("⚠️ Status: Currently Standing As An Unmapped Entity Node")
                    
            with cols[1]:
                st.markdown("**🤖 Top Computed Similarity Metrics:**")
                recs = run_hybrid_matching(c_row['desc'], c_row['country'], mothers_df)
                
                for r in recs:
                    sub_cols = st.columns([3, 1])
                    sub_cols[0].write(f"**{r['desc']}** ({r['code']})")
                    # Render progress bar for confidence matrix visibility
                    sub_cols[0].progress(float(r['score']), text=f"Match Score Confidence: {int(r['score']*100)}%")
                    if sub_cols[1].button("Link Node", key=f"wb_btn_{c_row['id']}_{r['id']}"):
                        st.session_state.children.at[idx, 'mother_id'] = str(r['id'])
                        st.session_state.children.at[idx, 'status'] = 'confirmed'
                        st.session_state.children.at[idx, 'confidence'] = float(r['score'])
                        st.session_state.children.at[idx, 'method'] = 'fuzzy'
                        log_event("Confirmed Matching Target Row", f"Linked code {c_row['code']} -> {r['code']}", "fuzzy")
                        st.success(f"Link target relationship index saved successfully.")
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
            st.markdown("### Staged Unmapped Queues")
            selected_child_idx = st.radio(
                "Select Child Record Targets", 
                unmapped_items.index, 
                format_func=lambda x: f"{unmapped_items.at[x, 'country']} | {unmapped_items.at[x, 'desc']}"
            )
        with uc2:
            c_data = children_df.loc[selected_child_idx]
            st.markdown(f"### Manual Binding Parameter Dashboard")
            st.metric("Source SKU System Reference Code", c_data['code'])
            st.text_area("Original Platform String Narrative Asset", c_data['desc'], disabled=True)
            
            target_mother = st.selectbox(
                "Assign To Target Mother SKU Node Reference Link", 
                mothers_df['id'].tolist(), 
                format_func=lambda x: f"{mothers_df[mothers_df['id']==x].iloc[0]['desc']} ({mothers_df[mothers_df['id']==x].iloc[0]['code']})"
            )
            audit_note = st.text_input("Enter Audit Trail Rationale Description (Required Field)")
            
            # Form guardrail validation rule constraint enforcement execution
            is_disabled = not bool(audit_note.strip())
            if st.button("Execute Binding Override Intercept Link", type="primary", disabled=is_disabled):
                st.session_state.children.at[selected_child_idx, 'mother_id'] = str(target_mother)
                st.session_state.children.at[selected_child_idx, 'status'] = 'confirmed'
                st.session_state.children.at[selected_child_idx, 'method'] = 'manual'
                st.session_state.children.at[selected_child_idx, 'note'] = audit_note
                log_event("Manual Override Binding Executed", f"Linked raw index code {c_data['code']}", "manual", audit_note)
                st.success("Transaction item mapping pipeline relationship finalized.")
                st.rerun()

elif selected_tab == "conflicts":
    st.header("Conflict Matrix Detector")
    st.markdown("Identifies overlapping structures across different regional entity systems.")
    
    # SQLite runtime analysis cross-check framework validation loop
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
        st.success("Zero transactional duplicate data structural overlapping blocks detected across operations.")
    else:
        st.error(f"Alert: Multi-Mother Code Allocation Matrix Overlaps Found!")
        st.dataframe(conflicts_found, use_container_width=True)

elif selected_tab == "registry":
    st.header("Mother SKU Registry Matrix Index")
    st.markdown("The authoritative master ledger data catalog interface configuration panel.")
    
    # Cast registry ID schemas explicitly to type strings to neutralize merge errors
    mothers_df['id'] = mothers_df['id'].astype(str).str.strip()
    mothers_df['code'] = mothers_df['code'].astype(str).str.strip()
    
    updated_registry = st.data_editor(mothers_df, num_rows="dynamic", use_container_width=True)
    if st.button("Commit Master Catalog Adjustments Matrix"):
        st.session_state.mothers = updated_registry
        log_event("Master Registry Modification Mutation Event", "Manual database row edits applied", "manual")
        st.success("Data persistence records updated successfully.")

elif selected_tab == "audit":
    st.header("Audit Analytics Ledger")
    st.markdown("Historical transaction record management tracker logging user and processing pipeline operations.")
    
    st.dataframe(pd.DataFrame(st.session_state.audit_trail), use_container_width=True)
    
    # Generate unified downstream export matrix pipelines safely
    confirmed_links = children_df[children_df['status'] == 'confirmed'].copy()
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
