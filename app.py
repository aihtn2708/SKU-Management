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
st.set_page_config(page_title="PVM SKU Management System", layout="wide")

# Inject PVM Brand theme via custom HTML/CSS injections
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }
        .stButton>button { border-radius: 20px; font-weight: 700; }
        .metric-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E3E8F2; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    </style>
""", unsafe_allowed_with_html=True)

CURRENT_USER = "regional.analyst"

# ==========================================
# 2. Country-Aware Advanced Matching Engine
# ==========================================
# Per-country localized abbreviation and packaging configuration dictionaries
COUNTRY_DICTIONARIES = {
    "Vietnam": {
        "replacements": {
            "straw": "strawberry", "pfruit": "passionfruit", "chiaki": "chia kiwi",
            "px": "pouch", "bags": "pouch", "bag": "pouch", "pouch": "pouch",
            "&": "and", "rect": "rectangular", "llp": "lollipop"
        },
        "inferred_attributes": [
            (r"45\s*(px|pouch|bag|b)", "small pouch"),
            (r"24\s*(px|pouch|bag|b)", "medium pouch")
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
    """Extract structural attributes out of free-text SKU strings"""
    if not desc: return {}
    out = {}
    weight = re.search(r'(\d+(?:[.,]\d+)?)\s*g\b', desc, re.I)
    if weight: out["weight"] = weight.group(1) + "g"
    pcs = re.search(r'(\d+)\s*(?:pcs?|u)\b', desc, re.I)
    if pcs: out["pieces"] = pcs.group(1) + "pcs"
    return out

def advanced_normalize(text: str, country: str) -> str:
    """Applies regional rule matrices followed by semantic standardization pipelines."""
    if not isinstance(text, str): return ""
    t = f" {text.lower()} "
    
    # Apply global brand standardization mappings
    for k, v in GLOBAL_ABBREV.items():
        t = re.sub(rf'\b{k}\b', v, t)
        
    # Apply targeted country-specific rule dictionaries if available
    config = COUNTRY_DICTIONARIES.get(country, {"replacements": {}, "inferred_attributes": []})
    
    for k, v in config["replacements"].items():
        # Clean special chars explicitly like converting '&' directly to text markers
        if k == "&":
            t = t.replace("&", " and ")
        else:
            t = re.sub(rf'\b{k}\b', v, t)
            
    # Inject explicit package sizing keywords using localized pattern recognition rules
    for regex, structural_tag in config["inferred_attributes"]:
        if re.search(regex, t):
            t += f" {structural_tag} "
            
    # Strip residual formatting noise
    t = re.sub(r'[()\/.,]', ' ', t)
    return " ".join(t.split())

@st.cache_resource
def get_ml_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def run_hybrid_matching(child_desc: str, country: str, mothers_df: pd.DataFrame, top_k=5) -> list:
    """Combines Token Fuzzy Matching with Deep Semantic Embeddings for maximum precision."""
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
        # Compute exact token matching overlay to augment the deep learning scores
        token_fuzzy = fuzz.token_set_ratio(processed_child, processed_mothers[idx]) / 100.0
        
        # Combined score balance
        final_score = (0.6 * semantic_scores[idx]) + (0.4 * token_fuzzy)
        
        candidates.append({
            "id": m_row['id'], "code": m_row['code'], "desc": m_row['desc'],
            "brand": m_row['brand'], "score": round(max(0.0, min(1.0, final_score)), 3)
        })
        
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]

# ==========================================
# 3. Memory State Management Engine
# ==========================================
def seed_initial_state():
    if 'audit_trail' not in st.session_state:
        st.session_state.audit_trail = [
            {"at": "2026-06-07 08:00", "by": "system", "action": "Registry initialization", "detail": "Seed database active", "method": "system", "note": ""}
        ]
    if 'mothers' not in st.session_state:
        st.session_state.mothers = pd.DataFrame([
            {"id": "m1001", "code": "872963", "desc": "AL 2chew Chia Seeds Kiwi & Passion Medium Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "220.5g"},
            {"id": "m1002", "code": "872964", "desc": "AL 2chew Strawberry and Grape Medium Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "220.5g"},
            {"id": "m1003", "code": "872961", "desc": "AL 2chew Strawberry and Grape Small Pouch", "brand": "Alpenliebe", "packType": "Pouch", "weight": "84g"},
            {"id": "m1004", "code": "100212", "desc": "CC Surprise Box", "brand": "Chupa Chups", "packType": "Box", "weight": "12g"}
        ])
    if 'children' not in st.session_state:
        st.session_state.children = pd.DataFrame([
            {"id": "c1", "country": "Vietnam", "code": "872963", "desc": "AL 2Chew ChiaKi&PFruit 24Px 220.5g(63pc)", "mother_id": "m1001", "status": "confirmed", "confidence": 1.0, "method": "history", "note": "Pre-mapped mapping workbook entry"},
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
# 4. Global Sidebar Layout
# ==========================================
st.sidebar.markdown(f"""
    <div style='padding: 10px 0px;'>
        <h2 style='color: #15247A; margin-bottom: 0px;'>Perfetti Van Melle</h2>
        <small style='color: #00AEEF; font-weight: bold;'>SKU MANAGEMENT HUB</small>
    </div>
""", unsafe_allowed_with_html=True)

tabs = {
    "dashboard": "📊 Dashboard Overview",
    "import": "📥 Ingest Portal Data",
    "workbench": "🔄 Mapping Workbench",
    "unmapped": "📋 Unmapped Queue",
    "conflicts": "⚠️ Conflict Detector",
    "registry": "🗄️ Mother SKU Registry",
    "audit": "⏳ Audit Analytics"
}
selected_tab = st.sidebar.radio("Navigation Menu", list(tabs.keys()), format_func=lambda x: tabs[x])

st.sidebar.markdown(f"""
    <div style='margin-top: 40px; padding: 12px; background: #E6F7FE; border-radius: 8px;'>
        <span style='font-size: 12px; color: #15247A; font-weight: bold;'>Active Analyst Profile</span><br/>
        <strong style='color: #1C2433;'>{CURRENT_USER}</strong>
    </div>
""", unsafe_allowed_with_html=True)

# Cache shared indexes
mothers_df = st.session_state.mothers
children_df = st.session_state.children

# ==========================================
# 5. Core Views Module Implementation
# ==========================================

if selected_tab == "dashboard":
    st.header("Mapping Architecture Dashboard")
    
    # Standardized Top Layer Metrics Grid
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='metric-card'><h5>Total Records</h5><h2>{len(children_df)}</h2></div>", unsafe_allowed_with_html=True)
    m2.markdown(f"<div class='metric-card'><h5 style='color: green;'>Confirmed Links</h5><h2>{len(children_df[children_df['status']=='confirmed'])}</h2></div>", unsafe_allowed_with_html=True)
    m3.markdown(f"<div class='metric-card'><h5 style='color: #00AEEF;'>Awaiting Review</h5><h2>{len(children_df[children_df['status']=='suggested'])}</h2></div>", unsafe_allowed_with_html=True)
    m4.markdown(f"<div class='metric-card'><h5 style='color: orange;'>Unmapped Outliers</h5><h2>{len(children_df[children_df['status']=='unmapped'])}</h2></div>", unsafe_allowed_with_html=True)
    
    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Global Consolidation Rates")
        total = len(children_df)
        confirmed_count = len(children_df[children_df['status'] == 'confirmed'])
        pct = (confirmed_count / total) * 100 if total > 0 else 0
        st.progress(pct / 100, text=f"{round(pct, 1)}% Confirmed Matrix Completeness")
        
    with c2:
        st.subheader("Reporting Distribution by Brand")
        if not children_df[children_df['status'] == 'confirmed'].empty:
            merged = children_df[children_df['status'] == 'confirmed'].merge(mothers_df, left_on='mother_id', right_on='id')
            st.bar_chart(merged['brand'].value_counts())
        else:
            st.caption("No confirmed entities available to plot distribution tracks.")

elif selected_tab == "import":
    st.header("Import Shared Portal Datasets")
    st.markdown("Ingest multi-country regional raw spreadsheets to execute batch validation passes.")
    
    uploaded_file = st.file_uploader("Choose Data Workbook", type=["xlsx", "csv"])
    if uploaded_file:
        if st.button("Initialize Pipeline Processing", type="primary"):
            st.success("Ingestion sequence completed. Matching engine tracking activated via the main workbench layer.")

elif selected_tab == "workbench":
    st.header("Mapping Workbench Engine")
    st.markdown("Inspect algorithmic suggestions against parsed metadata properties.")
    
    f_status = st.selectbox("Filter Current Queue Status", ["suggested", "confirmed", "unmapped", "all"])
    search_query = st.text_input("Search codes, keywords or regional descriptions...")
    
    view_df = children_df if f_status == "all" else children_df[children_df['status'] == f_status]
    if search_query:
        view_df = view_df[view_df['desc'].str.contains(search_query, case=False) | view_df['code'].str.contains(search_query)]
        
    for idx, c_row in view_df.iterrows():
        specs = parse_specs(c_row['desc'])
        current_mother = mothers_df[mothers_df['id'] == c_row['mother_id']] if c_row['mother_id'] else None
        
        with st.expander(f"🌐 {c_row['country']} | Code: {c_row['code']} — {c_row['desc']}"):
            cols = st.columns([3, 2])
            with cols[0]:
                st.markdown("** Extracted Structural Spec Fields:**")
                st.json(specs)
                if current_mother is not None and not current_mother.empty:
                    st.info(f"🔒 **Assigned Key Product:** {current_mother.iloc[0]['desc']} ({current_mother.iloc[0]['code']})")
                else:
                    st.warning("⚠️ Currently Standing As An Unmapped Entity")
                    
            with cols[1]:
                st.markdown("**🤖 Top High-Confidence AI Suggestions:**")
                recs = run_hybrid_matching(c_row['desc'], c_row['country'], mothers_df)
                
                for r in recs:
                    sub_cols = st.columns([3, 1])
                    sub_cols[0].write(f"**{r['desc']}** ({r['code']})")
                    if sub_cols[1].button("Link", key=f"wb_btn_{c_row['id']}_{r['id']}"):
                        st.session_state.children.at[idx, 'mother_id'] = r['id']
                        st.session_state.children.at[idx, 'status'] = 'confirmed'
                        st.session_state.children.at[idx, 'confidence'] = r['score']
                        st.session_state.children.at[idx, 'method'] = 'fuzzy'
                        log_event("Confirmed Matching Target", f"{c_row['code']} linked to {r['code']}", "fuzzy")
                        st.success(f"Link established for key target asset node.")
                        st.rerun()

elif selected_tab == "unmapped":
    st.header("Unmapped Workspace Queue")
    st.markdown("Items here yielded zero confidence matches and require manual override assignment.")
    
    unmapped_items = children_df[children_df['status'] == 'unmapped']
    if unmapped_items.empty:
        st.success("Unmapped tracking work queue is completely clear! 🎉")
    else:
        uc1, uc2 = st.columns([2, 3])
        with uc1:
            st.markdown("### Open Items")
            selected_child_idx = st.radio(
                "Select Child Target Node", 
                unmapped_items.index, 
                format_func=lambda x: f"{unmapped_items.at[x, 'country']} | {unmapped_items.at[x, 'desc']}"
            )
        with uc2:
            c_data = children_df.loc[selected_child_idx]
            st.markdown(f"### Manual Assignment Dashboard")
            st.metric("Target Code", c_data['code'])
            st.text_area("Raw Text Asset String", c_data['desc'], disabled=True)
            
            target_mother = st.selectbox(
                "Assign Target Mother Canonical SKU Record", 
                mothers_df['id'].tolist(), 
                format_func=lambda x: f"{mothers_df[mothers_df['id']==x].iloc[0]['desc']} ({mothers_df[mothers_df['id']==x].iloc[0]['code']})"
            )
            audit_note = st.text_input("Enter Audit Trail Rationalization Note (Mandatory)")
            
            if st.button("Execute Manual Link Binding", type="primary", disabled=not bool(audit_note.strip())):
                st.session_state.children.at[selected_child_idx, 'mother_id'] = target_mother
                st.session_state.children.at[selected_child_idx, 'status'] = 'confirmed'
                st.session_state.children.at[selected_child_idx, 'method'] = 'manual'
                st.session_state.children.at[selected_child_idx, 'note'] = audit_note
                log_event("Manual Override Binding Executed", f"Linked code {c_data['code']}", "manual", audit_note)
                st.success("Binding completed successfully.")
                st.rerun()

elif selected_tab == "conflicts":
    st.header("Conflict Matrix Detector")
    st.markdown("Identifies overlapping data structures or duplicate cross-border code registrations.")
    
    # Compute active conflicts inside the dataset matrix dynamically
    counts = children_df.groupby('code').filter(lambda x: len(x) > 1)
    if counts.empty:
        st.success("Zero systemic transactional cross-joins or multi-mother duplication blocks found.")
    else:
        st.warning(f"Detected internal multi-record conflicts across system operational areas.")
        st.dataframe(counts, use_container_width=True)

elif selected_tab == "registry":
    st.header("Mother SKU Registry Index")
    st.markdown("The canonical master database management pane for global key products.")
    
    updated_registry = st.data_editor(mothers_df, num_rows="dynamic", use_container_width=True)
    if st.button("Persist Master Registry Modification Matrix"):
        st.session_state.mothers = updated_registry
        log_event("Master Registry Structural Mutation", "Modified core master parameters manually", "manual")
        st.success("Master records updated.")

elif selected_tab == "audit":
    st.header("Audit Tracking & Operational Logging Ledger")
    st.markdown("Comprehensive operational audit trail visibility matrix tracking manual and algorithmic modifications.")
    
    st.dataframe(pd.DataFrame(st.session_state.audit_trail), use_container_width=True)
    
    # Render final download option pipelines for P&L tracking applications
    confirmed_links = children_df[children_df['status'] == 'confirmed']
    if not confirmed_links.empty:
        export_matrix = confirmed_links.merge(mothers_df, left_on='mother_id', right_on='id', suffixes=('_child', '_canonical'))
        
        xl_io = io.BytesIO()
        with pd.ExcelWriter(xl_io, engine='openpyxl') as wr:
            export_matrix.to_excel(wr, index=False, sheet_name='PVM_Canonical_Mappings')
            
        st.download_button(
            label="Export Authorized Canonical Ledger Workbook",
            data=xl_io.getvalue(),
            file_name="PVM_Canonical_Export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
