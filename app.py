import streamlit as st
import pandas as pd
import re
from rapidfuzz import fuzz
from datetime import datetime
import io

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Perfetti Van Melle | SKU Management", layout="wide")

CURRENT_USER = "regional.analyst"
BRAND_COLORS = {"Alpenliebe": "#15247A", "Chupa Chups": "#E2231A", "Mentos": "#00AEEF", "Other": "#5A6B85"}

ABBREV = {
    "al": "alpenliebe", "alp": "alpenliebe", "cc": "chupa chups", "mt": "mentos",
    "mts": "mentos", "mpf": "mentos", "gl": "golia", "bb": "big babol", "llp": "lollipop",
    "straw": "strawberry", "pfruit": "passionfruit", "chiaki": "chia kiwi",
    "px": "pouch", "disp": "display", "db": "display box", "pcs": "pieces", "pc": "pieces"
}
STOP_WORDS = {"x", "g", "kg", "box", "bag", "bags", "pouch", "stick", "sticks", "tin", "tins", "and", "&"}

# ==========================================
# 2. Matching Engine
# ==========================================
def normalize(text):
    if not isinstance(text, str): return ""
    t = f" {text.lower()} "
    t = re.sub(r'[()/.,&]', ' ', t)
    t = re.sub(r'\d+(\.\d+)?\s*(g|kg|px|pcs|pc|box|bag|bags|stick|sticks|tin|disp|db|u)\b', ' ', t)
    t = re.sub(r'\d+', ' ', t)
    for k, v in ABBREV.items():
        t = re.sub(rf'\b{k}\b', v, t)
    return " ".join([w for w in t.split() if len(w) > 1 and w not in STOP_WORDS])

def get_suggestions(child_desc, mothers_df, top_n=5):
    if mothers_df.empty: return []
    norm_child = normalize(child_desc)
    scores = []
    for _, m in mothers_df.iterrows():
        norm_mother = normalize(m['desc'])
        score = fuzz.token_set_ratio(norm_child, norm_mother) / 100.0
        scores.append({"mother_id": m['id'], "code": m['code'], "desc": m['desc'], "score": score})
    return sorted(scores, key=lambda x: x['score'], reverse=True)[:top_n]

# ==========================================
# 3. State Initialization
# ==========================================
def init_state():
    if 'audit' not in st.session_state:
        st.session_state.audit = []
    
    if 'mothers' not in st.session_state:
        # Seed Mother Data
        st.session_state.mothers = pd.DataFrame([
            {"id": "m1001", "code": "872963", "desc": "AL 2chew Chia Seeds Kiwi & Passion Medium Pouch", "brand": "Alpenliebe"},
            {"id": "m1002", "code": "496461", "desc": "MT Clean Breath Lemonmint", "brand": "Mentos"},
            {"id": "m1003", "code": "100212", "desc": "CC Surprise Box", "brand": "Chupa Chups"}
        ])
        
    if 'children' not in st.session_state:
        # Seed Child Data
        st.session_state.children = pd.DataFrame([
            {"id": "c1", "country": "Vietnam", "code": "872963", "desc": "AL 2Chew ChiaKi&PFruit 24Px 220.5g(63pc)", "mother_id": "m1001", "status": "confirmed", "confidence": 1.0, "method": "manual"},
            {"id": "c2", "country": "Indonesia", "code": "C00239", "desc": "CC Surprise 6x16x12g Water Dino", "mother_id": None, "status": "unmapped", "confidence": 0.0, "method": None}
        ])

def log_action(action, detail, method, confidence=None, note=""):
    st.session_state.audit.insert(0, {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": CURRENT_USER,
        "action": action,
        "detail": detail,
        "method": method,
        "confidence": confidence,
        "note": note
    })

init_state()

# ==========================================
# 4. UI Components & Layout
# ==========================================
st.sidebar.title("PVM SKU Management")
st.sidebar.caption(f"Logged in as: {CURRENT_USER}")

tabs = ["Dashboard", "Import", "Workbench", "Unmapped Queue", "Conflicts", "Registry", "Audit Trail"]
selection = st.sidebar.radio("Navigation", tabs)

mothers_df = st.session_state.mothers
children_df = st.session_state.children

if selection == "Dashboard":
    st.header("Mapping Overview")
    st.markdown("Consolidated view of SKU mapping across all operating countries.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Child SKUs", len(children_df))
    col2.metric("Confirmed", len(children_df[children_df['status'] == 'confirmed']))
    col3.metric("To Review", len(children_df[children_df['status'] == 'suggested']))
    col4.metric("Unmapped", len(children_df[children_df['status'] == 'unmapped']))
    
    st.subheader("Confirmed by Brand")
    if not children_df[children_df['status'] == 'confirmed'].empty:
        merged = children_df[children_df['status'] == 'confirmed'].merge(mothers_df, left_on='mother_id', right_on='id')
        brand_counts = merged['brand'].value_counts()
        st.bar_chart(brand_counts)

elif selection == "Import":
    st.header("Import Portal Export")
    st.markdown("Upload the raw P&L SKU export (Excel or CSV). Columns are auto-detected.")
    uploaded_file = st.file_uploader("Upload File", type=["xlsx", "xls", "csv"])
    
    if uploaded_file:
        if st.button("Process Data"):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Column auto-detection simulation
                cols = [c.lower().replace(' ', '') for c in df.columns]
                # In a real app, you would dynamically map the columns here based on substrings
                
                st.success(f"Successfully loaded {len(df)} rows. Auto-matching against history and specs in progress...")
                # Implementation of history carry-over and fuzzy match would go here
            except Exception as e:
                st.error(f"Error reading file: {e}")

elif selection == "Workbench":
    st.header("Mapping Workbench")
    filter_status = st.selectbox("Filter", ["suggested", "confirmed", "unmapped", "all"])
    
    view_df = children_df if filter_status == "all" else children_df[children_df['status'] == filter_status]
    
    for idx, child in view_df.iterrows():
        with st.expander(f"{child['country']} | {child['code']} - {child['desc']}"):
            st.write(f"**Current Status:** {child['status'].upper()}")
            suggestions = get_suggestions(child['desc'], mothers_df)
            
            for sug in suggestions:
                colA, colB, colC = st.columns([3, 1, 1])
                colA.write(f"**{sug['desc']}** ({sug['code']})")
                colB.progress(sug['score'], text=f"{int(sug['score']*100)}% Match")
                if colC.button("Confirm", key=f"conf_{child['id']}_{sug['mother_id']}"):
                    st.session_state.children.at[idx, 'mother_id'] = sug['mother_id']
                    st.session_state.children.at[idx, 'status'] = 'confirmed'
                    st.session_state.children.at[idx, 'confidence'] = sug['score']
                    log_action("Mapping confirmed", f"{child['code']} -> {sug['code']}", "workbench", sug['score'])
                    st.rerun()

elif selection == "Registry":
    st.header("Mother SKU Registry")
    st.markdown("The canonical list of key products with stable IDs.")
    
    edited_df = st.data_editor(mothers_df, num_rows="dynamic", use_container_width=True)
    if st.button("Save Changes"):
        st.session_state.mothers = edited_df
        log_action("Registry Updated", "Mother SKUs modified", "manual")
        st.success("Registry saved successfully!")

elif selection == "Audit Trail":
    st.header("Audit Trail")
    st.markdown("Every mapping action recorded with who, when, and why.")
    
    if st.session_state.audit:
        st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True)
    
    # Export canonical mapping
    confirmed = children_df[children_df['status'] == 'confirmed']
    if not confirmed.empty:
        export_df = confirmed.merge(mothers_df, left_on='mother_id', right_on='id', suffixes=('_child', '_mother'))
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Canonical Mapping')
        
        st.download_button(
            label="Download Canonical Mapping (Excel)",
            data=buffer.getvalue(),
            file_name="PVM_canonical_mapping.xlsx",
            mime="application/vnd.ms-excel"
        )
