import streamlit as st
import pandas as pd
import sqlite3
import io
import re
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="SKU ML Batch Mapper", layout="wide")

# ==========================================
# 1. ML Model Initialization (Cached)
# ==========================================
@st.cache_resource
def load_model():
    # all-MiniLM-L6-v2 is lightweight and highly accurate for short text/names
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ==========================================
# 2. Text Normalization 
# ==========================================
# Domain abbreviations for FMCG to help the ML understand the context
ABBREV = {
    "al": "alpenliebe", "alp": "alpenliebe", "cc": "chupa chups", "mt": "mentos",
    "mts": "mentos", "gl": "golia", "bb": "big babol", "llp": "lollipop",
    "straw": "strawberry", "pfruit": "passionfruit", "chiaki": "chia kiwi",
    "px": "pouch", "pcs": "pieces", "pc": "pieces"
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[()/.,&]', ' ', text)
    # Strip weights/packs so the ML focuses purely on product identity
    text = re.sub(r'\d+(\.\d+)?\s*(g|kg|px|pcs|pc|box|bag|bags|stick|sticks|tin)\b', ' ', text)
    words = text.split()
    expanded = [ABBREV.get(w, w) for w in words]
    return " ".join(expanded)

# ==========================================
# 3. Application UI
# ==========================================
st.title("⚡ SKU ML Batch Mapper (One-Time Process)")
st.markdown("Upload your Mother SKU registry and Child SKUs. The system uses an in-memory SQLite database and ML embeddings to find the best match, then exports the joined result.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Mother SKUs (Target Registry)")
    mother_file = st.file_uploader("Upload Mother SKUs (Excel/CSV)", type=['xlsx', 'csv'], key="mother")

with col2:
    st.subheader("2. Child SKUs (To Be Mapped)")
    child_file = st.file_uploader("Upload Child SKUs (Excel/CSV)", type=['xlsx', 'csv'], key="child")

if mother_file and child_file:
    if st.button("🚀 Run ML Matching Pipeline", type="primary"):
        with st.spinner("Processing files and running ML embeddings..."):
            
            # --- Step A: Read Files ---
            df_mother = pd.read_excel(mother_file) if mother_file.name.endswith('.xlsx') else pd.read_csv(mother_file)
            df_child = pd.read_excel(child_file) if child_file.name.endswith('.xlsx') else pd.read_csv(child_file)
            
            # Standardize column names dynamically
            mother_desc_col = next((c for c in df_mother.columns if 'desc' in str(c).lower() or 'name' in str(c).lower()), df_mother.columns[0])
            mother_code_col = next((c for c in df_mother.columns if 'code' in str(c).lower() or 'id' in str(c).lower()), df_mother.columns[1])
            
            child_desc_col = next((c for c in df_child.columns if 'desc' in str(c).lower() or 'name' in str(c).lower()), df_child.columns[0])
            child_code_col = next((c for c in df_child.columns if 'code' in str(c).lower() or 'id' in str(c).lower()), df_child.columns[1])

            # --- Step B: ML Embedding & Matching ---
            st.toast("Generating embeddings...", icon="🧠")
            
            # Clean text
            mother_clean = df_mother[mother_desc_col].apply(clean_text).tolist()
            child_clean = df_child[child_desc_col].apply(clean_text).tolist()
            
            # Encode
            mother_embs = model.encode(mother_clean, convert_to_tensor=True)
            child_embs = model.encode(child_clean, convert_to_tensor=True)
            
            # Calculate Cosine Similarity Matrix
            cosine_scores = util.cos_sim(child_embs, mother_embs)
            
            # Extract Top 1 match for each child
            best_scores, best_indices = torch.max(cosine_scores, dim=1)
            
            df_child['predicted_mother_code'] = [df_mother.iloc[i.item()][mother_code_col] for i in best_indices]
            df_child['confidence_score'] = [round(s.item() * 100, 1) for s in best_scores]
            
            # --- Step C: In-Memory SQLite Join ---
            st.toast("Executing SQL Join...", icon="🗄️")
            conn = sqlite3.connect(':memory:')
            
            # Dump to SQLite
            df_mother.to_sql('mother_skus', conn, index=False, if_exists='replace')
            df_child.to_sql('child_skus', conn, index=False, if_exists='replace')
            
            # Execute relational join
            query = f"""
                SELECT 
                    c.*,
                    m."{mother_desc_col}" as predicted_mother_desc
                FROM child_skus c
                LEFT JOIN mother_skus m 
                ON c.predicted_mother_code = m."{mother_code_col}"
                ORDER BY c.confidence_score DESC
            """
            
            final_mapped_df = pd.read_sql_query(query, conn)
            conn.close() # DB is destroyed here, satisfying the zero-storage requirement
            
            # --- Step D: Display & Export ---
            st.success("Mapping Complete!")
            
            st.dataframe(
                final_mapped_df.head(50), 
                use_container_width=True,
                column_config={"confidence_score": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100)}
            )
            
            # Prepare Excel Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                final_mapped_df.to_excel(writer, index=False, sheet_name='ML_Mapped_SKUs')
            
            st.download_button(
                label="⬇️ Download Full Mapped Results (Excel)",
                data=buffer.getvalue(),
                file_name="Automated_SKU_Mappings.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
