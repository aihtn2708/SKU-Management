# SKU-Management

A mock-up of the **SKU Management SKU Mapping Management System**: a Streamlit app that maps each market's local product codes (*child SKUs*) to one canonical global product (*mother SKU*), so sales reported in Vietnam, the Philippines or Thailand roll up to the same line in the consolidated P&L.

**Interactive explainer:** [`docs/index.html`](docs/index.html) — a browser replica of every Streamlit view with sample data, plus a walkthrough of the matching engine. Open the file locally, or enable GitHub Pages (Settings → Pages → Deploy from branch → `main` / `docs`) to serve it at `https://aihtn2708.github.io/SKU-Management/`.

---

## How it works

The app takes three uploads (`.xlsx` or `.csv`):

| # | Upload | Contents | Example row |
|---|--------|----------|-------------|
| 1 | **Master Mother SKUs** | Canonical registry: code, description; optional `id`, `brand`, `packType`, `weight` | `M1001 · Alpenliebe Strawberry Lollipop 24 Pouch` |
| 2 | **Historical Mappings** | Previously approved child → mother links | `VN-24001 · AL Straw LLP 24px → M1001` |
| 3 | **New Unmapped SKUs** | Raw P&L extraction: local code, description, optional country/market | `VN-24015 · AL Strawberry Lollipop 24 Pouch new` |

Columns are located by keyword (`get_col`), so exports don't need renaming:

| Field | Keywords searched |
|-------|-------------------|
| Mother code / description | `code`, `id` / `desc`, `name` |
| History child code / child description / mother | `child`, `code` / `child`, `desc` / `mother`, `target` |
| Child code / description / country | `code`, `id` / `desc`, `name` / `country`, `market` |

### Matching pipeline

```
uploads ─► normalise ─► Tier 1: exact history ──────────► auto_confirmed (1.0)
                    └─► Tier 2: fuzzy via history ─┐
                    └─► Tier 3: fuzzy direct ──────┴─► score ≥ 0.85 → auto_confirmed
                                                       0.45 – 0.85 → suggested
                                                       ≤ 0.45      → unmapped
```

1. **Normalisation** (`advanced_normalize`) lower-cases the description, expands brand abbreviations (`al` → alpenliebe, `cc` → chupa chups, `mt` → mentos, `gl` → golia, `bb` → big babol), applies a country dictionary (`straw` → strawberry, `llp` → lollipop, `px` → pouch…), and strips pack sizes and punctuation so the model compares product identity, not pack size.
2. **Tier 1 – exact history.** A child code already in the history file inherits its old mother with confidence 1.0.
3. **Tier 2 – fuzzy via history.** Descriptions are embedded with `sentence-transformers/all-MiniLM-L6-v2` and compared (cosine) to every past child description. If the best past synonym beats the best mother, that synonym's mother is proposed.
4. **Tier 3 – fuzzy direct.** Otherwise the closest mother description is proposed.
5. **Routing.** Scores ≥ 0.85 are auto-confirmed, 0.45–0.85 wait for review, the rest are flagged unmapped.

`parse_specs` extracts weight, piece count and pack type from the raw description so reviewers can sanity-check each suggestion.

### The seven views

| View | What it does |
|------|--------------|
| 📋 **Dashboard Overview** | Totals for processed, confirmed, awaiting-review and unmapped records. |
| 📥 **Data Ingestion Pipeline** | Three uploaders; once all are present, runs the history-aware ML pipeline. |
| 🔄 **Mapping Workbench** | Filter by status, search, inspect parsed specs and confidence, confirm a suggestion or override it by searching the mother registry (`manual_override`). |
| 📋 **Unmapped Queue** | Pick an unmapped child, assign a mother, and give a required audit rationale (`manual`). |
| ⚠️ **Conflict Detection** | In-memory SQLite query for child codes linked to more than one mother. |
| 🗄️ **Master Registry** | Editable mother catalogue (`st.data_editor`, dynamic rows). |
| 📜 **Audit Ledger** | Time-stamped log of every action, plus an Excel export (`Canonical_Export.xlsx`) joining confirmed children to their mothers. |

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or open the repo in **GitHub Codespaces**: the dev container installs the requirements and starts Streamlit on port 8501 automatically. The first pipeline run downloads the MiniLM model (~90 MB).

## Repository layout

```
app.py                 Streamlit application (all views + matching engine)
requirements.txt       Python dependencies
.devcontainer/         Codespaces / VS Code dev container
docs/index.html        Interactive explainer and mock-up of the app
docs/engine.js         Browser port of the normalisation, routing and conflict logic
```

## About the mock-up

`docs/` runs entirely in the browser with no backend. Normalisation, spec parsing, tier order, thresholds, conflict detection and the export join are ported from `app.py`. The one substitution is similarity: the browser can't load MiniLM, so the mock-up scores with character trigrams plus token overlap. Scores are therefore close to, but not identical to, the real app's. You can replace any of the three sample files with your own CSV.

## Known issues

- **History column detection.** `get_col(df_hist, ['child', 'desc'])` returns the first column containing *either* word. A history file with a `Child Code` column before its description column will use the codes as descriptions, which silently disables Tier 2. Workaround: name the history columns e.g. `Code`, `Description`, `Mother Target`. Fix: search for `desc` first, then fall back to `child`.
- **Country dictionaries are not applied.** The pipeline calls `advanced_normalize(str(x))` without a `country`, so the Vietnam dictionary (`straw`, `llp`, `px`, `pfruit`, `chiaki`…) is never used. Passing the row's country for child descriptions would raise scores for shorthand-heavy markets.
- **Session-only state.** Mappings, registry edits and the audit trail live in `st.session_state` and are lost on refresh. Download the export before closing the tab.
- **"Total Unprocessed"** on the dashboard counts every child row in the latest run, including confirmed ones.
