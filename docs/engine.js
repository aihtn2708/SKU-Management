/* SKU matching engine — browser port of app.py for the mock-up.
 * Normalisation and spec parsing are line-for-line ports of app.py.
 * Similarity is a lightweight stand-in for the MiniLM sentence embeddings
 * (character-trigram cosine blended with token overlap), so the mock-up
 * runs in any browser without downloading a model. Thresholds match app.py.
 */
(function (root) {
  const COUNTRY_DICTIONARIES = {
    Vietnam: {
      replacements: {
        straw: "strawberry", pfruit: "passionfruit", chiaki: "chia kiwi",
        px: "pouch", bags: "pouch", bag: "pouch", pouch: "pouch",
        rect: "rectangular", llp: "lollipop"
      },
      inferred_attributes: [
        [/45\s*(px|pouch|bag|b)\b/, "small pouch"],
        [/24\s*(px|pouch|bag|b)\b/, "medium pouch"]
      ]
    }
  };

  const GLOBAL_ABBREV = {
    al: "alpenliebe", alp: "alpenliebe", cc: "chupa chups",
    mt: "mentos", mts: "mentos", gl: "golia", bb: "big babol"
  };

  const AUTO_CONFIRM = 0.85; // score >= this → auto_confirmed
  const MIN_MATCH = 0.45;    // score <= this → unmapped

  function parseSpecs(desc) {
    if (typeof desc !== "string") return {};
    const out = {};
    const w = desc.match(/(\d+(?:[.,]\d+)?)\s*g\b/i);
    if (w) out.weight = w[1] + "g";
    const p = desc.match(/(\d+)\s*(?:pcs?|u)\b/i);
    if (p) out.pieces = p[1] + "pcs";
    const k = desc.match(/(\d+)\s*(box|bag|pouch|tin|stick|disp|db|px)/i);
    if (k) out.pack = k[1] + " " + k[2].toLowerCase();
    return out;
  }

  function normalize(text, country) {
    if (typeof text !== "string") return "";
    let t = " " + text.toLowerCase() + " ";
    t = t.split("&").join(" and ");
    for (const [k, v] of Object.entries(GLOBAL_ABBREV)) t = t.replace(new RegExp("\\b" + k + "\\b", "g"), v);
    const cfg = COUNTRY_DICTIONARIES[country] || { replacements: {}, inferred_attributes: [] };
    for (const [k, v] of Object.entries(cfg.replacements)) t = t.replace(new RegExp("\\b" + k + "\\b", "g"), v);
    for (const [re, tag] of cfg.inferred_attributes) if (re.test(t)) t += " " + tag + " ";
    t = t.replace(/[()\/.,\-\[\]]/g, " ");
    t = t.replace(/\d+(\.\d+)?\s*(g|kg|px|pcs|pc|box|bag|bags|stick|sticks|tin)\b/g, " ");
    return t.split(/\s+/).filter(Boolean).join(" ");
  }

  function trigrams(s) {
    const m = new Map();
    const p = "  " + s + " ";
    for (let i = 0; i < p.length - 2; i++) {
      const g = p.slice(i, i + 3);
      m.set(g, (m.get(g) || 0) + 1);
    }
    return m;
  }
  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (const [k, v] of a) { na += v * v; if (b.has(k)) dot += v * b.get(k); }
    for (const v of b.values()) nb += v * v;
    return na && nb ? dot / Math.sqrt(na * nb) : 0;
  }
  function dice(a, b) {
    const A = new Set(a.split(" ").filter(Boolean)), B = new Set(b.split(" ").filter(Boolean));
    if (!A.size || !B.size) return 0;
    let n = 0; for (const x of A) if (B.has(x)) n++;
    return (2 * n) / (A.size + B.size);
  }
  function similarity(a, b) {
    return 0.5 * cosine(trigrams(a), trigrams(b)) + 0.5 * dice(a, b);
  }

  function bestOf(vec, pool) {
    let best = -1, idx = -1;
    pool.forEach((p, i) => { const s = similarity(vec, p); if (s > best) { best = s; idx = i; } });
    return [best, idx];
  }

  /* Same three tiers and routing as app.py's "Execute History-Aware ML Pipeline". */
  function runPipeline(mothers, history, children) {
    const historyDict = new Map(history.map(h => [h.child_code, h.mother_code]));
    const motherClean = mothers.map(m => normalize(m.desc));
    const histClean = history.map(h => normalize(h.child_desc));
    return children.map((c, i) => {
      const base = { id: "c_" + i, country: c.country || "Unknown", code: c.code, desc: c.desc };
      if (historyDict.has(c.code)) {
        const m = mothers.find(x => x.code === historyDict.get(c.code));
        return { ...base, mother_id: m ? m.id : null, status: "auto_confirmed", confidence: 1, method: "history", note: "Auto-carried from history" };
      }
      const clean = normalize(c.desc);
      const [mScore, mIdx] = bestOf(clean, motherClean);
      const [hScore, hIdx] = bestOf(clean, histClean);
      let mother_id = null, status = "unmapped", score = 0, method = "";
      if (hScore > mScore && hScore > MIN_MATCH) {
        const m = mothers.find(x => x.code === history[hIdx].mother_code);
        mother_id = m ? m.id : null;
        score = Math.round(hScore * 1000) / 1000;
        method = "fuzzy_via_history";
        status = score < AUTO_CONFIRM ? "suggested" : "auto_confirmed";
      } else if (mScore > MIN_MATCH) {
        mother_id = mothers[mIdx].id;
        score = Math.round(mScore * 1000) / 1000;
        method = "fuzzy_direct";
        status = score < AUTO_CONFIRM ? "suggested" : "auto_confirmed";
      }
      return { ...base, mother_id, status, confidence: score, method, note: "", _best: Math.max(mScore, hScore) };
    });
  }

  /* Port of the SQLite conflict query: codes linked to more than one mother. */
  function findConflicts(children) {
    const by = new Map();
    for (const c of children) {
      if (c.mother_id == null) continue;
      if (!by.has(c.code)) by.set(c.code, new Set());
      by.get(c.code).add(c.mother_id);
    }
    return [...by].filter(([, s]) => s.size > 1).map(([code, s]) => ({ code, links: s.size, mothers: [...s] }));
  }

  const SAMPLE = {
    mothers: [
      { id: "M1001", code: "M1001", desc: "Alpenliebe Strawberry Lollipop 24 Pouch", brand: "Alpenliebe", packType: "pouch", weight: "240g" },
      { id: "M1002", code: "M1002", desc: "Alpenliebe Caramel Milk Candy 45 Pouch", brand: "Alpenliebe", packType: "pouch", weight: "135g" },
      { id: "M1003", code: "M1003", desc: "Chupa Chups Classic Lollipop Strawberry", brand: "Chupa Chups", packType: "single", weight: "12g" },
      { id: "M1004", code: "M1004", desc: "Mentos Mint Roll", brand: "Mentos", packType: "roll", weight: "37.5g" },
      { id: "M1005", code: "M1005", desc: "Mentos Fruit Roll", brand: "Mentos", packType: "roll", weight: "37.5g" },
      { id: "M1006", code: "M1006", desc: "Golia Activ Plus Candy Bag", brand: "Golia", packType: "bag", weight: "100g" },
      { id: "M1007", code: "M1007", desc: "Big Babol Strawberry Gum Tin", brand: "Big Babol", packType: "tin", weight: "30g" },
      { id: "M1008", code: "M1008", desc: "Chupa Chups Mini Lollipop Box", brand: "Chupa Chups", packType: "box", weight: "300g" },
      { id: "M1009", code: "M1009", desc: "Alpenliebe Passionfruit Hard Candy", brand: "Alpenliebe", packType: "bag", weight: "120g" },
      { id: "M1010", code: "M1010", desc: "Alpenliebe 2Chew Chia Kiwi Pouch", brand: "Alpenliebe", packType: "pouch", weight: "90g" }
    ],
    history: [
      { child_code: "VN-24001", child_desc: "AL Straw LLP 24px", mother_code: "M1001" },
      { child_code: "VN-24002", child_desc: "AL Caramel milk 45 bag", mother_code: "M1002" },
      { child_code: "VN-30010", child_desc: "MT Mint roll 37.5g", mother_code: "M1004" },
      { child_code: "PH-11020", child_desc: "CC Classic straw lollipop", mother_code: "M1003" },
      { child_code: "VN-31007", child_desc: "BB straw gum tin 30g", mother_code: "M1007" }
    ],
    children: [
      { code: "VN-24001", country: "Vietnam", desc: "AL Straw LLP 24px" },
      { code: "VN-30010", country: "Vietnam", desc: "MT Mint roll 37.5g (14pcs)" },
      { code: "VN-24015", country: "Vietnam", desc: "AL Strawberry Lollipop 24 Pouch new" },
      { code: "VN-24020", country: "Vietnam", desc: "Alpenliebe caramel 45px" },
      { code: "VN-30022", country: "Vietnam", desc: "Mentos fruity 37.5g stick" },
      { code: "VN-41003", country: "Vietnam", desc: "GL activ+ cool 100g bag" },
      { code: "VN-52001", country: "Vietnam", desc: "CC mini llp 50pcs disp" },
      { code: "VN-24033", country: "Vietnam", desc: "AL pfruit hard candy 120g" },
      { code: "VN-24040", country: "Vietnam", desc: "AL 2chew chiaki 24px" },
      { code: "VN-90011", country: "Vietnam", desc: "Promo display stand Tet 2027" },
      { code: "VN-90012", country: "Vietnam", desc: "Q4 gift set mixed SKUs" },
      { code: "PH-11021", country: "Philippines", desc: "Chupa Chups classic strawberry 12g" },
      { code: "TH-70031", country: "Thailand", desc: "Big Babol straw tin 30g" },
      { code: "VN-24015", country: "Vietnam", desc: "AL caramel milk 45 bag promo" }
    ]
  };

  root.SKUEngine = { COUNTRY_DICTIONARIES, GLOBAL_ABBREV, AUTO_CONFIRM, MIN_MATCH, parseSpecs, normalize, similarity, runPipeline, findConflicts, SAMPLE };
})(typeof window !== "undefined" ? window : globalThis);
