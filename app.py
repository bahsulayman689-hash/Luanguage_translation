import os
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import streamlit as st
from google import genai

# 1. Load environment variables from local .env file
load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY", "")
print(f"Loaded GEMINI_API_KEY: {key[:4]}...{len(key)}")  # Debugging line

st.set_page_config(
    page_title="Hybrid Translation Engine", page_icon="🌐", layout="wide"
)

DB_FILE = "translation_feedback.db"

# Updated low-resource language codes set
# Updated set of low-resource codes requiring Dual-Engine Routing
LOW_RESOURCE_LANGS = {
    "wo",
    "qu",
    "bm",
    "ast",
    "yo",
    "gd",
    "ff",
    "dyo",
    "mnk",
    "srr",
    "snk",
    "kri",
    "ha",
    "ig",
    "sw",
}

LANGUAGE_TIERS = {
    # High-Resource Global
    "Spanish (es)": {
        "code": "es",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    "German (de)": {
        "code": "de",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    "French (fr)": {
        "code": "fr",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    "Portuguese (pt)": {
        "code": "pt",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    "Arabic (ar)": {
        "code": "ar",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    "Mandarin (zh)": {
        "code": "zh",
        "tier": "High-Resource",
        "engine": "Gemini 3.6 Flash",
    },
    # Low-Resource West & East African
    "Wolof (wo)": {
        "code": "wo",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Mandinka (mnk)": {
        "code": "mnk",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Fula / Pulaar (ff)": {
        "code": "ff",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Jola / Fogny (dyo)": {
        "code": "dyo",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Serer (srr)": {
        "code": "srr",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Soninke (snk)": {
        "code": "snk",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Bambara (bm)": {
        "code": "bm",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Krio (kri)": {
        "code": "kri",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Hausa (ha)": {
        "code": "ha",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Yoruba (yo)": {
        "code": "yo",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
    "Swahili (sw)": {
        "code": "sw",
        "tier": "Low-Resource",
        "engine": "Dual-Engine (Gemini + NLLB)",
    },
}

# 2. Check key existence & initialize SDK
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
  st.error("❌ `GEMINI_API_KEY` was not found. Please add it to your `.env` file.")
  st.stop()

gemini_client = genai.Client(api_key=api_key)

# ==========================================
# 3. DATABASE UTILITIES
# ==========================================
def init_db():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_text TEXT NOT NULL,
            target_language TEXT NOT NULL,
            language_tier TEXT NOT NULL,
            gemini_output TEXT NOT NULL,
            nllb_output TEXT NOT NULL,
            selected_engine TEXT NOT NULL
        )
    """)
  conn.commit()
  conn.close()


def log_feedback(
    source_text, target_lang, tier, gemini_out, nllb_out, selected_engine
):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO feedback (timestamp, source_text, target_language, language_tier, gemini_output, nllb_output, selected_engine)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
      (
          datetime.now().isoformat(),
          source_text,
          target_lang,
          tier,
          gemini_out,
          nllb_out,
          selected_engine,
      ),
  )
  conn.commit()
  conn.close()


def fetch_summary():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "SELECT selected_engine, COUNT(*) FROM feedback GROUP BY selected_engine"
  )
  stats = dict(cursor.fetchall())
  conn.close()
  return stats


init_db()


# ==========================================
# 4. TRANSLATION EXECUTION LOGIC
# ==========================================
# import time


def call_gemini(text: str, target_lang: str) -> str:
  """Calls Gemini with automatic retry backoff and model fallback on 503 errors."""
  prompt = (
      f"Translate into target language '{target_lang}'. Output ONLY the final"
      f" translation:\n\n{text}"
  )

  # Model chain: Primary target first, fallback target second
  models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash"]

  for model_name in models_to_try:
    # Retry up to 3 times per model for 503 capacity spikes
    for attempt in range(3):
      try:
        response = gemini_client.models.generate_content(
            model=model_name, contents=prompt
        )
        return response.text.strip()
      except Exception as e:
        err_str = str(e)
        # If model is unavailable (503), pause briefly and retry
        if "503" in err_str or "UNAVAILABLE" in err_str:
          if attempt < 2:
            time.sleep(1.5 * (attempt + 1))  # Wait 1.5s, then 3.0s
            continue

        # If retries failed on primary model, break inner loop to try fallback model
        break

  return "[Gemini Error: High API demand (503). Please try clicking translate again in a few seconds.]"

def call_nllb(text: str, target_lang: str) -> str:
  try:
    resp = requests.post(
        "http://localhost:8080/generate",
        json={
            "inputs": text,
            "parameters": {"src_lang": "eng_Latn", "tgt_lang": target_lang},
        },
        timeout=3,
    ).json()
    return resp.get("generated_text", "")
  except Exception:
    return f"[NLLB Offline] Fallback translation active for '{target_lang}'."


# ==========================================
# 5. STREAMLIT UI
# ==========================================
st.title("🌐 Hybrid Translation Architecture")
st.caption("Gemini 3.6 Flash (Primary Engine) + NLLB-200 (Low-Resource Fallback)")
st.divider()

# Sidebar
st.sidebar.header("Routing Settings")
mode = st.sidebar.radio(
    "Strategy",
    options=["Auto-Route (Smart Fallback)", "Dual-Engine Compare (Side-by-Side)"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Evaluation Metrics")
stats = fetch_summary()
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Gemini Wins", stats.get("Gemini 3.6 Flash", 0))
col_b.metric("NLLB Wins", stats.get("NLLB-200", 0))

# Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
  st.subheader("Input Text")
  selected_lang_name = st.selectbox(
      "Target Language", list(LANGUAGE_TIERS.keys())
  )
  lang_info = LANGUAGE_TIERS[selected_lang_name]

  if lang_info["tier"] == "High-Resource":
    st.badge(f"⚡ {lang_info['tier']} — Default: {lang_info['engine']}", icon="✅")
  else:
    st.badge(f"🛡️ {lang_info['tier']} — Dual Engine Triggered", icon="⚠️")

  source_text = st.text_area(
      "Enter text to translate:",
      height=160,
      value="May the sun shine brightly on your home today.",
  )

  translate_btn = st.button(
      "Translate Text", type="primary", use_container_width=True
  )

with col2:
  st.subheader("Results & Evaluation")

  if translate_btn and source_text.strip():
    with st.spinner("Executing translation request..."):
      start_time = time.time()

      target_code = lang_info["code"]
      is_low_resource = target_code in LOW_RESOURCE_LANGS
      is_compare_mode = "Dual-Engine" in mode or is_low_resource

      if is_compare_mode:
        gemini_res = call_gemini(source_text, target_code)
        nllb_res = call_nllb(source_text, target_code)
        latency = round((time.time() - start_time) * 1000, 2)

        st.session_state["active_tx"] = {
            "mode": "dual",
            "source_text": source_text,
            "target_lang": selected_lang_name,
            "tier": lang_info["tier"],
            "gemini_out": gemini_res,
            "nllb_out": nllb_res,
            "latency": latency,
        }
      else:
        gemini_res = call_gemini(source_text, target_code)
        latency = round((time.time() - start_time) * 1000, 2)

        st.session_state["active_tx"] = {
            "mode": "single",
            "output": gemini_res,
            "latency": latency,
        }

  if "active_tx" in st.session_state:
    tx = st.session_state["active_tx"]
    st.caption(f"Processing Latency: `{tx['latency']} ms`")

    if tx["mode"] == "single":
      st.success("Primary Engine Executed (Gemini 3.6 Flash)")
      st.text_area("Translation Output", value=tx["output"], height=140)

    else:
      st.info("Dual Engine Compare — Evaluate & Vote")
      comp_c1, comp_c2 = st.columns(2)

      with comp_c1:
        st.markdown("### ♊ Gemini 3.6 Flash")
        st.text_area(
            "Gemini Output", value=tx["gemini_out"], height=100, key="g_out"
        )
        if st.button(
            "👍 Select Gemini", key="vote_gemini", use_container_width=True
        ):
          log_feedback(
              tx["source_text"],
              tx["target_lang"],
              tx["tier"],
              tx["gemini_out"],
              tx["nllb_out"],
              "Gemini 3.6 Flash",
          )
          st.success("Vote logged to SQLite DB!")
          st.rerun()

      with comp_c2:
        st.markdown("### 🌐 NLLB-200")
        st.text_area(
            "NLLB Output", value=tx["nllb_out"], height=100, key="n_out"
        )
        if st.button(
            "👍 Select NLLB", key="vote_nllb", use_container_width=True
        ):
          log_feedback(
              tx["source_text"],
              tx["target_lang"],
              tx["tier"],
              tx["gemini_out"],
              tx["nllb_out"],
              "NLLB-200",
          )
          st.success("Vote logged to SQLite DB!")
          st.rerun()