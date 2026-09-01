# 🌐 Hybrid Multimodal Translation System

A resilient, low-latency translation application built with **Streamlit**, routing high-resource global translations to **Gemini 3.6 Flash** and low-resource (West African/Regional) translations to a dual-engine workflow involving **NLLB-200**.

---

## 📌 Features

* **Smart Routing:** High-resource languages (Spanish, German, French) route directly to Gemini 3.6 Flash.
* **Dual-Engine Comparison:** Low-resource languages (Wolof, Fula, Mandinka, Jola, Serer) trigger side-by-side Gemini + NLLB output.
* **Resilient API Handling:** Exponential backoff and retries for temporary `503 UNAVAILABLE` capacity spikes.
* **FLORES-200 Mapping:** Native conversion to 4-letter language-script codes (e.g., `wol_Latn`, `fuc_Latn`).
* **Feedback Logging:** SQLite database (`translation_feedback.db`) logs output votes for evaluation.

---

## 🚀 Quick Start (Local Run)

### 1. Prerequisites
* Python 3.10 or higher installed.

### 2. Installation & Configuration

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
