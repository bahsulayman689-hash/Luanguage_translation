import os
import requests
from fastapi import FastAPI
from google import genai

app = FastAPI()

# Primary Engine Setup
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Low-resource languages threshold mapping
LOW_RESOURCE_LANGS = {"wo", "qu", "bm", "ast", "yo", "gd"}

@app.post("/translate")
async def translate(text: str, target_lang: str, mode: str = "auto"):
    # Target is low-resource or explicit compare mode requested
    if target_lang in LOW_RESOURCE_LANGS or mode == "compare":
        return await handle_dual_engine(text, target_lang)
    
    # Primary flow: Gemini 3.6 Flash
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Translate the following text into target language '{target_lang}'. Output ONLY the final translation:\n\n{text}"
    )
    return {"engine": "gemini-3.6-flash", "translation": response.text.strip()}

async def handle_dual_engine(text: str, target_lang: str):
    # Call NLLB Local Endpoint
    nllb_resp = requests.post(
        "http://nllb-service:8080/generate", 
        json={"inputs": text, "parameters": {"src_lang": "eng_Latn", "tgt_lang": target_lang}}
    ).json()
    
    # Call Gemini in parallel
    gemini_resp = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Translate to ISO language '{target_lang}': {text}"
    )

    return {
        "mode": "dual_engine",
        "gemini_output": gemini_resp.text.strip(),
        "nllb_output": nllb_resp.get("generated_text", "")
    }