import pandas as pd
import os
from gradio_client import Client, file

# === CONFIG ===
EXCEL_PATH = "sample.xlsx"
REF_AUDIO_PATH = "sample_converted.wav"
REF_TEXT = ""
SAVE_DIR = "tts_outputs"
API_URL = ""
API_NAME = "/basic_tts"

# === SETUP ===
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.read_excel(EXCEL_PATH)
texts = df.iloc[:, 0].tolist()

client = Client(API_URL)
ref_audio = file(REF_AUDIO_PATH)

# === SYNTHESIZE LOOP ===
for idx, text in enumerate(texts):
    try:
        print(f"🔊 Synthesizing {idx}: {text[:60]}...")
        output = client.predict(
            ref_audio,      # Reference audio
            REF_TEXT,       # Reference text spoken in ref audio
            text,           # Target generation text
            False,          # remove_silence
            True,           # randomize_seed
            0.0,            # seed_input
            0.15,           # cross_fade_duration_slider
            32,             # nfe_slider
            1.0,            # speed_slider
            api_name=API_NAME
        )
        # Save audio output
        output_path = os.path.join(SAVE_DIR, f"tts_{idx}.wav")
        with open(output[0], "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Failed on idx={idx}: {e}")

