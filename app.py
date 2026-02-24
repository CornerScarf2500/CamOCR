import os
from datetime import datetime
import zipfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gradio as gr
from rapidocr_onnxruntime import RapidOCR

IS_PUBLIC = bool(os.getenv("SPACE_ID"))  # True only on Hugging Face Spaces

print(f"🌍 Running on {os.name} / Platform: {os.uname().machine if hasattr(os, 'uname') else 'Windows'}")

COLLECTED_DIR = "collected_data"
os.makedirs(COLLECTED_DIR, exist_ok=True)

ocr = RapidOCR(use_angle_cls=True)

def process_image(image, allow_training=True):
    if image is None:
        return "Please upload an image", None, "Nothing processed"
    
    result, _ = ocr(image)
    lines = [f"{line[1]} (conf: {line[2]:.2f})" for line in result]
    full_text = "\n".join(lines)
    
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for box, text, score in result:
        pts = np.array(box).astype(np.int32)
        draw.polygon(pts.tolist(), outline="red", width=3)
    
    if not IS_PUBLIC or allow_training:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(COLLECTED_DIR, f"img_{ts}.jpg")
        img.save(img_path)
        with open(os.path.join(COLLECTED_DIR, f"text_{ts}.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)
        status = f"✅ Saved for training! ({len([f for f in os.listdir(COLLECTED_DIR) if f.endswith('.jpg')])} files)"
    else:
        status = "✅ Processed — data NOT saved (you disabled training)"
    
    return full_text, img, status

def download_zip():
    zip_path = "collected_data.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        for root, _, files in os.walk(COLLECTED_DIR):
            for file in files:
                z.write(os.path.join(root, file), file)
    return zip_path

TC = """# Terms & Conditions - CamOCR v0.1

**License**: Apache License 2.0 (free, open-source, commercial use allowed)

**Data Collection**
• Local run (your PC): All uploads automatically saved to `collected_data/` folder (private, only you see it).
• Public web (Hugging Face): Images saved **ONLY** if the checkbox is ticked. Uncheck anytime.
• All data is anonymous. Used only to train future versions of this open-source project.
• You can download everything anytime with one click.

**Privacy & Disclaimer**
• No tracking, no ads, no data selling.
• Provided "AS IS". Test with base model now, improve with your data later.
• Full source: https://github.com/CornerScarf2500/CamOCR

© 2026 CornerScarf2500 — Open Source for everyone ❤️"""

with gr.Blocks(title="CamOCR", theme=gr.themes.Soft(), css=".gradio-container{max-width:1100px;margin:auto;}") as demo:
    gr.HTML("<h1 style='text-align:center;color:#0ea5e9'>🌟 CamOCR v0.1</h1><p style='text-align:center'>Local Hybrid OCR • Mixed languages • Handwriting • Auto data collection</p>")
    
    with gr.Tab("📸 Live OCR"):
        with gr.Row():
            img_in = gr.Image(label="Upload image or drag & drop", type="numpy", height=520)
            with gr.Column():
                btn = gr.Button("🔥 Run OCR", variant="primary", size="large")
                text_out = gr.Textbox(label="✅ Extracted Text", lines=15)
                vis_out = gr.Image(label="📍 Detection Boxes (red)", height=520)
    
    with gr.Tab("📚 Contribute Training Data"):
        gr.Markdown("### Help train the next version!")
        with gr.Row(visible=IS_PUBLIC):
            allow_box = gr.Checkbox(value=True, label="✅ Allow this image for open-source training", info="Uncheck = keep your image 100% private")
        img_contrib = gr.Image(label="Upload any photo/document", height=400)
        contrib_btn = gr.Button("💾 Process & Save", variant="secondary")
        status_out = gr.Textbox(label="Status")
        dl_btn = gr.Button("📥 Download ALL collected data as ZIP")
        dl_btn.click(download_zip, outputs=gr.File(label="collected_data.zip"))
    
    with gr.Tab("ℹ️ About & Terms"):
        gr.Markdown(TC)
    
    btn.click(process_image, inputs=[img_in, gr.State(True)], outputs=[text_out, vis_out, status_out])
    contrib_btn.click(process_image, inputs=[img_contrib, allow_box if IS_PUBLIC else gr.State(True)], outputs=[status_out, gr.State(), status_out])

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
