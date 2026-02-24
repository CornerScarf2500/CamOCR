# Same converter UI (run with ./run.sh or run.bat then python converter.py)
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import gradio as gr

def ensure_deps():
    for p in ["gradio", "paddle2onnx"]:
        try: __import__(p)
        except: subprocess.check_call([sys.executable, "-m", "pip", "install", p])
ensure_deps()

def convert_to_onnx(model_zip, model_type):
    if model_zip is None:
        return "Upload ZIP first", None
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(model_zip, 'r') as z: z.extractall(tmp)
        pdmodel = next(Path(tmp).rglob("*.pdmodel"), None)
        pdiparams = next(Path(tmp).rglob("*.pdiparams"), None)
        if not pdmodel or not pdiparams:
            return "ZIP must contain .pdmodel + .pdiparams", None
        out_name = f"{model_type}_{datetime.now().strftime('%H%M%S')}.onnx"
        out_path = os.path.join(tempfile.gettempdir(), out_name)
        cmd = ["paddle2onnx", "--model_dir", str(pdmodel.parent), "--model_filename", pdmodel.name,
               "--params_filename", pdiparams.name, "--save_file", out_path, "--opset_version", "11"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            return "✅ Converted!", out_path
        except Exception as e:
            return f"❌ Error: {e}", None

with gr.Blocks(title="CamOCR Converter") as demo:
    gr.Markdown("# 🛠 Paddle → ONNX Converter\nDrag your Kaggle model ZIP")
    zip_in = gr.File(label="Model ZIP", file_types=[".zip"])
    mtype = gr.Dropdown(["det","rec","cls"], value="rec", label="Type")
    btn = gr.Button("Convert", variant="primary")
    status = gr.Textbox()
    out = gr.File()
    btn.click(convert_to_onnx, [zip_in, mtype], [status, out])

demo.launch(server_port=7861)
