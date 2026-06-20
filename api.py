from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import glob
from vector_inference import ejecutar_vectorial

app = FastAPI(title="DXF Inference API")

# Allow CORS for the frontend (dxf-viewer typically runs on localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), "best_models")

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    zones: str = Form("[]")
):
    zonas_list = json.loads(zones)
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        diferencial_path = os.path.join(BEST_MODELS_DIR, "best_interruptor_diferencial.pt")
        model_paths = [diferencial_path] if os.path.exists(diferencial_path) else []
        
        if not model_paths:
            return {"error": "No valid models found or selected"}
            
        # Run vector inference
        conteo, detecciones = ejecutar_vectorial(
            dxf_path=tmp_path,
            modelos_paths=model_paths,
            zonas=zonas_list or None,
            save_slices=False
        )
        
        # Build BOM
        global_bom = dict(conteo)
        
        # Build BOM per zone if zones exist
        zones_bom = {z['name']: {} for z in zonas_list}
        for d in detecciones:
            cls = d['clase']
            for z_name in d.get('zonas', []):
                if z_name in zones_bom:
                    zones_bom[z_name][cls] = zones_bom[z_name].get(cls, 0) + 1
                    
        return {
            "global_bom": global_bom,
            "zones_bom": zones_bom
        }
        
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
