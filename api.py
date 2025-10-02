# =============================================================================
# Student Name: Soham Chawla
# Student ID: 2022A7PS0069P
# Module 4: Human-in-the-Loop (HITL) API Backend
# =============================================================================

import os
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Any

app = FastAPI(title="Lab Report Review UI")

# --- FOLDER PATHS ---
EXTRACTION_FOLDER = "output_extracted_data"
CORRECTIONS_FOLDER = "output_corrections"
CONFIRMED_FOLDER = "output_confirmed"
TOKENS_FOLDER = "output_ocr_tokens"

os.makedirs(CORRECTIONS_FOLDER, exist_ok=True)
os.makedirs(CONFIRMED_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

class ReportData(BaseModel):
    fields: Dict[str, Any]
    test_results: List[Dict[str, Any]]

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def serve_reviewer_ui():
    with open("static/reviewer.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/api/reports")
async def get_list_of_reports():
    if not os.path.exists(EXTRACTION_FOLDER):
        return JSONResponse(content={"error": f"Directory not found: {EXTRACTION_FOLDER}"}, status_code=404)
    reports = [f for f in os.listdir(EXTRACTION_FOLDER) if f.endswith('.json')]
    return {"reports": reports}

@app.get("/api/report/{report_name}")
async def get_report_data(report_name: str):
    report_path = os.path.join(EXTRACTION_FOLDER, report_name)
    if not os.path.exists(report_path):
        return JSONResponse(content={"error": "Report not found"}, status_code=404)
    with open(report_path, 'r') as f:
        return json.load(f)

@app.post("/api/save/{report_name}")
async def save_corrected_data(report_name: str, corrected_data: ReportData):
    """Saves corrected data and generates a detailed training file."""
    
    # 1. Save the clean, confirmed JSON data (for final use)
    confirmed_path = os.path.join(CONFIRMED_FOLDER, report_name)
    with open(confirmed_path, 'w') as f:
        json.dump(corrected_data.dict(), f, indent=2)

    # 2. Generate the detailed training file
    # This file links the original OCR tokens to the corrected labels
    source_tokens_filename = report_name.replace('_extracted.json', '_tokens.csv')
    tokens_path = os.path.join(TOKENS_FOLDER, source_tokens_filename)
    
    if not os.path.exists(tokens_path):
        return JSONResponse(content={"error": f"Token file not found: {source_tokens_filename}"}, status_code=404)
        
    # Load original OCR tokens
    tokens_df = pd.read_csv(tokens_path)
    all_tokens = tokens_df.to_dict('records')
    
    # Create the training data structure
    training_data = {
        "source_file": report_name,
        "original_tokens": all_tokens,  # Include all original tokens
        "corrected_labels": corrected_data.dict() # Include the corrected high-level data
    }
    
    correction_filename = f"correction_{report_name}"
    correction_path = os.path.join(CORRECTIONS_FOLDER, correction_filename)
    with open(correction_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    return {"message": f"Successfully saved confirmed report and training data for {report_name}"}