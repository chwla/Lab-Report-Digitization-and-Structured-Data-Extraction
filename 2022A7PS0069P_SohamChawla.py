# =============================================================================
# Student Name: Soham Chawla
# Student ID: 2022A7PS0069P
# Main pipeline script
# =============================================================================

import os
import shutil

# Corrected import to match the function name in module_three.py
from module_one import process_file_for_ocr
from module_two import run_ocr_on_folder
from module_three import run_extraction_on_folder

if __name__ == "__main__":
    # Define project directories
    INPUT_FOLDER = "input_reports"
    CLEANED_IMAGES_FOLDER = "output_cleaned_images"
    OCR_TOKEN_FOLDER = "output_ocr_tokens"
    EXTRACTION_FOLDER = "output_extracted_data"

    print("=" * 70)
    print("LAB REPORT DIGITIZATION PIPELINE")
    print(f"Student: Soham Chawla (2022A7PS0069P)")
    print("=" * 70)

    # Clean up previous runs
    if os.path.exists(CLEANED_IMAGES_FOLDER): shutil.rmtree(CLEANED_IMAGES_FOLDER)
    if os.path.exists(OCR_TOKEN_FOLDER): shutil.rmtree(OCR_TOKEN_FOLDER)
    if os.path.exists(EXTRACTION_FOLDER): shutil.rmtree(EXTRACTION_FOLDER)
    
    if not os.path.exists(INPUT_FOLDER) or not os.listdir(INPUT_FOLDER):
        print(f"\n❌ Input folder '{INPUT_FOLDER}' is missing or empty.")
    else:
        # === MODULE 1: PREPROCESSING ===
        print("\n" + "=" * 70)
        print("MODULE 1: FILE INPUT & PREPROCESSING")
        print("=" * 70)
        for file_name in os.listdir(INPUT_FOLDER):
            process_file_for_ocr(os.path.join(INPUT_FOLDER, file_name), CLEANED_IMAGES_FOLDER)
        print("\n✓ Module 1 (Preprocessing) complete.")

        # === MODULE 2: OCR & TOKENIZATION ===
        print("\n" + "=" * 70)
        print("MODULE 2: OCR & TOKENIZATION")
        print("=" * 70)
        run_ocr_on_folder(CLEANED_IMAGES_FOLDER, OCR_TOKEN_FOLDER)
        print("\n✓ Module 2 (OCR & Tokenization) complete.")

        # === MODULE 3: RULE-BASED EXTRACTION ===
        print("\n" + "=" * 70)
        print("MODULE 3: RULE-BASED EXTRACTION")
        print("=" * 70)
        
        # Corrected function call to match the imported name
        run_extraction_on_folder(
            tokens_dir=OCR_TOKEN_FOLDER,
            output_dir=EXTRACTION_FOLDER
        )
        print("\n✓ Module 3 (Rule-Based Extraction) complete.")

        # === PIPELINE COMPLETE ===
        print("\n" + "=" * 70)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 70)
        print("\nOutput directories:")
        print(f"  1. Cleaned images: ./{CLEANED_IMAGES_FOLDER}/")
        print(f"  2. OCR tokens:     ./{OCR_TOKEN_FOLDER}/")
        print(f"  3. Extracted JSON: ./{EXTRACTION_FOLDER}/")