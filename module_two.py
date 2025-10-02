# =============================================================================
# Student Name: Soham Chawla
# Student ID: 2022A7PS0069P
# Module 2: OCR & Tokenization
# =============================================================================

import os
import pytesseract
import pandas as pd
from PIL import Image

def perform_ocr_on_image(image_path):
    """
    Performs OCR on a single cleaned image to extract detailed data for each
    recognized word (token).
    
    This function uses Tesseract's image_to_data method, which provides
    rich information about each word.
    
    Args:
        image_path (str): The path to the preprocessed image file.
        
    Returns:
        pandas.DataFrame: A DataFrame containing token-level data, including
                          text, bounding box coordinates, and confidence. 
                          Returns None if OCR fails.
    """
    try:
        # Use Tesseract to get a dictionary of detailed OCR data.
        ocr_data_dict = pytesseract.image_to_data(Image.open(image_path), 
                                                  output_type=pytesseract.Output.DICT)
        
        # Convert the dictionary to a pandas DataFrame for easy manipulation.
        df = pd.DataFrame(ocr_data_dict)
        
        # --- Data Cleaning and Preparation ---
        
        # 1. Remove rows with placeholder confidence values (-1), which represent
        #    structural information (like page or block numbers) rather than words.
        tokens_df = df[df.conf != -1].copy()
        
        # 2. Convert relevant columns to the correct numeric types.
        for col in ['left', 'top', 'width', 'height', 'conf']:
            tokens_df[col] = pd.to_numeric(tokens_df[col], errors='coerce')
            
        # 3. Filter out tokens that are just empty spaces or have no visible text.
        tokens_df = tokens_df[tokens_df['text'].str.strip().astype(bool)]
        
        # 4. Remove any rows with NaN values in critical columns
        tokens_df = tokens_df.dropna(subset=['conf', 'text', 'left', 'top', 'width', 'height'])
        
        # 5. Select and reorder the essential columns for our project.
        final_df = tokens_df[['conf', 'text', 'left', 'top', 'width', 'height']].copy()
        
        # 6. Reset index for cleaner output
        final_df.reset_index(drop=True, inplace=True)
        
        return final_df

    except Exception as e:
        print(f"  [Error] OCR process failed for {os.path.basename(image_path)}: {e}")
        return None


def run_ocr_on_folder(cleaned_images_dir, ocr_output_dir):
    """
    Iterates through a folder of cleaned images, performs OCR on each, and
    saves the resulting token data as a CSV file for each page.
    """
    if not os.path.exists(ocr_output_dir):
        os.makedirs(ocr_output_dir)
        
    print(f"\n--- Starting Module 2: OCR & Tokenization ---")
    print(f"Reading images from: '{cleaned_images_dir}'")

    # Get all image files (png, jpg, jpeg)
    image_files = sorted([f for f in os.listdir(cleaned_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("  No cleaned images found to process.")
        return
    
    print(f"Found {len(image_files)} image(s) to process.\n")
        
    for image_name in image_files:
        image_path = os.path.join(cleaned_images_dir, image_name)
        print(f"  - Processing: {image_name}")
        
        token_data = perform_ocr_on_image(image_path)
        
        if token_data is not None and not token_data.empty:
            # Define the output path for the CSV file.
            base_name = os.path.splitext(image_name)[0]
            output_csv_path = os.path.join(ocr_output_dir, f"{base_name}_tokens.csv")
            
            # Save the DataFrame to a CSV file.
            token_data.to_csv(output_csv_path, index=False)
            print(f"    -> Saved {len(token_data)} tokens to: {output_csv_path}")
        else:
            print(f"    -> No tokens extracted from {image_name}")
            
    print("\n-> Finished OCR processing.")