# =============================================================================
# Student Name: Sohaam Chawla
# Student ID: 2022A7PS0069P
# Module 1: Preprocessing
# =============================================================================

import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

def fix_page_orientation(image):
    """Corrects page orientation and minor skew."""
    try:
        osd_data = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotation = osd_data['rotate']
        if rotation != 0:
            if rotation == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            print(f"    - Applied {rotation}° rotation based on OSD")
    except Exception as e:
        print(f"  [Warning] OSD failed: {e}. Proceeding with skew correction only.")

    # Skew correction on the original (non-inverted) image
    coords = np.column_stack(np.where(image < 128))  # Find dark pixels
    
    if len(coords) == 0:
        print("    - No content detected for skew correction")
        return image
    
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Only apply correction if angle is significant (> 0.5 degrees)
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        print(f"    - Applied {angle:.2f}° skew correction")
        return deskewed
    
    return image

def process_file_for_ocr(input_path, output_dir):
    """Main function to handle a single file and prepare it for OCR."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    images = []

    if file_ext == '.pdf':
        print(f"-> Converting PDF: {base_name}.pdf")
        try:
            pil_images = convert_from_path(input_path, dpi=300)
            for pil_img in pil_images:
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                images.append(img)
        except Exception as e:
            print(f"  [Error] Failed to convert PDF: {e}")
            return
            
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        print(f"-> Loading image: {os.path.basename(input_path)}")
        img = cv2.imread(input_path)
        if img is None:
            print(f"  [Error] Failed to load image: {input_path}")
            return
        images.append(img)
    else:
        print(f"  [Error] Unsupported file type: {file_ext}. Skipping.")
        return

    for i, img in enumerate(images):
        print(f"  - Processing page {i + 1}...")
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Fix orientation and skew
        oriented_img = fix_page_orientation(gray_img)
        
        # Denoise
        denoised_img = cv2.medianBlur(oriented_img, 3)
        
        # Apply adaptive thresholding for better results on varied lighting
        _, final_img = cv2.threshold(denoised_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save the processed image
        output_filename = f"{base_name}_page_{i+1:02d}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, final_img)
        print(f"    Saved to: {output_path}")

    print(f"-> Finished processing {os.path.basename(input_path)}.")