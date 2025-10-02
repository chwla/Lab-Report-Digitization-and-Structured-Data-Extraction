# =============================================================================
# Student Name: Soham Chawla
# Student ID: 2022A7PS0069P
# Module 3: Rule-Based Extraction (Minimal & Accurate)
# =============================================================================

import os
import re
import json
import pandas as pd

def group_tokens_into_lines(df, y_tolerance=20):
    """Group tokens into lines by y-coordinate."""
    lines = []
    df_sorted = df.sort_values(['top', 'left']).reset_index(drop=True)
    
    current_line = []
    current_y = None
    
    for _, row in df_sorted.iterrows():
        if current_y is None or abs(row['top'] - current_y) <= y_tolerance:
            current_line.append(row.to_dict())
            current_y = row['top'] if current_y is None else (current_y + row['top']) / 2
        else:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x['left']))
            current_line = [row.to_dict()]
            current_y = row['top']
    
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['left']))
    
    return lines


def extract_fields(lines):
    """Extract patient demographic fields."""
    fields = {}
    patterns = {
        'Hospital': r'^([A-Z][A-Za-z\s&]+(?:Hospital|Lab|Centre|Clinic))',
        'Name': r'(?:Patient\s+)?Name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?=\s+Patient|\s+Age|\s*$)',
        'Patient ID': r'(?:Patient\s+)?ID\s*[:\-]\s*([A-Z]{2,}\d{4,})(?=\s|$)',
        'Age': r'Age\s*[:\-]\s*(\d{1,3})\s*(?:years?|yrs?)?',
        'Gender': r'(?:Gender|Sex)\s*[:\-]\s*(Male|Female|M|F)(?=\s|$)',
        'Date': r'Date\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        'Doctor': r'Doctor\s*[:\-]?\s*(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    }
    
    for line in lines:
        if sum(t['conf'] for t in line) / len(line) < 70:
            continue
        
        text = ' '.join(t['text'] for t in line)
        
        for field, pattern in patterns.items():
            if field not in fields:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields[field] = {
                        'value': match.group(1).strip(),
                        'confidence': round(sum(t['conf'] for t in line) / len(line), 2)
                    }
    
    return fields


def extract_tests(lines):
    """Extract test results with multiple strategies."""
    tests = []
    seen = set()
    
    # Medical test keywords for validation
    medical_keywords = [
        'hemoglobin', 'hematocrit', 'platelet', 'glucose', 'cholesterol',
        'triglyceride', 'creatinine', 'urea', 'bilirubin', 'protein',
        'albumin', 'ast', 'alt', 'phosphatase', 'rbc', 'wbc', 'hdl', 'ldl',
        'sgot', 'sgpt', 'count', 'fasting'
    ]
    
    # Valid unit patterns
    unit_pattern = r'(mg/dl|g/dl|mmol/l|%|u/l|million/[uμ]l|thousand/[uμ]l|cells/[uμ]l)'
    
    for line in lines:
        if len(line) < 2 or sum(t['conf'] for t in line) / len(line) < 70:
            continue
        
        tokens = [t['text'] for t in line]
        
        # Skip if first token doesn't look like a test name
        if not any(kw in tokens[0].lower() for kw in medical_keywords):
            # Check if first two tokens combined form a test name
            if len(tokens) >= 3:
                combined = f"{tokens[0]} {tokens[1]}".lower()
                if any(kw in combined for kw in medical_keywords):
                    # Multi-word test name
                    test_name = f"{tokens[0]} {tokens[1]}"
                    value = tokens[2]
                    unit = tokens[3] if len(tokens) > 3 else ''
                else:
                    continue
            else:
                continue
        else:
            # Single-word test name
            test_name = tokens[0]
            value = tokens[1]
            unit = tokens[2] if len(tokens) > 2 else ''
        
        # Validate value is numeric (not a reference range)
        if not re.match(r'^\d+\.?\d*$', value):
            continue
        
        # Validate unit if present
        if unit and not re.search(unit_pattern, unit.lower()):
            # Try next token
            if len(tokens) > 3 and re.search(unit_pattern, tokens[3].lower()):
                unit = tokens[3]
            else:
                unit = ''
        
        # Avoid duplicates
        key = test_name.lower()
        if key not in seen:
            seen.add(key)
            tests.append({
                'test_name': test_name,
                'value': value,
                'unit': unit,
                'confidence': round(sum(t['conf'] for t in line) / len(line), 2)
            })
    
    return tests


def process_token_file(csv_path):
    """Process a single token CSV file."""
    try:
        df = pd.read_csv(csv_path)
        df = df[df['conf'] > 30].copy()
        
        if df.empty:
            return {'fields': {}, 'test_results': []}
        
        lines = group_tokens_into_lines(df)
        
        return {
            'fields': extract_fields(lines),
            'test_results': extract_tests(lines)
        }
    except Exception as e:
        print(f"  [Error] {e}")
        return {'fields': {}, 'test_results': []}


def run_extraction_on_folder(tokens_dir, output_dir):
    """Run extraction on all token CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n--- Module 3: Rule-Based Extraction ---")
    
    csv_files = sorted([f for f in os.listdir(tokens_dir) if f.endswith('_tokens.csv')])
    
    if not csv_files:
        print("  No token CSV files found.")
        return
    
    print(f"Found {len(csv_files)} file(s) to process.\n")
    
    for csv_file in csv_files:
        csv_path = os.path.join(tokens_dir, csv_file)
        print(f"  - Processing: {csv_file}")
        
        result = process_token_file(csv_path)
        
        json_file = csv_file.replace('_tokens.csv', '_extracted.json')
        json_path = os.path.join(output_dir, json_file)
        
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        fields_count = len(result['fields'])
        tests_count = len(result['test_results'])
        print(f"    ✓ Extracted {fields_count} field(s), {tests_count} test(s)")
    
    print("\n-> Extraction complete.")