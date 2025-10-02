# =============================================================================
# Student Name: Soham Chawla
# Student ID: 2022A7PS0069P
# Module 3: Complete Rule-Based Extraction with Validation
# =============================================================================

import os
import re
import json
import pandas as pd

# ============================================================================
# CONFIGURATION: Medical Test Patterns and Reference Data
# ============================================================================

# Multi-word test name patterns (order matters - most specific first)
MEDICAL_TEST_PATTERNS = {
    'RBC Count': ['rbc', 'count'],
    'WBC Count': ['wbc', 'count'],
    'Platelet Count': ['platelet', 'count'],
    'Total Cholesterol': ['total', 'cholesterol'],
    'HDL Cholesterol': ['hdl', 'cholesterol'],
    'LDL Cholesterol': ['ldl', 'cholesterol'],
    'Total Bilirubin': ['total', 'bilirubin'],
    'Alkaline Phosphatase': ['alkaline', 'phosphatase'],
    'Total Protein': ['total', 'protein'],
    'Glucose Fasting': ['glucose', 'fasting'],
    'SGOT AST': ['sgot', 'ast'],
    'SGPT ALT': ['sgpt', 'alt'],
}

# Single-word test names
SINGLE_WORD_TESTS = {
    'hemoglobin', 'hematocrit', 'triglycerides', 'creatinine', 
    'urea', 'albumin', 'glucose'
}

# Expected units for validation and auto-correction
EXPECTED_UNITS = {
    'Hemoglobin': 'g/dL',
    'RBC Count': 'million/μL',
    'WBC Count': 'thousand/μL',
    'Platelet Count': 'thousand/μL',
    'Hematocrit': '%',
    'Glucose': 'mg/dL',
    'Glucose Fasting': 'mg/dL',
    'Total Cholesterol': 'mg/dL',
    'HDL Cholesterol': 'mg/dL',
    'LDL Cholesterol': 'mg/dL',
    'Triglycerides': 'mg/dL',
    'Creatinine': 'mg/dL',
    'Urea': 'mg/dL',
    'Total Bilirubin': 'mg/dL',
    'SGOT AST': 'U/L',
    'SGPT ALT': 'U/L',
    'Alkaline Phosphatase': 'U/L',
    'Total Protein': 'g/dL',
    'Albumin': 'g/dL',
}

# Expected value ranges (for validation)
EXPECTED_VALUE_RANGES = {
    'Hemoglobin': (5, 20),
    'RBC Count': (2.0, 8.0),
    'WBC Count': (2.0, 20.0),
    'Platelet Count': (50, 600),
    'Hematocrit': (20, 60),
    'Glucose Fasting': (40, 300),
    'Total Cholesterol': (100, 400),
    'HDL Cholesterol': (20, 100),
    'LDL Cholesterol': (50, 250),
    'Triglycerides': (30, 500),
    'Creatinine': (0.3, 3.0),
    'Urea': (10, 80),
    'Total Bilirubin': (0.1, 5.0),
    'SGOT AST': (5, 200),
    'SGPT ALT': (5, 200),
    'Alkaline Phosphatase': (20, 300),
    'Total Protein': (4.0, 10.0),
    'Albumin': (2.0, 6.0),
}


# ============================================================================
# CORE FUNCTIONS: Token Processing
# ============================================================================

def group_tokens_into_lines(df, y_tolerance=20):
    """
    Group tokens into lines based on vertical proximity (y-coordinate).
    
    Args:
        df: DataFrame with columns [text, left, top, width, height, conf]
        y_tolerance: Maximum vertical distance to group tokens in same line
    
    Returns:
        List of lines, where each line is a list of token dictionaries
    """
    lines = []
    df_sorted = df.sort_values(['top', 'left']).reset_index(drop=True)
    
    current_line = []
    current_y = None
    
    for _, row in df_sorted.iterrows():
        if current_y is None or abs(row['top'] - current_y) <= y_tolerance:
            current_line.append(row.to_dict())
            # Update running average of y-coordinate
            current_y = row['top'] if current_y is None else (current_y + row['top']) / 2
        else:
            # Start new line
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x['left']))
            current_line = [row.to_dict()]
            current_y = row['top']
    
    # Add final line
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['left']))
    
    return lines


# ============================================================================
# FIELD EXTRACTION: Patient Demographics
# ============================================================================

def extract_fields(lines):
    """
    Extract patient demographic fields using regex patterns.
    
    Args:
        lines: List of token lines from group_tokens_into_lines()
    
    Returns:
        Dictionary of extracted fields with confidence scores
    """
    fields = {}
    
    patterns = {
        'Hospital': r'([A-Z][A-Za-z\s&]+(?:Hospital|Centre|Center|Clinic))',
        'Name': r'(?:Patient\s+)?Name\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?=\s+Patient\s+ID|\s+Age|\s*$)',
        'Patient ID': r'(?:Patient\s+)?ID\s*[:\-]\s*([A-Z]{2,}\d{4,})(?=\s|$)',
        'Age': r'Age\s*[:\-]\s*(\d{1,3})\s*(?:years?|yrs?)?(?=\s|$)',
        'Gender': r'(?:Gender|Sex)\s*[:\-]\s*(Male|Female|M|F)(?=\s|$)',
        'Date': r'Date\s*[:\-]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})(?=\s|$)',
        'Doctor': r'Doctor\s*[:\-]?\s*(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?=\s|$)',
    }
    
    for line in lines:
        avg_conf = sum(t['conf'] for t in line) / len(line) if line else 0
        
        # Skip low confidence lines
        if avg_conf < 70:
            continue
        
        text = ' '.join(t['text'] for t in line)
        
        for field, pattern in patterns.items():
            if field not in fields:  # Don't overwrite already found fields
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields[field] = {
                        'value': match.group(1).strip(),
                        'confidence': round(avg_conf, 2)
                    }
    
    return fields


# ============================================================================
# TEST EXTRACTION: Medical Test Results
# ============================================================================

def match_test_name(tokens_lower, start_idx=0):
    """
    Match test name from tokens starting at given index.
    Uses context-aware multi-word matching.
    
    Args:
        tokens_lower: List of lowercase token strings
        start_idx: Index to start matching from
    
    Returns:
        tuple: (test_name, tokens_consumed) or (None, 0) if no match
    """
    # Try multi-word patterns first (more specific)
    for full_name, keywords in MEDICAL_TEST_PATTERNS.items():
        if start_idx + len(keywords) <= len(tokens_lower):
            match = True
            for i, keyword in enumerate(keywords):
                if keyword not in tokens_lower[start_idx + i]:
                    match = False
                    break
            
            if match:
                return full_name, len(keywords)
    
    # Try single-word tests
    if start_idx < len(tokens_lower):
        token = tokens_lower[start_idx]
        for test_name in SINGLE_WORD_TESTS:
            if test_name in token:
                return test_name.capitalize(), 1
    
    return None, 0


def extract_reference_range(tokens, start_idx):
    """
    Extract reference range from tokens after the unit.
    Handles formats like: "12.0 - 15.5", "< 200", "> 40"
    
    Args:
        tokens: List of token strings
        start_idx: Index to start searching from
    
    Returns:
        Normalized reference range string or None
    """
    ref_parts = []
    i = start_idx
    
    # Look ahead up to 6 tokens
    while i < len(tokens) and i < start_idx + 6:
        token = tokens[i]
        
        # Match numeric values, operators, and ranges
        if re.match(r'^[\d\.]+$', token):
            ref_parts.append(token)
        elif token in ['<', '>', '-', '–', '—', 'to']:
            ref_parts.append('-' if token in ['-', '–', '—', 'to'] else token)
        elif re.match(r'^\d+\.?\d*\s*[-–]\s*\d+\.?\d*$', token):
            # Complete range in single token (e.g., "12.0-15.5")
            return re.sub(r'\s*[-–]\s*', ' - ', token)
        elif ref_parts:
            # Stop if we've started collecting and hit non-range token
            break
        
        i += 1
    
    if not ref_parts:
        return None
    
    # Normalize format: ensure spaces around dashes
    result = ' '.join(ref_parts)
    result = re.sub(r'\s*-\s*', ' - ', result)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result if result else None


def validate_and_fix_test_result(test_result):
    """
    Validate and auto-correct common OCR errors in test results.
    
    Common issues fixed:
    - Missing units (auto-fill from EXPECTED_UNITS)
    - Decimal point errors (RBC: 45 → 4.5)
    - Unit format normalization (uL → μL)
    
    Args:
        test_result: Dictionary with test data
    
    Returns:
        Modified test_result dictionary with corrections and flags
    """
    test_name = test_result['test_name']
    value = test_result['value']
    unit = test_result.get('unit', '')
    
    # Fix 1: Add missing units
    if not unit and test_name in EXPECTED_UNITS:
        test_result['unit'] = EXPECTED_UNITS[test_name]
        test_result['auto_correction'] = 'Added missing unit'
        unit = test_result['unit']
    
    # Fix 2: Normalize unit format
    if unit:
        original_unit = unit
        unit = unit.replace('uL', 'μL').replace('ul', 'μL')
        if unit != original_unit:
            test_result['unit'] = unit
            if 'auto_correction' not in test_result:
                test_result['auto_correction'] = f'Normalized unit: {original_unit} → {unit}'
    
    # Fix 3: Value validation and correction
    try:
        numeric_value = float(value)
        
        # RBC Count: Often OCR reads "4.5" as "45"
        if test_name == 'RBC Count':
            if 40 <= numeric_value <= 60:
                corrected = numeric_value / 10
                test_result['value'] = str(corrected)
                test_result['auto_correction'] = f'Decimal correction: {value} → {corrected}'
        
        # WBC Count: Similar decimal issue
        elif test_name == 'WBC Count':
            if 40 <= numeric_value <= 120:
                corrected = numeric_value / 10
                test_result['value'] = str(corrected)
                test_result['auto_correction'] = f'Decimal correction: {value} → {corrected}'
        
        # Range validation (flag suspicious values)
        if test_name in EXPECTED_VALUE_RANGES:
            min_val, max_val = EXPECTED_VALUE_RANGES[test_name]
            current_val = float(test_result['value'])  # Use potentially corrected value
            
            if current_val < min_val or current_val > max_val:
                test_result['flag'] = 'OUT_OF_EXPECTED_RANGE'
                test_result['expected_range'] = f'{min_val} - {max_val}'
                
    except ValueError:
        test_result['flag'] = 'INVALID_NUMERIC_VALUE'
    
    return test_result


def extract_tests(lines):
    """
    Extract test results from token lines.
    
    Process:
    1. Match test name using context-aware patterns
    2. Extract numeric value
    3. Extract unit
    4. Extract reference range
    5. Validate and auto-correct
    
    Args:
        lines: List of token lines
    
    Returns:
        List of test result dictionaries
    """
    tests = []
    seen = set()
    
    unit_pattern = r'^(mg/dl|g/dl|mmol/l|%|u/l|iu/l|million/[uμ]l|thousand/[uμ]l|cells/[uμ]l)$'
    
    for line in lines:
        if len(line) < 2:
            continue
        
        avg_conf = sum(t['conf'] for t in line) / len(line)
        if avg_conf < 65:
            continue
        
        tokens = [t['text'] for t in line]
        tokens_lower = [t.lower() for t in tokens]
        
        # Skip header rows
        line_text = ' '.join(tokens_lower)
        if ('test' in line_text and 'name' in line_text) or \
           ('reference' in line_text and 'range' in line_text):
            continue
        
        # Match test name
        test_name, consumed = match_test_name(tokens_lower, 0)
        
        if test_name is None:
            continue
        
        # Extract value, unit, and reference range
        value = None
        unit = ""
        ref_range = None
        
        for i in range(consumed, len(tokens)):
            # Look for numeric value
            if re.match(r'^\d+\.?\d*$', tokens[i]):
                value = tokens[i]
                
                # Extract unit (next token)
                if i + 1 < len(tokens):
                    potential_unit = tokens[i + 1]
                    if re.match(unit_pattern, potential_unit.lower()):
                        unit = potential_unit
                        ref_range = extract_reference_range(tokens, i + 2)
                    else:
                        # No unit found, still try reference range
                        ref_range = extract_reference_range(tokens, i + 1)
                
                break
        
        if value is None:
            continue
        
        # Avoid duplicates (case-insensitive)
        key = test_name.lower().replace(' ', '')
        if key not in seen:
            seen.add(key)
            
            test_result = {
                'test_name': test_name,
                'value': value,
                'unit': unit,
                'confidence': round(avg_conf, 2)
            }
            
            if ref_range:
                test_result['reference_range'] = ref_range
            
            # Validate and fix
            test_result = validate_and_fix_test_result(test_result)
            
            tests.append(test_result)
    
    return tests


# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_token_file(csv_path, debug=False):
    """
    Process a single OCR token CSV file.
    
    Args:
        csv_path: Path to CSV file with OCR tokens
        debug: If True, print detailed processing info
    
    Returns:
        Dictionary with 'fields' and 'test_results'
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter low confidence tokens
        df = df[df['conf'] > 30].copy()
        
        if df.empty:
            return {'fields': {}, 'test_results': []}
        
        # Group tokens into lines
        lines = group_tokens_into_lines(df)
        
        if debug:
            print(f"\n  DEBUG: Found {len(lines)} lines")
            for i, line in enumerate(lines[:10]):
                text = ' '.join(t['text'] for t in line)
                avg_conf = sum(t['conf'] for t in line) / len(line)
                print(f"  Line {i}: (conf={avg_conf:.1f}) {text}")
        
        # Extract data
        return {
            'fields': extract_fields(lines),
            'test_results': extract_tests(lines)
        }
        
    except Exception as e:
        print(f"  [Error processing {csv_path}] {e}")
        return {'fields': {}, 'test_results': []}


def merge_multi_page_results(extraction_dir):
    """
    Merge extraction results from multiple pages of the same report.
    Demographics from first page are preserved across all pages.
    
    Args:
        extraction_dir: Directory containing extraction JSON files
    """
    json_files = sorted([f for f in os.listdir(extraction_dir) 
                        if f.endswith('_extracted.json') and '_merged' not in f])
    
    if len(json_files) <= 1:
        return
    
    # Group files by base report name
    report_groups = {}
    for json_file in json_files:
        # Remove page numbers from filename
        base_name = re.sub(r'_page\d+|_\d+', '', 
                          json_file.replace('_extracted.json', ''))
        if base_name not in report_groups:
            report_groups[base_name] = []
        report_groups[base_name].append(json_file)
    
    # Merge multi-page reports
    for base_name, files in report_groups.items():
        if len(files) > 1:
            merged_result = {'fields': {}, 'test_results': []}
            seen_tests = set()
            
            for json_file in sorted(files):
                json_path = os.path.join(extraction_dir, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Use demographic fields from first page with data
                if not merged_result['fields'] and data['fields']:
                    merged_result['fields'] = data['fields']
                
                # Merge test results (avoid duplicates)
                for test in data['test_results']:
                    test_key = test['test_name'].lower().replace(' ', '')
                    if test_key not in seen_tests:
                        seen_tests.add(test_key)
                        merged_result['test_results'].append(test)
            
            # Save merged result
            merged_path = os.path.join(extraction_dir, f"{base_name}_merged.json")
            with open(merged_path, 'w') as f:
                json.dump(merged_result, f, indent=2)
            
            print(f"  ✓ Merged {len(files)} pages → {base_name}_merged.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_extraction_on_folder(tokens_dir, output_dir, debug=False):
    """
    Run extraction pipeline on all token CSV files in a directory.
    
    Args:
        tokens_dir: Directory containing *_tokens.csv files
        output_dir: Directory to save extraction JSON files
        debug: Enable debug output
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*70)
    print("MODULE 3: RULE-BASED EXTRACTION")
    print("="*70)
    
    csv_files = sorted([f for f in os.listdir(tokens_dir) 
                       if f.endswith('_tokens.csv')])
    
    if not csv_files:
        print("  ⚠ No token CSV files found.")
        return
    
    print(f"\nProcessing {len(csv_files)} file(s)...\n")
    
    # Process each file
    for csv_file in csv_files:
        csv_path = os.path.join(tokens_dir, csv_file)
        print(f"  Processing: {csv_file}")
        
        result = process_token_file(csv_path, debug=debug)
        
        # Save extraction result
        json_file = csv_file.replace('_tokens.csv', '_extracted.json')
        json_path = os.path.join(output_dir, json_file)
        
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Print summary
        fields_count = len(result['fields'])
        tests_count = len(result['test_results'])
        corrections = sum(1 for t in result['test_results'] 
                         if 'auto_correction' in t)
        flags = sum(1 for t in result['test_results'] if 'flag' in t)
        
        print(f"    ✓ {fields_count} fields, {tests_count} tests", end='')
        if corrections:
            print(f", {corrections} auto-corrected", end='')
        if flags:
            print(f", {flags} flagged", end='')
        print()
    
    # Merge multi-page results
    print("\n" + "-"*70)
    print("Merging multi-page results...")
    print("-"*70)
    merge_multi_page_results(output_dir)
    
    print("\n" + "="*70)
    print("✓ MODULE 3 COMPLETE")
    print("="*70)
    print(f"\nExtraction results saved to: {output_dir}/")
    print("\nReady for Module 4: Human-in-the-Loop validation")