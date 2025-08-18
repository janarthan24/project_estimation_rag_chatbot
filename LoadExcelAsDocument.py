import os
import re
import pandas as pd
from CreateDatabaseHelper import *

SHEET_CONFIG = {
    "Dev Estimation": {
        "columns": [
            "Feature",
            "# of Units",
            "Technology",
            "Module type",
            "System complexity",
            "Change type",
            "Build Hrs",
            "Code",
            "SIT",
            "Documen",
            "Deploy",
            "Effort",
            "Total story",
        ],
        "header_row": 6  # 0-based index
    },
    "QA Estimation": {
        "columns": [
            "Scenarios",
            "# of Test cases",
            "Module type",
            "System complexity",
            "Reusability",
            "Analysis",
            "Test",
            "Regression",
            "Defect",
            "Total",
        ],
        "header_row": 4
    }
} 
REQUIRED_COLUMNS = [
    "Feature",
    "# of Units"
    "Technology",
    "Module type",
    "System complexity",
    "Change type",
    "Build Hrs",
    "Code",
    "SIT",
    "Documen",
    "Deploy",
    "Effort",
    "Total story",
]
coverSheetStrings=["Choose","Development","Architecture","Quality Assurance","Implementation","Innovation","Dev","Total"]


# ---------- FUNCTIONS ----------
def clean_column_name(name):
    """Normalize column names for matching"""
    return str(name).strip().lower()


def extract_file_info(file_path):
    """
    Extract domain, type, month, year, and date (if present) from a file path like:
    domain/type/month year/file.xml
    Returns a dict with keys: domain, type, month, year, date
    """
    # Normalize path separators
    parts = os.path.normpath(file_path).split(os.sep)
    info = {"domain": None, "type": None, "month": None, "year": None, "date": None}
    if len(parts) >= 4:
        info["domain"] = parts[-4]
        info["type"] = parts[-3]
        # Expecting 'month year' in parts[-2]
        month_year = parts[-2]
        match = re.match(r"([A-Za-z]+)[\s_-]*(\d{4})", month_year)
        if match:
            info["month"] = match.group(1)
            info["year"] = match.group(2)
        # Try to extract date from filename if present (e.g., 2025-07-31_file.xml)
        file_name = parts[-1]
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", file_name)
        if date_match:
            info["date"] = date_match.group(1)
    return info

def load_excel_as_documents(folder_path,conn):
    documents = []
    files_loaded = 0
    for root, _, files in os.walk(folder_path):
     for file in files:
        if file.endswith(".xlsx") or file.endswith(".xlsm"):
            file_path = os.path.join(root, file)
            # consider my file path is like domain/type/month year/file.xml extract date and domain , type, month, year
            info=extract_file_info(file_path)

            
            # Extract date & domain from filename
               # example filename:2025-07-31_domain.xlsm or 2025-07-31_domain_etc.xlsm          
            # Updated regex to handle both 2025-07-31_domain.xlsm and 2025-07-31_domain_etc.xlsm
            match = re.match(r"(\d{4}-\d{2}-\d{2})_([^.]+?)(?:_.*)?\.xlsm", file)
            file_date = match.group(1) if match else "unknown_date"
            file_domain = match.group(2) if match else "unknown_domain"
            print(f"Processing file: {file} (Date: {file_date}, Domain: {file_domain})")
            devList = []
            qaList = []
            for sheet in list(SHEET_CONFIG.keys()):
                tempList=process_sheet(file_path, file, file_date, file_domain, sheet, documents)
                if(sheet == "Dev Estimation"):devList = tempList
                if(sheet == "QA Estimation"):qaList = tempList
            summaryData=process_business_sheet(file_path, "Business Cover Sheet")
            print(f"Business Cover Sheet Data: {summaryData}")
            print(f"Dev Estimation Data: {devList}")
            print(f"QA Estimation Data: {qaList}")
            load_database(conn, summaryData, devList, qaList, info)
            files_loaded += 1
    return files_loaded 
    


def process_business_sheet(file_path, sheet):
    df = pd.read_excel(file_path, sheet_name=sheet, usecols=[0,1])
    # Create the map for filtered rows
    result_map = {}
    for _, row in df.iterrows():
        a_val = str(row.iloc[0])
        b_val = str(row.iloc[1])
        if any(a_val.startswith(s) for s in coverSheetStrings):
            result_map[a_val] = b_val
    return result_map
                

def process_sheet(file_path, file, file_date, file_domain, sheet, documents):
    """Process a single sheet and append documents"""
    df = pd.read_excel(file_path, sheet_name=sheet, header=SHEET_CONFIG[sheet]["header_row"])
    tempList=[]
    # Keep only columns starting with required names
    keep_cols = []
    for col in df.columns:
        col_clean = clean_column_name(col)
        for required in SHEET_CONFIG[sheet]["columns"]:
            if col_clean==required or col_clean.startswith(required.lower()):
                keep_cols.append(col)
                break

    df = df[keep_cols]
    print(f"Columns kept: {keep_cols}")
    # Remove completely empty rows
    df = df.dropna(how="all")

    # Create documents from each row
    for _, row in df.iterrows():
        row_data = []
        tempObj={}
        
        #skip rows for value is empty for column feature
        #
        if (sheet == "Dev Estimation" and pd.isna(row.get("Feature/Task"))) or \
            (sheet == "QA Estimation" and pd.isna(row.get("Scenarios"))):
                continue
        for col in keep_cols:
            val = row[col]
            if pd.notna(val):
               # print(f"col data {file} - {col}: {val}")  # Debug output
                tempObj[col] = val
                row_data.append(f"{col}: {val}")
        if row_data:
            tempList.append(tempObj)
            text = f"Date: {file_date}\nDomain: {file_domain}\n" + "\n".join(row_data)
            #print(f"Document content: {text[:500]}...")  # Debug output
           # documents.append(Document(page_content=text))
    return tempList

