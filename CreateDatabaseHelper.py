import pandas as pd

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS estimation_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prj_name TEXT,
            prj_type TEXT,
            type_of_estimation TEXT,
            Development_SP REAL,
            Architecture_SP REAL,
            QA_SP REAL,
            Implementation_SP REAL,
            Innovation_SP REAL,
            DevOps REAL,
            total_loe REAL,
            month TEXT,
            domain TEXT,
            date TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS estimation_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            feature_or_scenarios TEXT,
            technology_or_type TEXT,
            module_or_method TEXT,
            complexity TEXT,
            total_SP REAL,
            type TEXT,
            FOREIGN KEY (parent_id) REFERENCES parent(id)
        )
    """)
    conn.commit()

def coalesce(val, default=None):
    return default if (pd.isna(val) or val == "") else val


def load_database(conn,summaryData, devList, qaList,info):
    cursor = conn.cursor()
    
    cursor.execute("""
                        INSERT INTO estimation_summary
                        (prj_name,prj_type, type_of_estimation, Development_SP, Architecture_SP, QA_SP,
                         Implementation_SP, Innovation_SP, DevOps, total_loe, month, domain, date)
                        VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        coalesce(info.get("prj_name")),
                        coalesce(info.get("type")),
                        coalesce(summaryData.get("Choose Type of Estimate:")),
                        coalesce(summaryData.get("Development"), 0),
                        coalesce(summaryData.get("Architecture"), 0),
                        coalesce(summaryData.get("Quality Assurance (QA)"), 0),
                        coalesce(summaryData.get("Implementation Costs"), 0),
                        coalesce(summaryData.get("Innovation"), 0),
                        coalesce(summaryData.get("Dev Ops "), 0),
                        coalesce(summaryData.get("Totals: "), 0),
                        coalesce(info.get("month")),
                        coalesce(info.get("domain")),
                        coalesce(info.get("date")),
                    ))
    parent_id = cursor.lastrowid

 
    # DEV details (expected cols: feature_or_scenarios, technology_or_type, module_or_method, complexity, total_SP)
    for drow in devList:
                        cursor.execute("""
                            INSERT INTO estimation_details
                            (parent_id, feature_or_scenarios, technology_or_type, module_or_method, complexity, total_SP, type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            parent_id,
                            coalesce(drow.get("Feature/Task")),
                            coalesce(drow.get("Technology")),
                            coalesce(drow.get("Module Type")),
                            coalesce(drow.get("System Complexity")),
                            coalesce(drow.get("Total Story Points"), 0),
                            "Development"
                        ))


    # QA details
    for qrow in qaList:
                        cursor.execute("""
                            INSERT INTO estimation_details
                            (parent_id, feature_or_scenarios, technology_or_type, module_or_method, complexity, total_SP, type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            parent_id,
                            coalesce(qrow.get("Scenarios")),
                            coalesce(qrow.get("Test Method")),
                            coalesce(qrow.get("Module Type")),
                            coalesce(qrow.get("System Complexity")),
                            coalesce(qrow.get("Total SP"), 0),
                            "QA"
                        ))

    conn.commit()
