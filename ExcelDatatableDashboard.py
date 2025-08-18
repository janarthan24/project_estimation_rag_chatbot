import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from AccessCreateDatabaseHelper import create_tables_for_accessDB
from CreateDatabaseHelper import create_tables
from LoadExcelAsDocument import load_excel_as_documents
DB_PATH = "estimation.db"

import pyodbc

# For .mdb or .accdb files
#conn_str = (
#    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
#   r'DBQ=C:\\Users\\janarm\\Downloads\\Database1.accdb;'
#)
#conn = pyodbc.connect(conn_str)
# -----------------------------
# DB bootstrap
# -----------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist
create_tables(conn)
#create_tables_for_accessDB(conn)
# -----------------------------
# Utilities
# -----------------------------
def non_empty_row(sr: pd.Series) -> bool:
    """Return True if at least one non-null, non-empty cell exists in row."""
    for v in sr.values:
        if pd.notna(v) and str(v).strip() != "":
            return True
    return False

def coalesce(val, default=None):
    return default if (pd.isna(val) or val == "") else val

# -----------------------------
# Step 1: Load Excel files into DB
# -----------------------------
def load_excels_to_db(folder_path: str):
    files_loaded =load_excel_as_documents(folder_path, conn)
    return files_loaded

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Estimation Platform", layout="wide")
st.title("📊 Estimation Data Platform")

tab1, tab2, tab3 = st.tabs(["📂 Upload & Load Data", "🔎 Data Explorer", "📈 Dashboard"])

# ---- Tab 1: Load data
with tab1:
    st.subheader("📂 Load Estimation Data into SQLite")
    folder = st.text_input("Enter local folder path containing Excel files (includes subfolders):", "")
    if st.button("🚀 Load Data"):
        if folder and os.path.exists(folder):
            n = load_excels_to_db(folder)
            st.success(f"✅ Loaded data from {n} Excel file(s) into {DB_PATH}")
        else:
            st.error("⚠️ Please enter a valid folder path")

# ---- Tab 2: Explorer
with tab2:
    st.subheader("🔎 Estimation Data Explorer (Parent ➜ Child)")
    search = st.text_input("Global Search (Project, Feature, Technology, Module, Domain, Month, Estimation Type)")

    parents = pd.read_sql("SELECT * FROM estimation_summary", conn)

    if search:
        # LIKE filter across parent & child
        q = f"""
            SELECT DISTINCT es.*
            FROM estimation_summary es
            LEFT JOIN estimation_details ed ON ed.parent_id = es.id
            WHERE es.prj_name LIKE ? OR es.domain LIKE ? OR es.month LIKE ? OR es.type_of_estimation LIKE ?
               OR ed.feature_or_scenarios LIKE ? OR ed.technology_or_type LIKE ? OR ed.module_or_method LIKE ?
        """
        like = f"%{search}%"
        parents = pd.read_sql(q, conn, params=[like, like, like, like, like, like, like])

    st.caption("Parent (summary) records")
    st.dataframe(parents, use_container_width=True, hide_index=True)

    # Expand rows to show children
    for _, prow in parents.iterrows():
        with st.expander(f"🔽 Details for **{prow['prj_name']}** — {prow.get('month','')}, {prow.get('domain','')}"):
            child_df = pd.read_sql("SELECT * FROM estimation_details WHERE parent_id = ?", conn, params=[prow["id"]])
            st.dataframe(child_df, use_container_width=True, hide_index=True)

    # Export buttons
    colA, colB = st.columns(2)
    with colA:
        if st.button("⬇️ Export CURRENT VIEW (Parents + All Children) to Excel"):
            with pd.ExcelWriter("estimation_export.xlsx", engine="xlsxwriter") as writer:
                parents.to_excel(writer, sheet_name="Summary", index=False)
                all_children = pd.read_sql("SELECT * FROM estimation_details", conn)
                all_children.to_excel(writer, sheet_name="Details", index=False)
            st.success("Saved as estimation_export.xlsx (in working directory)")

    with colB:
        if st.button("⬇️ Export FULL DB (raw tables)"):
            with pd.ExcelWriter("estimation_full_db.xlsx", engine="xlsxwriter") as writer:
                pd.read_sql("SELECT * FROM estimation_summary", conn).to_excel(writer, sheet_name="Summary", index=False)
                pd.read_sql("SELECT * FROM estimation_details", conn).to_excel(writer, sheet_name="Details", index=False)
            st.success("Saved as estimation_full_db.xlsx")

# ---- Tab 3: Dashboard
with tab3:
    st.subheader("📈 Dashboard")

    summary_df = pd.read_sql("SELECT * FROM estimation_summary", conn)
    details_df = pd.read_sql("SELECT * FROM estimation_details", conn)

    if summary_df.empty:
        st.info("No data loaded yet. Please load Excel files in the first tab.")
    else:
        # ---- Filters
        with st.container():
            c1, c2, c3 = st.columns(3)
            month_sel = c1.multiselect("Month", sorted([m for m in summary_df["month"].dropna().unique()]))
            domain_sel = c2.multiselect("Domain", sorted([d for d in summary_df["domain"].dropna().unique()]))
            type_sel = c3.multiselect("Estimation Type", sorted([t for t in summary_df["type_of_estimation"].dropna().unique()]))

        fsum = summary_df.copy()
        if month_sel:
            fsum = fsum[fsum["month"].isin(month_sel)]
        if domain_sel:
            fsum = fsum[fsum["domain"].isin(domain_sel)]
        if type_sel:
            fsum = fsum[fsum["type_of_estimation"].isin(type_sel)]

        # Filter details to only those linked to filtered parents
        if not fsum.empty:
            parent_ids = tuple(fsum["id"].tolist())
            if len(parent_ids) == 1:
                parent_ids_sql = f"({parent_ids[0]})"
            else:
                parent_ids_sql = str(parent_ids)
            fdet = pd.read_sql(f"SELECT * FROM estimation_details WHERE parent_id IN {parent_ids_sql}", conn)
        else:
            fdet = details_df.iloc[0:0] # empty with same columns

        # ---- KPI Cards
        total_projects = int(fsum["id"].nunique()) if not fsum.empty else 0
        total_loe = float(fsum["total_loe"].fillna(0).sum()) if not fsum.empty else 0.0
        avg_loe = float(fsum["total_loe"].fillna(0).mean()) if not fsum.empty else 0.0

        dev_sp_sum = float(fsum["Development_SP"].fillna(0).sum()) if not fsum.empty else 0.0
        qa_sp_sum = float(fsum["QA_SP"].fillna(0).sum()) if not fsum.empty else 0.0

        # top technology in filtered details
        if not fdet.empty:
            top_tech = fdet["technology_or_type"].dropna().value_counts().idxmax()
        else:
            top_tech = "—"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Projects (filtered)", value=f"{total_projects}")
        with k2:
            st.metric("Total LOE (filtered)", value=f"{total_loe:,.2f}")
        with k3:
            st.metric("Avg LOE / Project", value=f"{avg_loe:,.2f}")
        with k4:
            st.metric("Top Technology", value=f"{top_tech}")

        k5, k6 = st.columns(2)
        with k5:
            st.metric("Sum Development SP", value=f"{dev_sp_sum:,.2f}")
        with k6:
            st.metric("Sum QA SP", value=f"{qa_sp_sum:,.2f}")

        st.divider()

        # ---- Charts (all filtered)
        if fsum.empty:
            st.warning("No data for selected filters.")
        else:
            # LOE trend by month & domain
            loe_fig = px.line(
                fsum.sort_values("month"),
                x="month", y="total_loe", color="domain",
                markers=True, title="Total LOE Trend by Month & Domain"
            )
            st.plotly_chart(loe_fig, use_container_width=True)

            # Technology usage (count of features) from filtered details
            if not fdet.empty:
                tech_usage = fdet.groupby("technology_or_type", dropna=False).size().reset_index(name="count")
                tech_usage["technology_or_type"] = tech_usage["technology_or_type"].fillna("Unspecified")
                tech_fig = px.bar(tech_usage.sort_values("count", ascending=False),
                                  x="technology_or_type", y="count",
                                  title="Most Used Technologies (Filtered)")
                st.plotly_chart(tech_fig, use_container_width=True)

                # Module usage (Dev vs QA)
                mod_usage = fdet.groupby(["module_or_method", "type"], dropna=False).size().reset_index(name="count")
                mod_usage["module_or_method"] = mod_usage["module_or_method"].fillna("Unspecified")
                mod_fig = px.bar(mod_usage.sort_values("count", ascending=False),
                                 x="module_or_method", y="count", color="type",
                                 title="Most Used Modules (Dev vs QA, Filtered)")
                st.plotly_chart(mod_fig, use_container_width=True)
            else:
                st.info("No detail rows available for current filters.")

        # Optional: download filtered slices
        with st.expander("⬇️ Download filtered data"):
            if not fsum.empty:
                st.download_button(
                    "Download filtered SUMMARY (CSV)",
                    fsum.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_summary.csv",
                    mime="text/csv"
                )
            if not fdet.empty:
                st.download_button(
                    "Download filtered DETAILS (CSV)",
                    fdet.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_details.csv",
                    mime="text/csv"
                )
