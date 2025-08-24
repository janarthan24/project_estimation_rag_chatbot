import os
import time
import uuid
import sqlite3
import pandas as pd
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from dash import Dash, dcc, html, Input, Output, dash_table, State
import plotly.express as px
from CreateDatabaseHelper import *
from ChatLLamaModel import *
from LoadExcelAsDocument import load_excel_as_documents
import dash
from dash.exceptions import PreventUpdate

# ----------------------------
# Config
# ----------------------------
DB_FILE = "estimation.db"
INDEX_PATH = "estimation_faiss.index"
# ----------------------------
# Load Excel -> SQLite
# ----------------------------
def load_excel_to_db(folder_path):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    create_tables(conn)

    # Clear old data to avoid duplicates
    cur.execute("DELETE FROM estimation_summary")
    cur.execute("DELETE FROM estimation_details")

    files_loaded =load_excel_as_documents(folder_path, conn)
    conn.commit()
    conn.close()


# ----------------------------
# Helpers
# ----------------------------
def get_parent_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM estimation_summary", conn)
    conn.close()

    # Rename columns
    df.rename(columns={
        "prj_name": "Project Name",
        "prj_type": "Type",
        "type_of_estimation": "Estimation Type",
        "Development_SP": "Development SP",
        "Architecture_SP": "Architecture SP",
        "QA_SP": "QA SP",
        "Implementation_SP": "Implementation SP",
        "Innovation_SP": "Innovation SP",
        "DevOps": "DevOps SP",
        "total_loe": "Total LOE",
        "month": "Month",
        "domain": "Domain",
        "date": "Date"
    }, inplace=True)

    # Format to 2 decimal places for all numeric columns except 'id' column
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'id':
         df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    return df
    


def get_child_data(parent_id):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM estimation_details WHERE parent_id={parent_id}", conn)
    conn.close()
    return df


# ----------------------------
# Dash App
# ----------------------------
app = Dash(__name__)

# DataTable Style
table_style = {
    "style_header": {
        "backgroundColor": "#2c3e50",
        "color": "white",
        "fontWeight": "bold",
        "textAlign": "center",
        "border": "1px solid #ddd"
    },
    "style_cell": {
        "textAlign": "left",
        "padding": "8px",
        "whiteSpace": "normal",
        "height": "auto",
        "border": "1px solid #ddd",
        "fontFamily": "Arial",
        "fontSize": "14px",
    },
    "style_data_conditional": [
        {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
        {"if": {"state": "selected"}, "backgroundColor": "#3498db", "color": "white"},
        {"if": {"state": "active"}, "border": "1px solid #2980b9"},
    ],
    "style_table": {
        "overflowX": "auto",
        "maxHeight": "400px",
        "overflowY": "scroll",
        "border": "1px solid #ccc",
        "borderRadius": "10px",
    },
}

# Layout
app.layout = html.Div([
    html.H1("Estimation Dashboard", style={"textAlign": "center"}),

    dcc.Tabs([

          # Tab 1: Chatbot
        dcc.Tab(label="💬 Chatbot", children=[
            html.Div([
                html.Div(id="chat-window",
                    style={
                        "height": "350px",
                        "overflowY": "auto",
                        "border": "1px solid #ddd",
                        "padding": "10px",
                        "backgroundColor": "#fafafa",
                        "borderRadius": "10px"
                    }
                ),
                html.Div(id="scroll-anchor"),
                html.Div([
                    dcc.Input(
                        id="chat-input",
                        type="text",
                        placeholder="Ask anything about projects, LOE, features…",
                        style={
                            "width": "80%", "padding": "10px",
                            "borderRadius": "15px", "border": "1px solid #ccc"
                        },
                        debounce=True
                    ),
                    html.Button("Send", id="send-btn",
                        style={
                            "padding": "10px 20px",
                            "marginLeft": "10px",
                            "borderRadius": "15px",
                            "backgroundColor": "#4CAF50",
                            "color": "white",
                            "border": "none"
                        })
                ], style={"marginTop": "10px", "display": "flex", "alignItems": "center"}),
                dcc.Store(id="chat-store", data=[]),
                dcc.Store(id="gen-lock", data=False),
                dcc.Interval(id="typing-interval", interval=500, disabled=True)  # typing dots animator
            ], style={"maxWidth": "900px", "margin": "auto", "marginTop": "20px"})
        ]),
        # ---------------- Data Explorer ----------------
        dcc.Tab(label="📊 Data Explorer", children=[
            html.Div([
                html.H3("🔎 Search"),
                dcc.Input(
                    id="global-search",
                    type="text",
                    placeholder="Search across summary tables...",
                    style={"width": "60%", "padding": "10px", "marginBottom": "20px"}
                ),

                html.H3("Estimation Summary"),
                dash_table.DataTable(
                    id="parent-table",
                    columns=[{"name": i, "id": i} for i in get_parent_data().columns],
                    data=get_parent_data().to_dict("records"),
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                    row_selectable="single",
                    **table_style
                ),
                html.Button("⬇ Export Estimation Summary Data", id="export-parent-btn", n_clicks=0),
                dcc.Download(id="download-parent"),

                html.Hr(),
                html.H3("Estimation Details (select summary row to view)"),
                dash_table.DataTable(
                    id="child-table",
                    columns=[],
                    data=[],
                    page_size=10,
                    filter_action="native",
                    sort_action="native",
                    **table_style
                ),
                html.Button("⬇ Export Estimation Details", id="export-child-btn", n_clicks=0),
                dcc.Download(id="download-child"),
            ])
        ]),

        # ---------------- Dashboard ----------------
        dcc.Tab(label="📈 Dashboard", children=[
            html.Div([
                html.H3("Filters"),
                html.Div([
                    dcc.Dropdown(id="filter-project", placeholder="Filter by Project Type"),
                    dcc.Dropdown(id="filter-month", placeholder="Filter by Month"),
                    dcc.Dropdown(id="filter-domain", placeholder="Filter by Domain"),
                    dcc.Dropdown(id="filter-estimation", placeholder="Filter by Estimation Type"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"}),

                html.Hr(),
                html.Div(id="kpi-cards", style={"display": "flex", "gap": "20px"}),

                html.Hr(),
                dcc.Graph(id="trend-graph"),
                dcc.Graph(id="bar-domain"),
                dcc.Graph(id="pie-estimation"),
            ])
        ])
    ])
])


# ----------------------------
# Callbacks
# ----------------------------

@app.callback(
    Output("parent-table", "data"),
    Output("parent-table", "columns"),
    Output("child-table", "data"),
    Output("child-table", "columns"),
    Input("global-search", "value"),
    Input("parent-table", "selected_rows")
)
def global_search_update(search_value, selected_rows):
    # --- Load data ---
    parent_df = get_parent_data()
    conn = sqlite3.connect(DB_FILE)
    child_df_full = pd.read_sql("SELECT * FROM estimation_details", conn)
    conn.close()


    child_rename = {
        "feature_or_scenarios": "Feature / Scenario",
        "technology_or_type": "Technology / Type",
        "module_or_method": "Module / Method",
        "complexity": "Complexity",
        "total_SP": "Total SP",
        "type": "Type"
    }

    # --- Filter Logic for Global Search ---
    if search_value:
        # Parent filter
        mask_parent = parent_df.apply(
            lambda row: row.astype(str).str.contains(search_value, case=False).any(),
            axis=1
        )
        matched_parents = parent_df[mask_parent]

        # Child filter
        mask_child = child_df_full.apply(
            lambda row: row.astype(str).str.contains(search_value, case=False).any(),
            axis=1
        )
        matched_children = child_df_full[mask_child]

        # Keep parent rows where children matched
        matched_parent_ids = matched_children["parent_id"].unique()
        matched_parents = pd.concat(
            [matched_parents, parent_df[parent_df["id"].isin(matched_parent_ids)]]
        ).drop_duplicates()

        parent_df = matched_parents

    # --- Prepare Parent Table ---
    parent_display = parent_df
    parent_data = parent_display.to_dict("records")
    parent_cols = [{"name": i, "id": i} for i in parent_display.columns]

    # --- Prepare Child Table (based on selection) ---
    if selected_rows and not parent_df.empty:
        print(f"Selected rows: {selected_rows}")
        parent_id = parent_df.iloc[selected_rows[0]]["id"]
        print(f"Selected parent ID: {parent_id}")
        

        # Pick children of selected parent
        child_df = child_df_full[child_df_full["parent_id"] == parent_id]

        # If global search, filter child rows too
        if search_value:
            child_df = child_df[child_df.apply(
                lambda row: row.astype(str).str.contains(search_value, case=False).any(),
                axis=1
            )]

        # Drop technical IDs, add row number
        child_df = child_df.drop(columns=["id", "parent_id"], errors="ignore").reset_index(drop=True)
        child_df.insert(0, "Row No", range(1, len(child_df) + 1))

        # Rename columns
        child_df = child_df.rename(columns=child_rename)

        child_data = child_df.to_dict("records")
        child_cols = [{"name": i, "id": i} for i in child_df.columns]
    else:
        print("No parent selected or no children available")
        child_data, child_cols = [], []

    return parent_data, parent_cols, child_data, child_cols



# ─────────────────────────────────────────────────────────────────────────────
# Chatbot flow (two-step): show typing → run RAG+LLM → replace typing
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("chat-store", "data"),
    Output("typing-interval", "disabled"),
    Output("chat-input", "value"),  # clear input after send
    Output("gen-lock", "data"),
    Input("send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def add_user_and_typing(n_clicks, user_msg, history):
    if not n_clicks or not user_msg:
        raise PreventUpdate

    history = list(history or [])
    history.append({"role": "user", "content": user_msg, "id": str(uuid.uuid4())})
    history.append({"role": "assistant", "content": "typing", "id": "typing"})
    # Enable typing animation and set generation lock ON
    return history, False, "", True

@app.callback(
    Output("chat-store", "data", allow_duplicate=True),
    Output("typing-interval", "disabled", allow_duplicate=True),
    Output("gen-lock", "data", allow_duplicate=True),
    Input("gen-lock", "data"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def generate_real_reply(gen_lock, history):
    # Only run when lock is True
    if not gen_lock:
        raise PreventUpdate

    if not history or history[-1].get("id") != "typing":
        # Nothing to generate
        return history, True, False

    # Get latest user message
    user_msg = next((m["content"] for m in reversed(history[:-1]) if m["role"] == "user"), "").strip()
    if not user_msg:
        # Safety guard
        history[-1] = {"role": "assistant", "content": "Please enter a question.", "id": str(uuid.uuid4())}
        return history, True, False

    # 1) Retrieve + rerank
    print(f"Retrieving context for: {user_msg}")
    contexts = retrieve_with_rerank(user_msg)

    # 2) Build prompt
    print("Building prompt...")
    prompt = build_prompt(user_msg, contexts)

    # 3) Call local LLaMA (Ollama)
    print("Calling Ollama...")
    answer = call_ollama(prompt, temperature=0.2, max_tokens=700)
    print(f"LLM Answer: {answer}")

    # 4) Replace typing with final answer
    history[-1] = {"role": "assistant", "content": answer, "id": str(uuid.uuid4())}

    # Disable typing animation and clear generation lock
    return history, True, False

@app.callback(
    Output("chat-window", "children"),
    Input("chat-store", "data"),
    Input("typing-interval", "n_intervals")  # re-render to animate dots
)
def render_chat(history, _n):
    # Render bubbles and animated typing dots
    bubbles = []
    now_ticks = int(time.time() * 2)  # for dot animation ~2Hz
    for msg in history or []:
        if msg["role"] == "user":
            bubbles.append(html.Div(msg["content"], style={
                "textAlign": "right",
                "backgroundColor": "#DCF8C6",
                "padding": "8px 12px",
                "margin": "6px",
                "borderRadius": "15px",
                "maxWidth": "70%",
                "marginLeft": "auto",
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.35",
                "boxShadow": "0 1px 1px rgba(0,0,0,0.08)"
            }))
        else:  # assistant
            if msg["content"] == "typing":
                dots = "." * ((now_ticks % 3) + 1)
                bubbles.append(html.Div(f"Assistant is thinking{dots}", style={
                    "textAlign": "left",
                    "fontStyle": "italic",
                    "color": "#555",
                    "margin": "6px"
                }))
            else:
                bubbles.append(html.Div(msg["content"], style={
                    "textAlign": "left",
                    "backgroundColor": "#f1f0f0",
                    "padding": "8px 12px",
                    "margin": "6px",
                    "borderRadius": "15px",
                    "maxWidth": "70%",
                    "marginRight": "auto",
                    "whiteSpace": "pre-wrap",
                    "lineHeight": "1.35",
                    "boxShadow": "0 1px 1px rgba(0,0,0,0.08)"
                }))
    bubbles.append(html.Div(id="scroll-anchor"))
    return bubbles


@app.callback(
    Output("filter-project", "options"),
    Output("filter-month", "options"),
    Output("filter-domain", "options"),
    Output("filter-estimation", "options"),
    Input("parent-table", "data")
)
def update_filter_options(data):
    df = pd.DataFrame(data)
    return (
        [{"label": i, "value": i} for i in df["Type"].dropna().unique()],
        [{"label": i, "value": i} for i in df["Month"].dropna().unique()],
        [{"label": i, "value": i} for i in df["Domain"].dropna().unique()],
        [{"label": i, "value": i} for i in df["Estimation Type"].dropna().unique()],
    )


@app.callback(
    Output("kpi-cards", "children"),
    Output("trend-graph", "figure"),
    Output("bar-domain", "figure"),
    Output("pie-estimation", "figure"),
    Input("filter-project", "value"),
    Input("filter-month", "value"),
    Input("filter-domain", "value"),
    Input("filter-estimation", "value"),
)
def update_dashboard(project, month, domain, estimation_type):
    df = get_parent_data()

    # Apply filters
    if project: df = df[df["Type"] == project]
    if month: df = df[df["Month"] == month]
    if domain: df = df[df["Domain"] == domain]
    if estimation_type: df = df[df["Estimation Type"] == estimation_type]

    # KPIs
    print(df["Development SP"])
    #total_loe = df["Total LOE"].sum()
    total_loe = pd.to_numeric(df['Total LOE'], errors='coerce').sum()
   # avg_dev = df["Development SP"].mean()
    avg_dev = pd.to_numeric(df['Development SP'], errors='coerce').sum()
    #avg_qa = df["QA SP"].mean(numeric_only=True)
    avg_qa = pd.to_numeric(df['QA SP'], errors='coerce').sum()
    projects = df["id"].nunique()

    kpi_cards = [
        html.Div([
            html.H4("Total LOE"), html.P(f"{total_loe:,.0f}")
        ], style={"background": "#3498db", "color": "white", "padding": "20px", "borderRadius": "10px", "flex": 1}),
        html.Div([
            html.H4("Avg Development SP"), html.P(f"{avg_dev:,.1f}")
        ], style={"background": "#2ecc71", "color": "white", "padding": "20px", "borderRadius": "10px", "flex": 1}),
        html.Div([
            html.H4("Avg QA SP"), html.P(f"{avg_qa:,.1f}")
        ], style={"background": "#e67e22", "color": "white", "padding": "20px", "borderRadius": "10px", "flex": 1}),
        html.Div([
            html.H4("Projects"), html.P(f"{projects}")
        ], style={"background": "#9b59b6", "color": "white", "padding": "20px", "borderRadius": "10px", "flex": 1}),
    ]

    # Charts
    df_grouped = df.groupby(["Month", "Domain"], as_index=False)["Total LOE"].sum()
    trend_fig = px.line(df_grouped, x="Month", y="Total LOE", color="Domain", markers=True,
                        title="Trend of Total LOE by Month")

    bar_fig = px.bar(df.groupby("Domain", as_index=False)["Total LOE"].sum(),
                     x="Domain", y="Total LOE", title="Total LOE by Domain")

    pie_fig = px.pie(df.groupby("Estimation Type", as_index=False)["Total LOE"].sum(),
                     names="Estimation Type", values="Total LOE", title="Contribution by Estimation Type")

    return kpi_cards, trend_fig, bar_fig, pie_fig


# Export callbacks stay same
@app.callback(
    Output("download-parent", "data"),
    Input("export-parent-btn", "n_clicks"),
    prevent_initial_call=True
)
def export_parent(n_clicks):
    df = get_parent_data()
    return dcc.send_data_frame(df.to_excel, "parent_data.xlsx", index=False)


@app.callback(
    Output("download-child", "data"),
    Input("export-child-btn", "n_clicks"),
    State("parent-table", "selected_rows"),
    prevent_initial_call=True
)
def export_child(n_clicks, selected_rows):
    if not selected_rows:
        return dash.no_update
    parent_df = get_parent_data()
    parent_id = parent_df.iloc[selected_rows[0]]["id"]
    df = get_child_data(parent_id)
    return dcc.send_data_frame(df.to_excel, f"child_data_parent_{parent_id}.xlsx", index=False)


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    folder_path = "C:\\Users\\janarm\\Downloads\\llm projects\\python\\project_estimation_rag_chatbot\\data1"
    if not os.path.exists(DB_FILE):
        load_excel_to_db(folder_path)
    if not os.path.exists(INDEX_PATH):
        print("⚡ Building FAISS index...")
        build_faiss_index(get_conn())

    app.run(debug=True)