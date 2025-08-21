import os
import sqlite3
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table, State
import plotly.express as px
from CreateDatabaseHelper import *
from LoadExcelAsDocument import load_excel_as_documents
import dash

DB_FILE = "estimation.db"

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
    Output("child-table", "data"),
    Output("child-table", "columns"),
    Input("global-search", "value"),
    Input("parent-table", "selected_rows"),
)
def global_search_filter(search_value, selected_rows):
    parent_df = get_parent_data()

    if search_value:
        search_value = search_value.lower()
        parent_df = parent_df[parent_df.apply(
            lambda row: row.astype(str).str.lower().str.contains(search_value).any(), axis=1
        )]

    child_data, child_columns = [], []
    if not parent_df.empty and selected_rows:
        if selected_rows[0] < len(parent_df):
            parent_id = parent_df.iloc[selected_rows[0]]["id"]
            child_df = get_child_data(parent_id)

            if search_value:
                child_df = child_df[child_df.apply(
                    lambda row: row.astype(str).str.lower().str.contains(search_value).any(), axis=1
                )]
            # Drop unnecessary columns
            child_df.drop(columns=["id", "parent_id"], inplace=True, errors="ignore")

            # Add row numbers (dynamic, adjusts when filtered)
            child_df.insert(0, "Row No", range(1, len(child_df) + 1))

            child_df.rename(columns={
            "feature_or_scenarios": "Feature/Scenario",
            "technology_or_type": "Technology",
            "module_or_method": "Module/Method",
            "complexity": "Complexity",
            "total_SP": "Total Story Points",
            "type": "Type"
            }, inplace=True)

            child_data = child_df.to_dict("records")
            child_columns = [{"name": i, "id": i} for i in child_df.columns]

    return parent_df.to_dict("records"), child_data, child_columns


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

    app.run(debug=True)
