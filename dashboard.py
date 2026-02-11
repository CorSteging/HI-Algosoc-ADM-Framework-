from adms import *
from pathlib import Path
import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import base64

# ===========================
# STATIC METRIC DEFINITIONS
# ===========================
AVAILABLE_METRICS = [
    {"label": "Demographic Parity", "value": "demographic_parity"},
    {"label": "Statistical Parity Difference", "value": "statistical_parity_difference"},
    {"label": "Disparate Impact Ratio", "value": "disparate_impact_ratio"},
    {"label": "Selection Rate Difference", "value": "selection_rate_difference"},
    {"label": "Selection Rate Ratio", "value": "selection_rate_ratio"},
    {"label": "80% Rule Compliance", "value": "four_fifths_rule"},
]

# ---------------------------
# Temporary storage
# ---------------------------
uploaded_file_store = {}
model_output_store = {}

# ---------------------------
# Helpers
# ---------------------------
def fairness_flag(disparity):
    if disparity < 0.10:
        return "Fair", "success"
    elif disparity < 0.20:
        return "Warning: noticeable disparity", "warning"
    else:
        return "High disparity: potential unfairness", "danger"

def generate_stacked_bar(percentages, colors=None, title=None, height=220):
    fig = go.Figure()
    default_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for i, key in enumerate(percentages):
        val = percentages[key]
        color = colors[key] if colors and key in colors else default_colors[i % len(default_colors)]
        fig.add_trace(go.Bar(
            x=[val],
            y=[""],
            orientation="h",
            name=str(key),
            text=[f"{val:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            marker_color=color,
            hovertemplate=f"{key}: {val:.1f}%<extra></extra>"
        ))
    fig.update_layout(
        barmode="stack",
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return fig

def generate_numeric_distribution(df, column, selected_mask, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[column],
        name="All applicants",
        opacity=0.6,
        marker_color="#636EFA",
        nbinsx=20
    ))
    fig.add_trace(go.Histogram(
        x=df[selected_mask][column],
        name="Selected",
        opacity=0.7,
        marker_color="#EF553B",
        nbinsx=20
    ))
    selected_mean = df[selected_mask][column].mean()
    fig.add_vline(
        x=selected_mean,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text="Selected mean",
        annotation_position="top right"
    )
    fig.update_layout(
        barmode="overlay",
        title=title,
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=column,
        yaxis_title="Count",
        legend=dict(x=0.8, y=0.95)
    )
    return fig

def compute_disparity_info(df, attr, selected_mask):
    all_groups = sorted(df[attr].dropna().unique())
    selected_dist = df[selected_mask][attr].value_counts(normalize=True).reindex(all_groups).fillna(0.0)
    dataset_dist = df[attr].value_counts(normalize=True).reindex(all_groups).fillna(0.0)
    parity_dist = pd.Series(1.0 / len(all_groups), index=all_groups)
    dataset_disp = (selected_dist - dataset_dist).abs().max()
    parity_disp = (selected_dist - parity_dist).abs().max()
    actual_percentages = (selected_dist * 100).to_dict()
    return actual_percentages, float(dataset_disp), float(parity_disp)

def compute_numeric_disparity(df, column, selected_mask):
    overall_mean = df[column].mean()
    selected_mean = df[selected_mask][column].mean()
    std = df[column].std() if df[column].std() > 0 else 1.0
    disparity = abs(selected_mean - overall_mean) / std
    return overall_mean, selected_mean, float(disparity)

def compute_selection_robustness_overall(df, selected_col="selected", runs=10):
    temp_path = Path("temp_upload.json")
    with open(temp_path, "w") as f:
        json.dump(uploaded_file_store["latest"], f)
    n = len(df)
    selection_counts = pd.Series(0, index=df.index)
    for _ in range(runs):
        output = run_model(df["model"].iloc[0], k=10, path=temp_path)
        selection_counts += df["name"].isin(output).astype(int)
    stable = ((selection_counts == 0) | (selection_counts == runs)).sum()
    changed = n - stable
    return {"Stable": stable / n * 100, "Changed": changed / n * 100}

# ---------------------------
# Initialize Dash
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "eValue-8"

# ---------------------------
# Layout
# ---------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Img(src="/assets/logo.png", height="80px"),
            width="auto",
            style={"paddingRight": "20px"}
        ),
        dbc.Col(
            html.H1("A Hybrid Tool for Evaluating AI models"),
            style={"display": "flex", "alignItems": "center"}
        )
    ], align="center"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H5("Upload Applicant JSON"),
            dcc.Upload(
                id="upload-json",
                children=html.Div(["Drag & Drop or ", html.A("Select JSON Files")]),
                multiple=False,
                style={"border": "1px dashed #aaa", "padding": "20px", "textAlign": "center"}
            ),
            html.Div(id="upload-status", style={"marginTop": "10px"})
        ], width=6),
        dbc.Col([
            html.H5("Select Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": f"Model {i}", "value": f"model_{i}"} for i in range(1, 6)],
                value="model_1",
                clearable=False
            )
        ], width=6)
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([html.Button("Run Model", id="run-model-btn", className="btn btn-primary")], width=4),
        dbc.Col([html.Button("Download CSV", id="download-btn", className="btn btn-success"),
                 dcc.Download(id="download-output")], width=4)
    ]),
    html.Hr(),
    html.Div(id="metrics-dashboard")
], fluid=True, style={"backgroundColor": "#f5f5f5", "minHeight": "100vh"})

# ---------------------------
# Callbacks
# ---------------------------
@app.callback(
    Output("upload-status", "children"),
    Input("upload-json", "contents"),
    State("upload-json", "filename")
)
def upload_json(contents, filename):
    if contents is None:
        return "No file uploaded yet."
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    uploaded_file_store["latest"] = json.loads(decoded)
    return f"File '{filename}' uploaded successfully."

@app.callback(
    Output("metrics-dashboard", "children"),
    Input("run-model-btn", "n_clicks"),
    State("model-dropdown", "value")
)
def run_model_and_metrics(n_clicks, selected_model):
    if not n_clicks or "latest" not in uploaded_file_store:
        return "Metrics will appear after running the model."

    temp_path = Path("temp_upload.json")
    with open(temp_path, "w") as f:
        json.dump(uploaded_file_store["latest"], f)

    system_output = run_model(selected_model, k=10, path=temp_path)
    full_info = load_applicants(temp_path)
    df_full = pd.DataFrame(full_info)
    df_full["selected"] = df_full["name"].isin(system_output)
    df_full["model"] = selected_model

    age_bins = [0, 20, 25, 30, 35, 40, 50, 100]
    age_labels = ["<20", "20-25", "25-30", "30-35", "35-40", "40-50", "50+"]
    df_full["age"] = pd.cut(df_full["age"], bins=age_bins, labels=age_labels, right=False)

    model_output_store["latest"] = df_full
    children = []

    # System output table
    children.append(
        dbc.Card([
            dbc.CardHeader("System Output (Selected Candidates)"),
            dbc.CardBody(
                html.Div(
                    dbc.Table.from_dataframe(pd.DataFrame({"Selected Name": system_output}), striped=True, hover=True),
                    style={"maxHeight": "300px", "overflowY": "auto"}
                )
            )
        ])
    )

    children.append(html.Hr())
    children.append(html.H4("Fairness Metrics"))

    for attr, colors in [("gender", {"male": "blue", "female": "orange"}), ("country", None), ("age", None)]:
        actual_pct, dataset_disp, parity_disp = compute_disparity_info(df_full, attr, df_full["selected"])
        text_flag, color_flag = fairness_flag(parity_disp)
        children.append(
            dbc.Alert(
                f"{attr.capitalize()} fairness status: {text_flag} "
                f"(ideal disparity={parity_disp:.2f}, dataset disparity={dataset_disp:.2f})",
                color=color_flag
            )
        )
        fig = generate_stacked_bar(actual_pct, colors=colors, title=f"{attr.capitalize()} Distribution (Selected)")
        children.append(dcc.Graph(figure=fig))

    overall_mean, selected_mean, score_disp = compute_numeric_disparity(df_full, "final_score", df_full["selected"])
    text_flag, color_flag = fairness_flag(score_disp)
    children.append(
        dbc.Alert(
            f"Final Score fairness status: {text_flag} "
            f"(selected mean={selected_mean:.2f}, overall mean={overall_mean:.2f}, normalized disparity={score_disp:.2f})",
            color=color_flag
        )
    )
    children.append(
        dcc.Graph(
            figure=generate_numeric_distribution(df_full, "final_score", df_full["selected"], "Final Score Distribution")
        )
    )

    # Selection robustness
    robustness_pct = compute_selection_robustness_overall(df_full, selected_col="selected", runs=10)
    children.append(html.Hr())
    children.append(html.H4("Selection Robustness Across 10 Runs"))
    children.append(
        dcc.Graph(
            figure=generate_stacked_bar(
                robustness_pct,
                colors={"Stable": "#00CC96", "Changed": "#EF553B"},
                title="Candidate Selection Robustness (%)",
                height=250
            )
        )
    )

    # Metric modal button
    children.append(html.Hr())
    children.append(
        dbc.Button(
            "Add Metric",
            id="open-metric-modal",
            color="primary",
            size="lg",
            className="w-100",
            style={"marginBottom": "80px"}
        )
    )
    children.append(
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Add Metric")),
            dbc.ModalBody(dcc.Dropdown(id="metric-dropdown", options=AVAILABLE_METRICS, placeholder="Select a metric")),
            dbc.ModalFooter([dbc.Button("Cancel", id="close-metric-modal"), dbc.Button("Add", id="confirm-metric-modal", color="primary")])
        ], id="metric-modal", is_open=False, centered=True)
    )

    return children

@app.callback(
    Output("metric-modal", "is_open"),
    Input("open-metric-modal", "n_clicks"),
    Input("close-metric-modal", "n_clicks"),
    Input("confirm-metric-modal", "n_clicks"),
    State("metric-modal", "is_open")
)
def toggle_metric_modal(open_clicks, close_clicks, confirm_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return button_id == "open-metric-modal"

@app.callback(
    Output("download-output", "data"),
    Input("download-btn", "n_clicks")
)
def download_csv(n_clicks):
    if not n_clicks or "latest" not in model_output_store:
        return dash.no_update
    df = model_output_store["latest"]
    return dcc.send_data_frame(df.to_csv, "system_output.csv", index=False)

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8051)
