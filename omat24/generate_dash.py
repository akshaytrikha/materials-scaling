import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import os
import json
import numpy as np
from math import ceil
from scipy.optimize import curve_fit

################################
# 1. DEFINE YOUR EXPERIMENTS
################################
EXPERIMENTS = {
    "experiment_20250126_190425": {
        "title": "Scaled vs. Unscaled loss",
        "description": "Are Luis' loss coefficients really necessary?",
        "json_files": ["experiments_20250126_190425.json", "fcn_batch_size=64.json"],
    },
}


################################
# 2. HELPER FUNCTIONS
################################
def load_and_process_single_json(json_filepath):
    """
    Load and process a single JSON into (summary_val, detailed_runs).
    """
    with open(os.path.join("results", json_filepath), "r") as f:
        data = json.load(f)

    summary_val = {}
    detailed_runs = {}
    for ds_size_str, runs in data.items():
        ds_size = float(ds_size_str)
        run_list = []
        for r in runs:
            val_losses = [
                x["val_loss"] for x in r["losses"].values() if x["val_loss"] is not None
            ]
            train_losses = [
                x["train_loss"]
                for x in r["losses"].values()
                if x["train_loss"] is not None
            ]
            if not val_losses or not train_losses:
                continue
            best_val = min(val_losses)
            best_train = min(train_losses)
            nparams = r["config"]["num_params"]
            run_list.append((nparams, best_val, best_train, r))
        if run_list:
            detailed_runs[ds_size] = run_list
            summary_val[ds_size] = min(x[1] for x in run_list)

    return summary_val, detailed_runs


def load_and_process_multiple(json_filepaths):
    """
    Returns a dict of summary_vals and a dict of detailed_runs keyed by filename.
      summary_vals[filename] = {ds_size -> best_val_loss}
      detailed_runs[filename] = {ds_size -> runs}
    """
    all_summary_vals = {}
    all_detailed_runs = {}
    for fp in json_filepaths:
        s_val, d_runs = load_and_process_single_json(fp)
        all_summary_vals[fp] = s_val
        all_detailed_runs[fp] = d_runs
    return all_summary_vals, all_detailed_runs


def prepare_scaling_data(summary_val):
    ds_sizes = np.array(list(summary_val.keys()))
    val_losses = np.array(list(summary_val.values()))
    idx = np.argsort(ds_sizes)
    return ds_sizes[idx], val_losses[idx]


def compute_powerlaw_fit(xvals, yvals):
    def power_law(x, a, b):
        return a * x ** (-b)

    if len(xvals) < 2:
        return yvals, None, None

    try:
        popt, _ = curve_fit(power_law, xvals, yvals, p0=(1, 1))
        a, b = popt
        fitted_curve = power_law(xvals, a, b)
        return fitted_curve, a, b
    except:
        return yvals, None, None


def create_plot1_figure(all_summary_vals, exp_id):
    """
    Creates Plot-1 showing Val Loss vs Dataset Size for each JSON file,
    along with a power-law fit. The X/Y scale toggle buttons are moved
    to the bottom. The fit-line color is matched to its corresponding data.
    """
    fig = go.Figure()

    # Use a color cycle so each data + fit pair share the same color
    color_cycle = px.colors.qualitative.Plotly
    filenames = list(all_summary_vals.keys())

    for i, filename in enumerate(filenames):
        summary_val = all_summary_vals[filename]
        xvals, yvals = prepare_scaling_data(summary_val)

        color = color_cycle[i % len(color_cycle)]

        # MAIN data trace
        main_trace = go.Scatter(
            x=xvals,
            y=yvals,
            mode="lines+markers",
            name=f"{os.path.basename(filename)}",
            customdata=[(filename, ds) for ds in xvals],
            hovertemplate="File: " + filename + "<br>DS: %{x}<br>ValLoss: %{y}",
            line=dict(color=color),
            marker=dict(color=color),
        )
        fig.add_trace(main_trace)

        # Optional power-law fit
        fitted_vals, a, b = compute_powerlaw_fit(xvals, yvals)
        if a is not None:
            label = f"y={a:.2f} x^(-{b:.2f})"
            fit_trace = go.Scatter(
                x=xvals,
                y=fitted_vals,
                mode="lines",
                line=dict(color=color, dash="dash"),
                name=label,
                hoverinfo="skip",
            )
            fig.add_trace(fit_trace)

    # Define buttons for axis‐scale toggling (placed at the bottom)
    xaxis_buttons = [
        dict(label="X-Linear", method="relayout", args=[{"xaxis.type": "linear"}]),
        dict(label="X-Log", method="relayout", args=[{"xaxis.type": "log"}]),
    ]
    yaxis_buttons = [
        dict(label="Y-Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
        dict(label="Y-Log", method="relayout", args=[{"yaxis.type": "log"}]),
    ]

    fig.update_layout(
        title=f"Plot-1: Scaling Law ~ {exp_id}",
        xaxis_title="Dataset Size",
        yaxis_title="Val Loss",
        template="plotly_white",
        hovermode="closest",
        clickmode="event+select",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Place legend near the bottom
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=150),  # More room at the bottom
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=xaxis_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.25,
                y=-0.35,
                xanchor="center",
                yanchor="top",
            ),
            dict(
                type="buttons",
                direction="left",
                buttons=yaxis_buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.75,
                y=-0.35,
                xanchor="center",
                yanchor="top",
            ),
        ],
    )

    return fig


def create_saturation_figs(runs_for_size, ds_size, exp_id, filename):
    """
    Build two figures side-by-side: Train vs epoch, Val vs epoch for all runs at ds_size.
    """
    fig_train = go.Figure(
        layout=dict(
            title=f"Train Loss (ds={int(ds_size)}) ~ {exp_id}\n{filename}",
            xaxis_title="Epoch",
            yaxis_title="Train Loss",
            template="plotly_white",
        )
    )
    fig_val = go.Figure(
        layout=dict(
            title=f"Val Loss (ds={int(ds_size)}) ~ {exp_id}\n{filename}",
            xaxis_title="Epoch",
            yaxis_title="Val Loss",
            template="plotly_white",
        )
    )
    for nparams, best_val, best_train, run_data in runs_for_size:
        batch_size = run_data["config"]["batch_size"]
        ds_int = int(run_data["config"]["dataset_size"])
        n_batches = ceil(ds_int / batch_size)

        epoch_map = {}
        for step_str, stepvals in run_data["losses"].items():
            step_i = int(step_str)
            e = step_i // n_batches
            if e not in epoch_map:
                epoch_map[e] = {"train": [], "val": []}
            if stepvals["train_loss"] is not None:
                epoch_map[e]["train"].append(stepvals["train_loss"])
            if stepvals["val_loss"] is not None:
                epoch_map[e]["val"].append(stepvals["val_loss"])

        epochs_sorted = sorted(epoch_map.keys())
        train_y, val_y = [], []
        for e in epochs_sorted:
            t_list = epoch_map[e]["train"]
            v_list = epoch_map[e]["val"]
            train_y.append(t_list[-1] if t_list else None)
            val_y.append(v_list[-1] if v_list else None)

        fig_train.add_trace(
            go.Scatter(
                x=epochs_sorted,
                y=train_y,
                mode="lines+markers",
                name=f"{nparams} params",
            )
        )
        fig_val.add_trace(
            go.Scatter(
                x=epochs_sorted,
                y=val_y,
                mode="lines+markers",
                name=f"{nparams} params",
            )
        )

    return fig_train, fig_val


################################
# 3. MULTI-PAGE DASH APP
################################
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


def layout_homepage():
    cards = []
    for exp_id, exp_info in EXPERIMENTS.items():
        all_summary_vals, _ = load_and_process_multiple(exp_info["json_files"])
        fig = create_plot1_figure(all_summary_vals, exp_id)

        card = dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4(exp_info["title"], className="card-title"),
                        html.P(exp_info["description"], className="card-text"),
                        dcc.Graph(
                            figure=fig,
                            style={"height": "750px"},
                        ),
                        dbc.Button(
                            "View Details",
                            color="primary",
                            href=f"/experiment/{exp_id}",
                            external_link=True,
                            className="mt-3",
                        ),
                    ]
                )
            ],
            className="h-100",
            style={"width": "100%"},
        )

        col = dbc.Col(
            card,
            xs=12,
            sm=12,
            md=6,
            lg=6,
            xl=6,
            className="d-flex align-items-stretch mb-4",
        )
        cards.append(col)

    row = dbc.Row(
        cards,
        className="justify-content-center",
    )

    return dbc.Container(
        [html.H2("Experiments Overview", className="text-center my-4"), row],
        fluid=True,
        className="py-4",
    )


def layout_experiment_detail(exp_id):
    exp_info = EXPERIMENTS[exp_id]
    all_summary_vals, all_detailed_runs = load_and_process_multiple(
        exp_info["json_files"]
    )
    fig1 = create_plot1_figure(all_summary_vals, exp_id)

    store_data = {
        "exp_id": exp_id,
        "detailed_runs": {
            fn: {str(k): v for k, v in runs.items()}
            for fn, runs in all_detailed_runs.items()
        },
    }

    fig1_component = dcc.Graph(
        id="plot1-figure",
        figure=fig1,
        style={"width": "800px", "height": "500px"},
    )
    train_fig_comp = dcc.Graph(
        id="train-fig",
        figure=go.Figure(),
        style={"width": "550px", "display": "inline-block"},
    )
    val_fig_comp = dcc.Graph(
        id="val-fig",
        figure=go.Figure(),
        style={"width": "550px", "display": "inline-block"},
    )

    return dbc.Container(
        [
            html.H3(exp_info["title"]),
            html.P(exp_info["description"]),
            fig1_component,
            dcc.Store(id="detail-data-store", data=store_data),
            html.Hr(),
            train_fig_comp,
            val_fig_comp,
        ],
        fluid=True,
    )


###################
# 4. PAGE ROUTING
###################
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return layout_homepage()
    if pathname.startswith("/experiment/"):
        exp_id = pathname.replace("/experiment/", "")
        if exp_id in EXPERIMENTS:
            return layout_experiment_detail(exp_id)
        else:
            return html.Div([html.H3("Unknown experiment!")])
    return html.Div([html.H3("404 - Not found")])


######################################
# 5. Callback for Plot-2 from Plot-1
######################################
@app.callback(
    [Output("train-fig", "figure"), Output("val-fig", "figure")],
    Input("plot1-figure", "clickData"),
    State("detail-data-store", "data"),
    prevent_initial_call=True,
)
def update_plots_from_click(clickData, data_store):
    if not clickData or not data_store:
        return go.Figure(), go.Figure()

    filename, ds_size = clickData["points"][0]["customdata"]
    all_runs = data_store["detailed_runs"]
    exp_id = data_store["exp_id"]

    if filename not in all_runs:
        return go.Figure(), go.Figure()
    ds_dict = all_runs[filename]
    ds_key = str(ds_size)
    if ds_key not in ds_dict:
        return go.Figure(), go.Figure()

    runs_for_ds = ds_dict[ds_key]
    fig_train, fig_val = create_saturation_figs(runs_for_ds, ds_size, exp_id, filename)
    return fig_train, fig_val


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
