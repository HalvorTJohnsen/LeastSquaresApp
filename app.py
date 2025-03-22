import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
import plotly.graph_objects as go
from scipy.optimize import minimize
import os
import copy

APP_VERSION = "1.0.0"  # Change this to your current version
LAST_UPDATED = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

clicked_points = set()  # Track clicked points for removal



# Generate synthetic data for fitting
def generate_data(n_points=100, noise=0.1, outlier_ratio=0, true_complexity=2):
    """
    Generates synthetic data points (x, y, z) with a true underlying model
    governed by the true_complexity parameter. Adds noise and optional outliers.
    """
    np.random.seed(0)
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)

    # True model based on the selected complexity
    if true_complexity == 1:  # Linear
        z = 2 * x + 3 * y + 1  # Example linear model
    elif true_complexity == 2:  # Quadratic
        z = 2 * x**2 - 3 * y**2 + x * y + 1  # Example quadratic model
    elif true_complexity == 3:  # Cubic
        z = x**3 - y**3 + 2 * x**2 * y - y**2 * x + 1  # Example cubic model
    elif true_complexity == 4:  # Sinusoidal
        z = 2 * np.sin(2 * np.pi * x) + 3 * np.cos(2 * np.pi * y) + 1  # Example sinusoidal model
    elif true_complexity == 5:  # Mixed complex
        z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + x**2 - y**2 + x * y  # Example mixed model
    else:
        raise ValueError("Unsupported true complexity level")

    # Add noise to the data
    z += noise * np.random.randn(n_points)

    # Add outliers
    if outlier_ratio > 0:
        n_outliers = int(outlier_ratio * n_points)
        z[:n_outliers] += np.random.uniform(-10, 10, n_outliers)

    return x, y, z

# Fitting functions
fit_types = {
    "polynomial": lambda x, y, degree: np.vstack([((x ** i) * (y ** j)).flatten() 
                                                  for i in range(degree + 1) 
                                                  for j in range(degree + 1 - i)]).T,
    "linear": lambda x, y, _: np.vstack([x, y, np.ones_like(x)]).T,  # Ensure linear is included
    "quadratic": lambda x, y, _: np.vstack([
        x**2, y**2, x * y, x, y, np.ones_like(x)
    ]).T,
    "cubic": lambda x, y, _: np.vstack([
        x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x * y, x, y, np.ones_like(x)
    ]).T,
    "sinusoidal": lambda x, y, _: np.vstack([
        np.sin(2 * np.pi * x), np.cos(2 * np.pi * y), np.ones_like(x)
    ]).T,
}


def fit_function(x, y, z, degree, fit_type, regularization=0, reg_type="L2", cost_function='squared'):
    X = fit_types[fit_type](x, y, degree)

    # Define cost function
    def cost(params):
        z_pred = X @ params
        residuals = z - z_pred
        regularization_term = 0
        if reg_type == "L1":
            regularization_term = regularization * np.sum(np.abs(params))
        elif reg_type == "L2":
            regularization_term = regularization * np.sum(params ** 2)
        elif reg_type == "L0":
            regularization_term = regularization * np.sum(params != 0)
        elif reg_type == "None":
            regularization_term = 0
        elif reg_type == "elasticnet":
            alpha = 0.5  # You can make this adjustable later
            l1_term = np.sum(np.abs(params))
            l2_term = np.sum(params ** 2)
            regularization_term = regularization * (alpha * l1_term + (1 - alpha) * l2_term)


        
        if cost_function == 'squared':
            return np.sum(residuals ** 2) + regularization_term
        elif cost_function == 'absolute':
            return np.sum(np.abs(residuals)) + regularization_term

    # Initial guess
    initial_params = np.zeros(X.shape[1])

    # Optimize
    result = minimize(cost, initial_params, method='L-BFGS-B')

    return result.x, cost(result.x)

# Create Dash app
app = dash.Dash(__name__)
server = app.server  # Expose Flask server for Gunicorn
app.title = "3D Fit with Least Squares"

# Global data
x_data, y_data, z_data = generate_data(n_points=100, noise=0.1, outlier_ratio=0.1)

@app.callback(
    Output("debug-output", "children"),
    Input("add-point-button", "n_clicks"),
    prevent_initial_call=True
)
def test_button(n_clicks):
    print("DEBUG: Button callback fired!")
    return f"Button clicked {n_clicks} times"



# Update the layout to include the fitted model selection
app.layout = html.Div([
    dcc.Store(id="stored-data"),
    html.Div(id="debug-output", style={"marginTop": "10px", "color": "green"}),
    html.Div([
        html.H1("3D Fit with Least Squares"),
        html.Div([
            html.P(f"Version: {APP_VERSION}", style={"margin-right": "15px", "display": "inline"}),
            html.P(f"Last Updated: {LAST_UPDATED}", style={"display": "inline"})
        ], style={"position": "absolute", "top": "10px", "right": "20px", "textAlign": "right", "fontSize": "14px"})
    ], style={"position": "relative"}),
    html.Div([
        html.Label("Plot Type:"),
        dcc.RadioItems(
            id="plot-dimension-toggle",
            options=[
                {"label": "3D", "value": "3D"},
                {"label": "2D", "value": "2D"},
            ],
            value="3D",  # Default view
            labelStyle={"display": "inline-block", "margin-right": "10px"}
        )
    ], style={"marginBottom": "20px"}),
    html.Div([
        html.Div([
            html.Label("True Model Type:"),
            dcc.Slider(
                id="complexity-slider",
                min=1,
                max=5,
                step=1,
                value=2,
                marks={
                    1: "Linear",
                    2: "Quadratic",
                    3: "Cubic",
                    4: "Sinusoidal",
                    5: "Mixed Complex"
                }
            ),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Fitted Model Type:"),
            dcc.Dropdown(
                id="fit-type-dropdown",
                options=[
                    {"label": "Linear", "value": "linear"},
                    {"label": "Quadratic", "value": "quadratic"},
                    {"label": "Cubic", "value": "cubic"},
                    {"label": "Polynomial", "value": "polynomial"},
                    {"label": "Sinusoidal", "value": "sinusoidal"}
                ],
                value="polynomial"
            ),

        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Degree (For Polynomials only):"),
            dcc.Slider(id="degree-slider", min=1, max=5, step=1, value=2, marks={i: str(i) for i in range(1, 6)}),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Number of Samples:"),
            dcc.Input(id="npoints-input", type="number", value=100, min=10, max=1000, step=10),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Standard deviation of the noise:"),
            dcc.Slider(
                id="noise-slider",
                min=-4,
                max=1,
                step=0.5,
                value=-1,
                marks={
                    -4: "10⁻⁴",
                    -3: "10⁻³",
                    -2: "10⁻²",
                    -1: "10⁻¹",
                    0: "1",
                    1: "10"
                },
            ),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Outlier Ratio:"),
            dcc.Slider(id="outlier-slider", min=0, max=0.5, step=0.05, value=0.1, marks={i / 10: str(i / 10) for i in range(6)}),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Regularization parameter:"),
            dcc.Slider(
                id="regularization-slider",
                min=-4,
                max=5,  # ✅ now goes from 10^-4 to 10^5
                step=0.5,
                value=-1,
                marks={i: f"10^{i}" for i in range(-4, 6)},  # ✅ include 5
            ),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Regularization Type:"),
            dcc.Dropdown(
                id="regtype-dropdown",
                options=[
                    {"label": "None", "value": "None"},
                    {"label": "L0", "value": "L0"},
                    {"label": "L1", "value": "L1"},
                    {"label": "L2", "value": "L2"},
                    {"label": "Elastic Net", "value": "elasticnet"}  # ✅ Add this line
                ],
                value="L2"
            ),


        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Cost Function:"),
            dcc.Dropdown(
                id="cost-dropdown",
                options=[{"label": "Squared Error", "value": "squared"},
                         {"label": "Absolute Error", "value": "absolute"}],
                value="squared"
            ),
        ], style={"marginBottom": "20px"}),
        html.Div([
            html.Label("Manually Add a Sample:"),
            html.Div(dcc.Input(id="input-x", type="number", placeholder="x", debounce=True), style={"marginBottom": "10px"}),
            html.Div(dcc.Input(id="input-y", type="number", placeholder="y", debounce=True), style={"marginBottom": "10px"}),
            html.Div(dcc.Input(id="input-z", type="number", placeholder="z", debounce=True), style={"marginBottom": "10px"}),
            html.Button("Add Point", id="add-point-button", n_clicks=0),
        ], style={"marginTop": "30px"}),

    ], style={"margin": "20px", "width": "40%", "display": "inline-block", "verticalAlign": "top"}),
    html.Div([
        dcc.Graph(id="3d-plot"),
        html.Div(id="metrics-output", style={"marginTop": "20px"}),
        html.Div(id="cost-equation", style={"marginTop": "20px", "fontSize": "18px"}),
    ], style={"width": "55%", "display": "inline-block", "verticalAlign": "top"}),
])

@app.callback(
    [Output("3d-plot", "figure"),
     Output("metrics-output", "children"),
     Output("cost-equation", "children")],
    [Input("fit-type-dropdown", "value"),
     Input("degree-slider", "value"),
     Input("complexity-slider", "value"),
     Input("npoints-input", "value"),
     Input("noise-slider", "value"),
     Input("outlier-slider", "value"),
     Input("regularization-slider", "value"),
     Input("regtype-dropdown", "value"),
     Input("cost-dropdown", "value"),
     Input("plot-dimension-toggle", "value"),
     Input("stored-data", "data")],
    [State("3d-plot", "figure")]
)

def update_plot(fit_type, degree, true_complexity, n_points, noise, outlier_ratio, regularization, reg_type, cost_function, plot_dim, current_figure, stored_data):
    # Ensure stored_data has x, y, z
    # Handle missing or corrupted stored data
    noise = 10 ** noise
    regularization = 10 ** regularization

    if stored_data is None or not all(k in stored_data for k in ("x", "y", "z")):
        print("stored_data missing or incomplete, generating fresh data")
        x_data, y_data, z_data = generate_data(n_points, noise, outlier_ratio, true_complexity)
        stored_data = {"x": x_data.tolist(), "y": y_data.tolist(), "z": z_data.tolist()}
    else:
        print(f"Using stored data: {len(stored_data['x'])} points")
        x_data = np.array(stored_data["x"])
        y_data = np.array(stored_data["y"])
        z_data = np.array(stored_data["z"])

    print(f"Using stored data: {len(x_data)} points")
    print("Raw stored_data in update_plot:", stored_data)

    # print("===== DEBUGGING INFO =====")
    # print(f"Received fit_type: '{fit_type}'")  # Check what is actually received
    # print(f"Available fit_types: {list(fit_types.keys())}")  # Print the dictionary keys
    
    # # Check if the issue is a hidden whitespace problem
    # if fit_type.strip() not in fit_types:
    #     print(f"Possible whitespace issue: '{fit_type}' != '{fit_type.strip()}'")
    
    # # Check if it is a case-sensitivity issue
    # lowercase_keys = {k.lower(): k for k in fit_types.keys()}  # Create a mapping of lowercase keys
    # if fit_type.lower() in lowercase_keys:
    #     corrected_fit_type = lowercase_keys[fit_type.lower()]
    #     print(f"Case issue detected. Converting '{fit_type}' to '{corrected_fit_type}'")
    #     fit_type = corrected_fit_type  # Fix it automatically

    # # Check final condition before raising an error
    # if fit_type not in fit_types:
    #     print(f"!!! ERROR: fit_type '{fit_type}' is missing from fit_types at runtime !!!")
    #     print(f"Full fit_types dictionary at error moment: {fit_types}")

    #     # Explicitly check if 'linear' is accessible
    #     if "linear" in fit_types:
    #         print(f"fit_types['linear'] is accessible: {fit_types['linear']}")
    #     else:
    #         print("WARNING: 'linear' is missing from fit_types despite being listed!")

    #     raise ValueError(f"Unsupported fit type: {fit_type}")


    # Fit the data using the chosen model
    params, cost_val = fit_function(x_data, y_data, z_data, degree, fit_type, regularization, reg_type, cost_function)

    # Design matrix and predicted values
    X = fit_types[fit_type](x_data, y_data, degree)
    z_fit = X @ params

    # Define hypothesis function text
    if fit_type == "polynomial":
        hypothesis_text = "ẑ = " + " + ".join([
            f"β{i}{j}"
            + (f"x^{i}" if i > 1 else ("x" if i == 1 else ""))
            + (f"y^{j}" if j > 1 else ("y" if j == 1 else ""))
            for i in range(degree+1) for j in range(degree+1-i)
        ])
    elif fit_type == "sinusoidal":
        hypothesis_text = "ẑ = β₁sin(x) + β₂cos(y)"
    else:
        hypothesis_text = ""

    reg_term_map = {
        "L0": "||β||₀",
        "L1": "||β||₁",
        "L2": "||β||₂²",
        "elasticnet": "0.5 · ||β||₁ + 0.5 · ||β||₂²",
        "None": "0"
    }
    reg_term = reg_term_map.get(reg_type, "0")


    if cost_function == "squared":
        cost_text = f"J(β) = Σ(zᵢ - ẑᵢ)² + {regularization} · {reg_term}"
    elif cost_function == "absolute":
        cost_text = f"J(β) = Σ|zᵢ - ẑᵢ| + {regularization} · {reg_term}"
    else:
        cost_text = ""

    # Generate the plot
    x_plot = np.linspace(-1, 1, 50)
    y_plot = np.linspace(-1, 1, 50)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)

    if fit_type == "linear":
        z_plot = params[0] * x_plot + params[1] * y_plot + params[2]  # Correct formula
    elif fit_type in ["polynomial", "quadratic", "cubic"]:
        z_plot = np.sum([params[k] * (x_plot ** i) * (y_plot ** j)
                        for k, (i, j) in enumerate([(i, j) for i in range(degree+1) for j in range(degree+1-i)])], axis=0)
    elif fit_type == "sinusoidal":
        z_plot = params[0] * np.sin(x_plot) + params[1] * np.cos(y_plot)
    else:
        print(f"Invalid fit_type received: {fit_type}")
        print(f"Available types: {list(fit_types.keys())}")
        raise ValueError(f"Unsupported fit type: {fit_type}")

    if plot_dim == "2D":
        # Generate a clean grid of y values
        y_line = np.linspace(-1, 1, 100)
        x_fixed = np.mean(x_data)
        
        # Build the design matrix for fixed x and sweeping y
        X_line = fit_types[fit_type](np.full_like(y_line, x_fixed), y_line, degree)
        z_line = X_line @ params

        fig = go.Figure()

        # Add the actual noisy data (still showing model/data mismatch)
        sorted_idx = np.argsort(y_data)
        fig.add_trace(go.Scatter(x=y_data[sorted_idx], y=z_data[sorted_idx],
                                mode="markers", name="Data Points"))

        # Add the clean, fixed-x model prediction
        fig.add_trace(go.Scatter(x=y_line, y=z_line,
                                mode="lines", name="Model Fit"))

        fig.update_layout(
            title="2D View (z vs y, x fixed)",
            xaxis_title="x",  # We still rename it
            yaxis_title="y",
            margin=dict(l=0, r=0, b=0, t=40)
        )

    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data,
                                mode='markers', marker=dict(size=5, color='red'), name='Data Points'))
        fig.add_trace(go.Surface(x=x_plot, y=y_plot, z=z_plot,
                                colorscale='Viridis', opacity=0.7, name='Fit Surface'))

        if current_figure and "scene" in current_figure.get("layout", {}):
            fig.update_layout(scene_camera=current_figure["layout"]["scene"].get("camera", {}))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )


    residual_error = np.sum((z_data - z_fit) ** 2)
    metrics = [
        html.P(f"Cost Function Value: {cost_val:.4f}"),
        html.P(f"Residual Error (Sum of Squared Residuals): {residual_error:.4f}"),
    ]
    cost_equation = html.Div([
        html.P("True Model:"),
        html.Pre("Generated internally based on selected complexity.", style={"fontSize": "16px", "whiteSpace": "pre-wrap"}),

        html.P("Estimated Model:"),
        html.Pre(hypothesis_text, style={"fontSize": "16px", "whiteSpace": "pre-wrap"}),

        html.P("Cost Function:"),
        html.Pre(cost_text, style={"fontSize": "16px", "whiteSpace": "pre-wrap"}),
    ])


    return fig, metrics, cost_equation

@app.callback(
    Output("stored-data", "data"),
    Input("add-point-button", "n_clicks"),
    State("stored-data", "data"),
    State("input-x", "value"),
    State("input-y", "value"),
    State("input-z", "value"),
    prevent_initial_call=True
)
def add_manual_point(n_clicks, data, x_input, y_input, z_input):
    print("Triggered add_manual_point callback")
    if x_input is None or y_input is None or z_input is None:
        raise dash.exceptions.PreventUpdate
    print(f"Adding point: ({x_input}, {y_input}, {z_input})")
    if data is None or not all(k in data for k in ("x", "y", "z")):
        data = {"x": [], "y": [], "z": []}
    data["x"].append(x_input)
    data["y"].append(y_input)
    data["z"].append(z_input)
    return data


# Run the app
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

    