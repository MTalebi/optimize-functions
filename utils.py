import sympy
import numpy as np
import re
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from scipy.optimize import minimize
import plotly.graph_objects as go

def parse_latex_function(latex_str):
    """
    Parse a LaTeX string into a symbolic sympy function.
    Example of acceptable LaTeX input: 
        f(x, y) = x^2 + y^2
        or 
        x^2 + 3*x + 1
    """
    # Remove common LaTeX function wrappers if present.
    # We'll try a quick approach that strips out "f(...)=..." if found.
    # We only keep the right side of the equation for parsing.
    latex_str = latex_str.strip()
    # A naive approach to remove f(...) if present:
    latex_str = re.sub(r'f\(.*?\)\s*=\s*', '', latex_str)

    # Parse the expression using sympy's parse_latex
    try:
        expr = parse_latex(latex_str)
    except Exception:
        # If latex parsing fails, fall back to sympy expression parsing
        expr = parse_expr(latex_str)

    return expr


def parse_python_function(py_str):
    """
    Dynamically create a Python function from user code input.
    The user might provide something like:
        def objective_function(x, y):
            return x**2 + y**2
    We assume the function is named 'objective_function' with
    the correct signature.
    """
    local_env = {}
    # Execute the code block to define the function in local_env
    try:
        exec(py_str, {}, local_env)
        obj_func = local_env['objective_function']
    except Exception as e:
        raise ValueError(f"Error parsing function: {e}")

    return obj_func


def sympy_function_to_callable(expr, var_names):
    """
    Convert a sympy expression into a Python callable that can
    be called like func([x1, x2, ...]).
    """
    variables = sympy.symbols(var_names)
    func = sympy.lambdify(variables, expr, 'numpy')
    return func


def optimize_function(obj_func, x0, method='Gradient Descent'):
    """
    Wrapper around scipy.optimize.minimize.
    Accepts an objective function obj_func that takes a numpy array [x1, x2, ...]
    and returns a scalar value.
    x0 is the initial guess array.
    method can be: 'Gradient Descent' (BFGS), 'PSO', or 'Least Squares' (though 
    least_squares is typically for residual minimization, we demonstrate a possible approach).
    """
    # For demonstration, we'll treat:
    # 'Gradient Descent' => method='BFGS'
    # 'PSO' => a naive approach or alternative global method (not implemented in SciPy by default).
    # We can do a placeholder approach or differential_evolution for a global approximation.
    # 'Least Squares' => we'll pretend we do standard minimize with method='BFGS' again for simplicity
    # or we could raise a NotImplementedError for methods not fully coded.
    if method == 'Gradient Descent':
        res = minimize(obj_func, x0, method='BFGS')
    elif method == 'Least Squares':
        # We'll just call the same routine in this demonstration 
        # (In practice, you'd do something specialized with residuals)
        res = minimize(obj_func, x0, method='BFGS')
    elif method == 'PSO':
        # As an example, let's do a naive global approach using differential_evolution
        from scipy.optimize import differential_evolution
        
        # We guess a default domain for all variables from -10 to 10
        bounds = [(-10, 10) for _ in x0]
        global_res = differential_evolution(obj_func, bounds=bounds)
        # Use that global result as x0 in local refine
        res = minimize(obj_func, global_res.x, method='BFGS')
    else:
        # Default fallback
        res = minimize(obj_func, x0, method='BFGS')
        
    return res


def generate_plot_1d(obj_func, x_opt):
    """
    Generate a 1D plot using Plotly of the function over a range.
    x_opt is the found optimum point (array-like).
    """
    x_vals = np.linspace(-10, 10, 200)
    y_vals = [obj_func(np.array([x])) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Objective'))
    fig.add_trace(go.Scatter(x=[x_opt[0]], y=[obj_func(x_opt)], 
                             mode='markers', 
                             marker=dict(color='red', size=10),
                             name='Optimal Point'))
    fig.update_layout(title='1D Objective Function', xaxis_title='x', yaxis_title='f(x)')
    return fig


def generate_plot_2d_contour(obj_func, x_opt):
    """
    Generate a 2D contour plot for a function of two variables, x & y.
    Safely handles potential math/domain errors by substituting NaN.
    """
    import numpy as np
    import plotly.graph_objects as go
    
    # Create a grid
    x_vals = np.linspace(-10, 10, 50)
    y_vals = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Make sure Z is a float array
    Z = np.zeros_like(X, dtype=float)

    # Safely evaluate the function on each grid point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                val = obj_func([X[i, j], Y[i, j]])
                # If the function returns something invalid or complex, catch it:
                if np.iscomplex(val):
                    Z[i, j] = np.nan
                else:
                    Z[i, j] = float(val)
            except:
                Z[i, j] = np.nan

    # Build the contour figure
    fig = go.Figure(
        data=[
            go.Contour(
                z=Z,
                x=x_vals,
                y=y_vals,
                colorscale="Viridis",
                # We configure "contours" via a dictionary:
                contours=dict(
                    coloring="heatmap",   # or "lines", etc.
                    showlines=False,
                    smoothing=1.3
                )
            )
        ]
    )

    # Mark the optimum on the contour
    fig.add_trace(
        go.Scatter(
            x=[x_opt[0]],
            y=[x_opt[1]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Optimal Point'
        )
    )

    fig.update_layout(
        title='2D Contour Plot',
        xaxis_title='x',
        yaxis_title='y'
    )

    return fig


def generate_plot_2d_surface(obj_func, x_opt):
    """
    Generate a 3D surface plot for a function of two variables, x & y.
    """
    x_vals = np.linspace(-10, 10, 50)
    y_vals = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj_func([X[i, j], Y[i, j]])
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    # Add a scatter for the optimum
    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]],
        y=[x_opt[1]],
        z=[obj_func(x_opt)],
        mode='markers',
        marker=dict(color='red', size=5),
        name='Optimal Point'
    ))
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x,y)'
        )
    )
    return fig


def generate_plot_3d(obj_func, x_opt):
    """
    If the user has three variables, we can attempt to show some 3D contour.
    This is purely illustrative, as a full 3D contour is tricky to visualize.
    We'll fix one variable and vary the other two, or do a slice approach.
    For simplicity, we fix the third variable at the optimum value and plot the first two.
    """
    # Let's fix z at optimum value
    # x1, x2, x3 -> we fix x3 at x_opt[2]
    z_fixed = x_opt[2]
    
    x_vals = np.linspace(-10, 10, 40)
    y_vals = np.linspace(-10, 10, 40)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj_func([X[i, j], Y[i, j], z_fixed])

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    # Show the optimum projected
    fig.add_trace(go.Scatter3d(
        x=[x_opt[0]],
        y=[x_opt[1]],
        z=[obj_func(x_opt)],
        mode='markers',
        marker=dict(color='red', size=4),
        name='Optimal'
    ))
    fig.update_layout(
        title='3D Slice (Fixing 3rd variable at optimum)',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1,x2,z_fixed)'
        )
    )
    return fig
