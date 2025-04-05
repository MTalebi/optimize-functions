import streamlit as st
import numpy as np

# Developer: Mohammad Talebi-Kalaleh (mtalebi.com)
# This is the main Streamlit app file that provides:
# 1) A simple interface for defining or choosing a function (via LaTeX or Python),
# 2) Caching of optimization results so we don't re-run expensive computations when only the visualization changes,
# 3) Options for choosing different optimization methods,
# 4) Interactive plotting for 1D, 2D, and 3D functions.

from utils import (
    parse_latex_function,
    parse_python_function,
    sympy_function_to_callable,
    optimize_function,
    generate_plot_1d,
    generate_plot_2d_contour,
    generate_plot_2d_surface,
    generate_plot_3d
)

# We'll store the optimization result in a cache to avoid re-running
# when user only changes visualization settings.
@st.cache_data(show_spinner=False)
def run_optimization(obj_func, x0, method):
    return optimize_function(obj_func, x0, method)

def main():
    st.title("Multivariate Function Optimizer")
    st.write("Hi there! This app helps you find a global (approximate) minimum of your function.")
    st.write("Developer: **Mohammad Talebi-Kalaleh**  |  [mtalebi.com](https://mtalebi.com)")

    # --- Sidebar Input Options ---
    st.sidebar.header("Function Input Mode")
    input_mode = st.sidebar.selectbox(
        "How would you like to provide your function?",
        ["LaTeX Expression", "Python Function"]
    )

    st.sidebar.header("Optimizer Settings")
    method = st.sidebar.selectbox(
        "Choose Optimization Method:",
        ["Gradient Descent", "PSO", "Least Squares"]
    )

    st.sidebar.header("Initial Guess")
    x0_input = st.sidebar.text_input(
        "Comma-separated initial guesses (e.g. '0,0')",
        value="0, 0"
    )

    st.sidebar.header("Variable Names (for LaTeX mode)")
    var_names = st.sidebar.text_input(
        "Comma-separated variable names (e.g. 'x,y')",
        value="x, y"
    )

    # --- Main Panel ---
    if input_mode == "LaTeX Expression":
        st.subheader("Provide or Choose a LaTeX Function")

        # For convenience, we offer some famous test functions to play with.
        st.write("**Pick a well-known function OR define your own custom expression:**")
        predefined_functions = {
            "Sphere (2 vars)": "x^2 + y^2",
            "Rosenbrock (2 vars)": "(1 - x)^2 + 100*(y - x^2)^2",
            "Rastrigin (2 vars)": "x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y) + 20"
        }

        use_predefined = st.radio(
            "Use a predefined test function or a custom expression?",
            ("Predefined", "Custom")
        )

        if use_predefined == "Predefined":
            selected_function = st.selectbox("Choose a test function", list(predefined_functions.keys()))
            latex_str = predefined_functions[selected_function]
        else:
            st.write("Example custom expression: `x^2 + y^2` (No need to include `f(x,y) = ...`)")
            latex_str = st.text_area("LaTeX Expression", value="x^2 + y^2")

    else:
        st.subheader("Provide a Python Function")
        st.write("Here is a sample. You can tweak it or paste your own:")
        py_str = st.text_area(
            "Python Code",
            value=(
                "def objective_function(x, y):\n"
                "    # Just a simple sphere function shifted to (-1, 2)\n"
                "    return (x + 1)**2 + (y - 2)**2\n"
            )
        )

    # When the user clicks 'Optimize', we'll parse & run.
    if st.button("Optimize"):
        # Parse initial guess
        x0_list = [float(v.strip()) for v in x0_input.split(",")]
        x0 = np.array(x0_list)

        if input_mode == "LaTeX Expression":
            # Convert the LaTeX expression to a sympy expression, then to a callable function
            expr = parse_latex_function(latex_str)
            var_list = [v.strip() for v in var_names.split(",")]
            obj_func_sympy = sympy_function_to_callable(expr, var_list)

            # Wrap so that optimize_function can call with x-array
            def wrapped_func(x_array):
                return obj_func_sympy(*x_array)

            # Run optimization with caching
            res = run_optimization(wrapped_func, x0, method)

        else:
            # Parse Python function from the provided code block
            user_func = parse_python_function(py_str)

            def wrapped_func(x_array):
                return user_func(*x_array)

            # Run optimization with caching
            res = run_optimization(wrapped_func, x0, method)

        # --- Display Results ---
        st.write("### Optimization Results")
        st.write(f"**Optimal Point**: {res.x}")
        st.write(f"**Optimal Objective Value**: {res.fun}")

        # --- Plot based on dimension ---
        dim = len(res.x)
        if dim == 1:
            fig = generate_plot_1d(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 2:
            plot_type = st.selectbox("Select Plot Type for 2D", ["3D Surface", "2D Contour"])
            if plot_type == "3D Surface":
                fig = generate_plot_2d_surface(wrapped_func, res.x)
            else:
                fig = generate_plot_2d_contour(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 3:
            fig = generate_plot_3d(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Plotting for dimensions higher than 3 is not currently supported.")

if __name__ == "__main__":
    main()
