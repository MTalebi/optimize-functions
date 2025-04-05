import streamlit as st
import numpy as np

# Developer: Mohammad Talebi-Kalaleh (mtalebi.com)
# This main Streamlit app file:
# 1) Allows defining or choosing a function (via LaTeX or Python code),
# 2) Uses caching but only with hashable inputs (like strings/arrays) to avoid hash errors,
# 3) Runs various optimization methods,
# 4) Provides interactive plotting for 1D, 2D, and 3D functions.

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

# ------------------------------------------------------------------------------
# Caching function: we pass only hashable data (strings, lists, etc.).
# We re-parse inside this function so there's no unhashable function object in the signature.
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_optimization_cached(
    input_mode: str,
    func_text: str,
    var_names_str: str,
    x0_list: list,
    method: str
):
    """
    Parses user input (either LaTeX or Python code) and runs the optimization.
    Returns the optimization result object from scipy.
    """
    # Convert list to a numpy array for optimization
    x0 = np.array(x0_list, dtype=float)

    if input_mode == "LaTeX Expression":
        expr = parse_latex_function(func_text)
        var_list = [v.strip() for v in var_names_str.split(",")]
        # Convert sympy to a Python callable
        obj_func_sympy = sympy_function_to_callable(expr, var_list)

        def wrapped_func(x_array):
            return obj_func_sympy(*x_array)

        res = optimize_function(wrapped_func, x0, method)

    else:  # Python Function mode
        user_func = parse_python_function(func_text)

        def wrapped_func(x_array):
            return user_func(*x_array)

        res = optimize_function(wrapped_func, x0, method)

    return res


def main():
    st.title("Multivariate Function Optimizer")
    st.write("Hi there! This app helps you find a global (approximate) minimum of your function.")
    st.write("Developer: **Mohammad Talebi-Kalaleh**  |  [mtalebi.com](https://mtalebi.com)")

    # --------------------------------------------------------------------------
    # Sidebar: Input Options
    # --------------------------------------------------------------------------
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

    st.sidebar.header("Variable Names (LaTeX Mode)")
    var_names = st.sidebar.text_input(
        "Comma-separated variable names (e.g. 'x,y')",
        value="x, y"
    )

    # --------------------------------------------------------------------------
    # Main Panel
    # --------------------------------------------------------------------------
    if input_mode == "LaTeX Expression":
        st.subheader("Provide or Choose a LaTeX Function")

        # Offer some famous test functions for convenience
        st.write("**Pick a well-known function OR define your own custom expression:**")
        predefined_functions = {
            "Sphere (2 vars)": "x^2 + y^2",
            "Rosenbrock (2 vars)": "(1 - x)^2 + 100*(y - x^2)^2",
            "Rastrigin (2 vars)": "x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y) + 20"
        }

        use_predefined = st.radio(
            "Predefined test function or custom expression?",
            ("Predefined", "Custom")
        )

        if use_predefined == "Predefined":
            selected_function = st.selectbox("Choose a test function", list(predefined_functions.keys()))
            latex_str = predefined_functions[selected_function]
        else:
            st.write("Example custom expression: `x^2 + y^2` (no need to include `f(x,y)=...`)")
            latex_str = st.text_area("LaTeX Expression", value="x^2 + y^2")

        func_text = latex_str  # for caching function

    else:
        st.subheader("Provide a Python Function")
        st.write("Sample below. You can modify or replace it with your own code:")
        py_str = st.text_area(
            "Python Code",
            value=(
                "def objective_function(x, y):\n"
                "    # A simple sphere shifted to (-1, 2)\n"
                "    return (x + 1)**2 + (y - 2)**2\n"
            )
        )
        func_text = py_str  # for caching function

    # --------------------------------------------------------------------------
    # Run Optimization
    # --------------------------------------------------------------------------
    if st.button("Optimize"):
        # Convert x0_input to a list of floats
        x0_list = [float(v.strip()) for v in x0_input.split(",")]

        # Call our cached optimization function
        res = run_optimization_cached(
            input_mode=input_mode,
            func_text=func_text,
            var_names_str=var_names,
            x0_list=x0_list,
            method=method
        )

        # Display results
        st.write("### Optimization Results")
        st.write(f"**Optimal Point**: {res.x}")
        st.write(f"**Optimal Objective Value**: {res.fun}")

        # Dimension-based plotting
        dim = len(res.x)
        # Because we re-parse inside the cache function, let's reconstruct the same callable for plotting:
        if input_mode == "LaTeX Expression":
            expr_plot = parse_latex_function(func_text)
            vars_plot = [v.strip() for v in var_names.split(",")]
            obj_func_sympy_plot = sympy_function_to_callable(expr_plot, vars_plot)

            def wrapped_plot_func(x_array):
                return obj_func_sympy_plot(*x_array)
        else:
            user_func_plot = parse_python_function(func_text)

            def wrapped_plot_func(x_array):
                return user_func_plot(*x_array)

        if dim == 1:
            fig = generate_plot_1d(wrapped_plot_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 2:
            plot_type = st.selectbox("Select Plot Type for 2D", ["3D Surface", "2D Contour"])
            if plot_type == "3D Surface":
                fig = generate_plot_2d_surface(wrapped_plot_func, res.x)
            else:
                fig = generate_plot_2d_contour(wrapped_plot_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 3:
            fig = generate_plot_3d(wrapped_plot_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Plotting for dimensions higher than 3 is not currently supported.")


if __name__ == "__main__":
    main()
