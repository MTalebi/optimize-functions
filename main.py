import streamlit as st
import numpy as np

# Developer: Mohammad Talebi-Kalaleh (mtalebi.com)
# This main Streamlit app file:
#  - Stores optimization results in session_state so we don't re-run on small UI changes.
#  - Caches the optimization but only with hashable inputs (strings, lists, etc.).
#  - Lets the user switch between 2D/3D plots without losing their results or reverting to defaults.

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

@st.cache_data(show_spinner=False)
def run_optimization_cached(input_mode, func_text, var_names_str, x0_list, method):
    """
    Parses user input (LaTeX or Python) and runs the optimization (cached).
    Returns the scipy result object.
    """
    import sympy  # local import, just to be safe for parse_latex
    import numpy as np

    x0 = np.array(x0_list, dtype=float)

    if input_mode == "LaTeX Expression":
        expr = parse_latex_function(func_text)
        var_list = [v.strip() for v in var_names_str.split(",")]
        obj_func_sympy = sympy_function_to_callable(expr, var_list)

        def wrapped_func(x_array):
            return obj_func_sympy(*x_array)

        res = optimize_function(wrapped_func, x0, method)
    else:
        user_func = parse_python_function(func_text)

        def wrapped_func(x_array):
            return user_func(*x_array)

        res = optimize_function(wrapped_func, x0, method)

    return res


def main():
    st.title("Multivariate Function Optimizer")
    st.write("Hi there! This app finds global (approximate) minima for your function.")
    st.write("Developer: **Mohammad Talebi-Kalaleh**  |  [mtalebi.com](https://mtalebi.com)")

    # --------------------------------------------------------------------------
    # Initialize session state variables if needed
    # --------------------------------------------------------------------------
    if "res" not in st.session_state:
        st.session_state["res"] = None  # will hold the optimization result
    if "wrapped_plot_func" not in st.session_state:
        st.session_state["wrapped_plot_func"] = None
    if "dim" not in st.session_state:
        st.session_state["dim"] = None

    # This is where we'll store the text of the function for re-parsing
    if "func_text" not in st.session_state:
        st.session_state["func_text"] = ""
    if "var_names" not in st.session_state:
        st.session_state["var_names"] = ""

    # For 2D scenarios, store user's chosen plot type in session_state
    if "plot_type_2d" not in st.session_state:
        st.session_state["plot_type_2d"] = "3D Surface"

    # --------------------------------------------------------------------------
    # Sidebar Input
    # --------------------------------------------------------------------------
    st.sidebar.header("Function Input Mode")
    input_mode = st.sidebar.selectbox(
        "How would you like to provide your function?",
        ["LaTeX Expression", "Python Function"]
    )

    st.sidebar.header("Optimizer Settings")
    method = st.sidebar.selectbox("Choose Optimization Method:", 
                                  ["Gradient Descent", "PSO", "Least Squares"])

    st.sidebar.header("Initial Guess")
    x0_input = st.sidebar.text_input("Comma-separated initial guesses (e.g. '0,0')", "0, 0")

    st.sidebar.header("Variable Names (LaTeX Mode)")
    var_names_str = st.sidebar.text_input("Comma-separated variable names", "x, y")

    # --------------------------------------------------------------------------
    # Main Panel for function definition
    # --------------------------------------------------------------------------
    if input_mode == "LaTeX Expression":
        st.subheader("Provide or Choose a LaTeX Function")

        predefined_functions = {
            "Sphere (2 vars)": "x^2 + y^2",
            "Rosenbrock (2 vars)": "(1 - x)^2 + 100*(y - x^2)^2",
            "Rastrigin (2 vars)": "x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y) + 20",
        }
        use_predefined = st.radio("Predefined function or custom?", ("Predefined", "Custom"))

        if use_predefined == "Predefined":
            selected_function = st.selectbox(
                "Choose a test function", 
                list(predefined_functions.keys())
            )
            latex_str = predefined_functions[selected_function]
        else:
            st.write("Example: `x^2 + y^2` (no need to include `f(x,y)=...`).")
            latex_str = st.text_area("LaTeX Expression", "x^2 + y^2")

        st.session_state["func_text"] = latex_str
        st.session_state["var_names"] = var_names_str

    else:
        st.subheader("Provide a Python Function")
        sample_py = (
            "def objective_function(x, y):\n"
            "    # Simple sphere shifted to (-1, 2)\n"
            "    return (x + 1)**2 + (y - 2)**2\n"
        )
        py_str = st.text_area("Python Code", sample_py)
        st.session_state["func_text"] = py_str
        st.session_state["var_names"] = var_names_str  # might not be used in Python mode

    # --------------------------------------------------------------------------
    # Optimization Button
    # --------------------------------------------------------------------------
    if st.button("Optimize"):
        # Convert x0 to list of floats
        x0_list = [float(v.strip()) for v in x0_input.split(",")]

        # Run the cached optimization
        res = run_optimization_cached(
            input_mode=input_mode,
            func_text=st.session_state["func_text"],
            var_names_str=st.session_state["var_names"],
            x0_list=x0_list,
            method=method
        )

        # Store in session state so we can keep the result
        st.session_state["res"] = res
        st.session_state["dim"] = len(res.x)

        # Now also store a "wrapped" function for plotting:
        if input_mode == "LaTeX Expression":
            expr_plot = parse_latex_function(st.session_state["func_text"])
            var_list_plot = [v.strip() for v in st.session_state["var_names"].split(",")]

            def wrapped_plot_func(x_array):
                return sympy_function_to_callable(expr_plot, var_list_plot)(*x_array)

        else:
            user_func_plot = parse_python_function(st.session_state["func_text"])

            def wrapped_plot_func(x_array):
                return user_func_plot(*x_array)

        st.session_state["wrapped_plot_func"] = wrapped_plot_func

    # --------------------------------------------------------------------------
    # If we have a valid result in session_state, show it + allow re-plotting
    # without losing everything
    # --------------------------------------------------------------------------
    if st.session_state["res"] is not None and st.session_state["wrapped_plot_func"] is not None:
        res = st.session_state["res"]
        wrapped_plot_func = st.session_state["wrapped_plot_func"]
        dim = st.session_state["dim"]

        st.write("### Optimization Results")
        st.write(f"**Optimal Point**: {res.x}")
        st.write(f"**Optimal Objective Value**: {res.fun}")

        # Let the user pick or switch the plot type for 2D,
        # and store that preference in session_state so it doesn't reset.
        if dim == 1:
            fig = generate_plot_1d(wrapped_plot_func, res.x)
            st.plotly_chart(fig, use_container_width=True)

        elif dim == 2:
            # The user’s chosen plot type is stored in st.session_state["plot_type_2d"].
            # We'll show a selectbox that updates it if changed.
            plot_type_2d = st.selectbox(
                "Select Plot Type for 2D",
                ["3D Surface", "2D Contour"],
                index=["3D Surface", "2D Contour"].index(st.session_state["plot_type_2d"])
            )
            if plot_type_2d != st.session_state["plot_type_2d"]:
                st.session_state["plot_type_2d"] = plot_type_2d

            if st.session_state["plot_type_2d"] == "3D Surface":
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
