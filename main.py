import streamlit as st
import numpy as np

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

def main():
    st.title("Multivariate Function Optimizer")
    st.write("Welcome! This app finds a global (approximate) minimum of a user-provided function using various optimization methods.")
    
    # Sidebar for user input selection
    st.sidebar.header("Input Options")
    input_mode = st.sidebar.selectbox("Select input mode:", ["LaTeX Expression", "Python Function"])
    
    st.sidebar.header("Optimizer Settings")
    method = st.sidebar.selectbox("Choose Optimization Method", ["Gradient Descent", "PSO", "Least Squares"])
    
    st.sidebar.header("Initial Guess")
    st.sidebar.write("Provide a comma-separated list of initial guesses (numbers).")
    x0_input = st.sidebar.text_input("x0", "0, 0")
    
    st.sidebar.header("Variable Names")
    var_names = st.sidebar.text_input("Variable Names (comma-separated)", "x, y")
    
    st.sidebar.write("Click **Optimize** when ready.")
    
    if input_mode == "LaTeX Expression":
        st.subheader("Enter Your Function in LaTeX Format")
        st.write("**Example**: `f(x, y) = x^2 + y^2` or just `x^2 + y^2`.")
        latex_str = st.text_area("LaTeX Expression", value="x^2 + y^2")
    else:
        st.subheader("Enter Your Python Function")
        st.write("**Example**:\n```python\ndef objective_function(x, y):\n    return x**2 + y**2\n```")
        py_str = st.text_area("Python Code", value="def objective_function(x, y):\n    return x**2 + y**2")
    
    if st.button("Optimize"):
        # Parse initial guess
        x0_list = [float(v.strip()) for v in x0_input.split(",")]
        x0 = np.array(x0_list)
        
        # Build the objective function
        if input_mode == "LaTeX Expression":
            expr = parse_latex_function(latex_str)
            # Convert Sympy expression to python function
            # var_names example: "x, y" -> var_names_list = ["x", "y"]
            var_list = [v.strip() for v in var_names.split(",")]
            obj_func_sympy = sympy_function_to_callable(expr, var_list)
            
            # We create a wrapper that takes a single array argument
            def wrapped_func(x_array):
                return obj_func_sympy(*x_array)
            
        else:
            user_func = parse_python_function(py_str)
            
            # Similarly, create a wrapper that can handle a single array argument
            def wrapped_func(x_array):
                return user_func(*x_array)
        
        # Perform optimization
        res = optimize_function(
            obj_func=wrapped_func, 
            x0=x0, 
            method=method
        )
        
        st.write("### Optimization Results")
        st.write(f"**Optimal Point**: {res.x}")
        st.write(f"**Optimal Objective Value**: {res.fun}")
        
        # Attempt to plot based on dimension
        dim = len(res.x)
        
        if dim == 1:
            fig = generate_plot_1d(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 2:
            # Give user the option to switch plot types
            plot_type = st.selectbox("Select Plot Type", ["3D Surface", "2D Contour"])
            if plot_type == "3D Surface":
                fig = generate_plot_2d_surface(wrapped_func, res.x)
            else:
                fig = generate_plot_2d_contour(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        elif dim == 3:
            # A simple 3D slice approach
            fig = generate_plot_3d(wrapped_func, res.x)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Plotting for dimensions higher than 3 is not supported at this time.")

if __name__ == "__main__":
    main()
