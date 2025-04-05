# Multivariate Function Optimizer

A simple Streamlit application that finds the **global minimum** (approximate) of user-provided functions using various optimization approaches (Gradient Descent, PSO-like, etc.). It also provides convenient **visualizations** for 1D, 2D, and 3D functions.

## Features

1. **Multiple Input Modes**  
   - **LaTeX Expression**: Paste or type a LaTeX-style function, e.g. `x^2 + y^2`.  
   - **Python Function**: Paste a Python code block that defines `objective_function`.  

2. **Optimizer Methods**  
   - **Gradient Descent** (BFGS under the hood).  
   - **PSO** (simulated here using `differential_evolution` + local refinement).  
   - **Least Squares** (simplified example using the same method as gradient descent).  

3. **Visualizations**  
   - **1D Functions**: Plot a 2D line (x vs. f(x)) and mark the optimal point.  
   - **2D Functions**: Select between a 3D surface plot or a 2D contour plot.  
   - **3D Functions**: Display a 3D surface by fixing the 3rd variable at the optimum (slice-based).  

4. **Interactive Interface**  
   - Quickly enter an initial guess, variable names (if using LaTeX), and optimize.  
   - Switch between different plots for 2D scenarios (contour vs. surface).

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/yourusername/multivariate-function-optimizer.git
   cd multivariate-function-optimizer
   ```
2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Make sure your `requirements.txt` includes:
   ```text
   streamlit
   sympy
   numpy
   scipy
   plotly
   ```
   
4. **Run the app**:
   ```bash
   streamlit run main.py
   ```

## Usage Examples

### Example 1: LaTeX Input
1. Select **LaTeX Expression** in the sidebar.  
2. Enter `x^2 + y^2` in the text area.  
3. Set your initial guess, e.g. `0, 0`.  
4. Set your variable names to `x, y`.  
5. Choose an optimization method (e.g. **Gradient Descent**).  
6. Click **Optimize**.  

You will see:
- The resulting **optimal point** near `[0, 0]`.
- A **2D contour** or **3D surface** plot (your choice) with the minimum highlighted.

### Example 2: Python Function
1. Select **Python Function** in the sidebar.  
2. Enter:
   ```python
   def objective_function(x, y):
       return (x - 1)**2 + (y + 2)**2 + 3
   ```
3. Provide initial guess, e.g. `0, 0`.  
4. Click **Optimize**.  

You will see:
- The **optimal point** near `[1, -2]`.  
- The corresponding **minimum value** near `3`.  
- A plot of the function in 2D or 3D as appropriate.

## Limitations
- Parsing advanced LaTeX with complex symbols might fail. Simple expressions work best.  
- For higher than 3 variables, only numeric results are shown (no advanced plots).  
- The PSO method is emulated with `differential_evolution` + a local BFGS refine, but no classic particle swarm approach is provided by default.
