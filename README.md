# Pairwise Exchange Layout Optimizer

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:

   pip install -r requirements.txt
   streamlit run app.py

## Streamlit Community Cloud

Upload `app.py` and `requirements.txt` to the root of a GitHub repository. In Streamlit Community Cloud, create an app and select that repository, branch, and `app.py` as the main file.

## Inputs and interpretation

The flow matrix is indexed by machines. Enter directional flows separately: each nonzero cell is counted once. If both directions are entered, both count. A single entry per pair works for the slide example. The distance matrix is indexed by fixed positions, not machines, and can be symmetric or asymmetric. The default is a row of equally spaced positions. The final layout graphic shows order only, not actual physical geometry. Use Run after editing inputs.

At every iteration the algorithm evaluates every two-machine swap and selects the largest strict improvement. Ties are resolved by machine-number order. It stops when none improves the objective, so its solution is locally optimal with respect to one swap, not necessarily globally optimal.
