
import numpy as np
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd

@dataclass
class PolyRegResult:
    order: int
    params: np.ndarray          # [beta0, beta1, ..., betak]
    model: sm.regression.linear_model.RegressionResultsWrapper
    x_min: float
    x_max: float


def _extract_x_y(full_df: pd.DataFrame,x_col:str,y_col:str) -> Tuple[np.ndarray, np.ndarray]:

    x = full_df[x_col].to_array()
    y = full_df[y_col].to_array()

    return x,y
    


def _poly_features(x: np.ndarray, order: int) -> np.ndarray:
    """
    Build polynomial feature matrix WITHOUT the intercept term.
    Columns: x^1, x^2, ..., x^order
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([x**i for i in range(1, order + 1)])

def _add_const(X: np.ndarray) -> np.ndarray:
    """Add intercept column as the first column."""
    return sm.add_constant(X, has_constant="add")

# ---------- Fit ----------

def fit_poly_ols(x: np.ndarray, y: np.ndarray, order: int = 1) -> PolyRegResult:
    """
    Fit y ~ 1 + x + x^2 + ... + x^order via OLS (statsmodels).

    Returns:
        PolyRegResult with fitted params and the statsmodels result.
    """
    
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if order < 1:
        raise ValueError("order must be >= 1")

    X = _add_const(_poly_features(x, order))
    model = sm.OLS(y, X).fit()
    return PolyRegResult(
        order=order,
        params=model.params.copy(),
        model=model,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
    )




# ---------- Predict (single fit) ----------

def predict_poly(result: PolyRegResult, x_new: np.ndarray) -> np.ndarray:
    """
    Predict fitted curve at x_new using the fitted OLS params.
    """
    x_new = np.asarray(x_new, dtype=float).reshape(-1)
    X_new = _add_const(_poly_features(x_new, result.order))
    return X_new @ result.params

# ---------- Bootstrap CI band (Seaborn-style) ----------

def bootstrap_curve_ci(
    x: np.ndarray,
    y: np.ndarray,
    order: int,
    x_grid: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: Optional[int] = None,
    replace: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Seaborn-like bootstrap confidence band for the fitted curve.

    For each bootstrap resample of the (x, y) pairs:
      1) Fit the polynomial model.
      2) Predict on x_grid.
    The CI is taken pointwise over bootstrap predictions.

    Returns:
        (lower, upper) arrays matching x_grid shape.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=float).reshape(-1)
    n = x.size

    preds = np.empty((n_boot, x_grid.size), dtype=float)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=replace)
        xb, yb = x[idx], y[idx]
        res_b = fit_poly_ols(xb, yb, order=order)
        preds[b, :] = predict_poly(res_b, x_grid)

    alpha = (100.0 - ci) / 100.0
    lo = np.quantile(preds, alpha / 2, axis=0)
    hi = np.quantile(preds, 1 - alpha / 2, axis=0)
    return lo, hi
