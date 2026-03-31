import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr


def compute_ence(y_true, y_pred, y_std, n_bins=10):
    """Expected Normalized Calibration Error.

    Bins predictions by uncertainty magnitude and compares
    root mean variance (RMV) to RMSE per bin.

    ENCE = (1/B) * sum_b |RMSE_b - RMV_b| / RMV_b
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)

    sorted_idx = np.argsort(y_std)
    bins = np.array_split(sorted_idx, n_bins)

    ence = 0.0
    n_valid_bins = 0
    for bin_idx in bins:
        if len(bin_idx) < 2:
            continue
        bin_errors = y_true[bin_idx] - y_pred[bin_idx]
        rmse = np.sqrt(np.mean(bin_errors ** 2))
        rmv = np.sqrt(np.mean(y_std[bin_idx] ** 2))
        if rmv > 1e-10:
            ence += np.abs(rmse - rmv) / rmv
            n_valid_bins += 1

    return ence / max(n_valid_bins, 1)


def compute_ence_decomposition(y_true, y_pred, y_std, n_bins=10):
    """Per-bin RMSE and RMV for ENCE decomposition plotting.

    Returns dict with bin-level calibration data showing where the model
    is overconfident (RMSE > RMV) vs underconfident (RMV > RMSE).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)

    sorted_idx = np.argsort(y_std)
    bins = np.array_split(sorted_idx, n_bins)

    bin_labels = []
    rmse_vals = []
    rmv_vals = []
    bin_counts = []

    for i, bin_idx in enumerate(bins):
        if len(bin_idx) < 2:
            continue
        bin_errors = y_true[bin_idx] - y_pred[bin_idx]
        rmse = np.sqrt(np.mean(bin_errors ** 2))
        rmv = np.sqrt(np.mean(y_std[bin_idx] ** 2))
        bin_labels.append(f'B{i + 1}')
        rmse_vals.append(rmse)
        rmv_vals.append(rmv)
        bin_counts.append(len(bin_idx))

    rmse_arr = np.array(rmse_vals)
    rmv_arr = np.array(rmv_vals)

    return {
        'bin_labels': bin_labels,
        'rmse': rmse_arr,
        'rmv': rmv_arr,
        'bin_counts': np.array(bin_counts),
        'overconfident': rmse_arr > rmv_arr,
    }


def compute_nll(y_true, y_pred, y_std):
    """Gaussian Negative Log-Likelihood.

    NLL = 0.5 * mean[log(2*pi*sigma^2) + (y - mu)^2 / sigma^2]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)
    var = y_std ** 2

    nll = 0.5 * np.mean(
        np.log(2 * np.pi * var) + (y_true - y_pred) ** 2 / var
    )
    return nll


def compute_miscalibration_area(y_true, y_pred, y_std, n_points=20):
    """Miscalibration area: integral of |empirical_coverage - expected_coverage|.

    For a range of confidence levels p, compute the fraction of true values
    falling within the predicted p-interval. The miscalibration area is the
    area between the calibration curve and the perfect diagonal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)

    confidence_levels = np.linspace(0.05, 0.95, n_points)
    deviations = []

    for p in confidence_levels:
        z = norm.ppf((1 + p) / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        empirical = np.mean((y_true >= lower) & (y_true <= upper))
        deviations.append(np.abs(empirical - p))

    return np.trapz(deviations, confidence_levels)


def compute_calibration_curve(y_true, y_pred, y_std, n_points=20):
    """Compute calibration curve data for plotting.

    Returns expected coverage levels and corresponding empirical coverages.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)

    confidence_levels = np.linspace(0.05, 0.95, n_points)
    empirical_coverages = []

    for p in confidence_levels:
        z = norm.ppf((1 + p) / 2)
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        empirical = np.mean((y_true >= lower) & (y_true <= upper))
        empirical_coverages.append(empirical)

    return confidence_levels, np.array(empirical_coverages)


def compute_ause(y_true, y_pred, y_std, n_points=50):
    """Area Under Sparsification Error.

    Measures how well uncertainty rankings match error rankings.
    AUSE = area between model sparsification curve and oracle curve.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    n = len(y_true)
    if n < 10:
        return np.nan

    errors = np.abs(y_true - y_pred)
    fractions = np.linspace(0, 0.95, n_points)

    unc_order = np.argsort(y_std)
    err_order = np.argsort(errors)

    model_maes = []
    oracle_maes = []

    for frac in fractions:
        k = int(frac * n)
        if k >= n:
            k = n - 1

        keep_model = unc_order[:n - k] if k > 0 else unc_order
        keep_oracle = err_order[:n - k] if k > 0 else err_order

        model_maes.append(np.mean(errors[keep_model]))
        oracle_maes.append(np.mean(errors[keep_oracle]))

    model_arr = np.array(model_maes)
    oracle_arr = np.array(oracle_maes)

    ause = np.trapz(model_arr - oracle_arr, fractions)
    return ause


def compute_sparsification_data(y_true, y_pred, y_std, n_points=50):
    """Compute sparsification curve data for plotting."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    n = len(y_true)
    errors = np.abs(y_true - y_pred)
    fractions = np.linspace(0, 0.95, n_points)

    unc_order = np.argsort(y_std)
    err_order = np.argsort(errors)

    model_maes = []
    oracle_maes = []
    random_mae = np.mean(errors)

    for frac in fractions:
        k = int(frac * n)
        if k >= n:
            k = n - 1

        keep_model = unc_order[:n - k] if k > 0 else unc_order
        keep_oracle = err_order[:n - k] if k > 0 else err_order

        model_maes.append(np.mean(errors[keep_model]))
        oracle_maes.append(np.mean(errors[keep_oracle]))

    return {
        'fractions': fractions,
        'model_maes': np.array(model_maes),
        'oracle_maes': np.array(oracle_maes),
        'random_mae': random_mae
    }


def compute_spearman(y_true, y_pred, y_std):
    """Spearman rank correlation between uncertainty and absolute error."""
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    corr, p_value = spearmanr(np.asarray(y_std), errors)
    return corr, p_value


def compute_all_uq_metrics(y_true, y_pred, y_std, n_bins=10):
    """Compute all UQ metrics at once.

    Returns dict with metric names as keys and values as floats.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_std)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_std = y_std[mask]

    if len(y_true) < 10:
        return {
            'ence': np.nan, 'nll': np.nan, 'miscalibration_area': np.nan,
            'ause': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan,
            'mae': np.nan, 'rmse': np.nan, 'n_samples': len(y_true)
        }

    corr, p_val = compute_spearman(y_true, y_pred, y_std)

    return {
        'ence': compute_ence(y_true, y_pred, y_std, n_bins=n_bins),
        'nll': compute_nll(y_true, y_pred, y_std),
        'miscalibration_area': compute_miscalibration_area(y_true, y_pred, y_std),
        'ause': compute_ause(y_true, y_pred, y_std),
        'spearman_rho': corr,
        'spearman_p': p_val,
        'mae': np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'n_samples': len(y_true)
    }


def calibrate_std_scaling(y_true, y_pred, y_std):
    """Find optimal STD scaling factor by minimizing NLL on given data.

    Returns the optimal scale factor k such that k * y_std minimizes NLL.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    y_std = np.clip(y_std, 1e-10, None)

    def nll_objective(log_k):
        k = np.exp(log_k)
        scaled_std = k * y_std
        scaled_var = scaled_std ** 2
        return 0.5 * np.mean(
            np.log(2 * np.pi * scaled_var) + (y_true - y_pred) ** 2 / scaled_var
        )

    result = minimize_scalar(nll_objective, bounds=(-3, 3), method='bounded')
    optimal_k = np.exp(result.x)

    return optimal_k


def apply_std_scaling(y_std, scale_factor):
    """Apply STD scaling calibration."""
    return np.asarray(y_std, dtype=float) * scale_factor


def compute_error_by_uncertainty_quantile(y_true, y_pred, y_std, n_quantiles=5):
    """Bin compounds by uncertainty quantile and compute error stats per bin.

    Returns dict with quantile labels, median errors, and full error arrays per bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    errors = np.abs(y_true - y_pred)
    sorted_idx = np.argsort(y_std)
    bins = np.array_split(sorted_idx, n_quantiles)

    labels = [f'Q{i+1}' for i in range(n_quantiles)]
    bin_errors = [errors[b] for b in bins]
    bin_medians = [np.median(e) for e in bin_errors]
    bin_means = [np.mean(e) for e in bin_errors]

    return {
        'labels': labels,
        'bin_errors': bin_errors,
        'medians': bin_medians,
        'means': bin_means
    }
