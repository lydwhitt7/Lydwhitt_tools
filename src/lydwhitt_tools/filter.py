import pandas as pd
import lydwhitt_tools as lwt
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def make_phase_rules(df, phase):
    """
    Build phase-specific rules and a recalculated DataFrame.

    Returns a dict with:
      - features: columns used for density features
      - mode_columns: columns to evaluate with KDE
      - trend_axes: axes used for y~t trend diagnostics
      - t_name: progress variable column name
      - df: phase-aware recalculated DataFrame (APFU/mol as needed)
    """
    if not isinstance(phase, str):
        raise TypeError("phase must be a string")

    phase = phase.strip()
    allowed = ("Liq", "Plg", "Cpx")
    if phase not in allowed:
        raise ValueError(f"phase must be one of {allowed}")

    # phase-aware recalculation (anhydrous; mol values)
    df = lwt.recalc(df, phase, anhydrous=True, mol_values=True)

    if phase == "Liq":
        features = ["SiO2_Liq","MgO_Liq","FeOt_Liq","CaO_Liq","Al2O3_Liq","Na2O_Liq","K2O_Liq","TiO2_Liq"]
        mode_columns = ["SiO2_Liq","MgO_Liq","FeOt_Liq","CaO_Liq","Al2O3_Liq","Na2O_Liq","K2O_Liq","TiO2_Liq"]
        trend_axes = ["SiO2_Liq","FeOt_Liq","CaO_Liq","Al2O3_Liq","K2O_Liq"]
        t_name = "MgO_Liq"

    elif phase == "Plg":
        features = ["CaO_Plg","Na2O_Plg","K2O_Plg","Al2O3_Plg","SiO2_Plg","FeOt_Plg","MgO_Plg"]
        mode_columns = ["CaO_Plg","Na2O_Plg","K2O_Plg","Al2O3_Plg","SiO2_Plg"]
        trend_axes = ["CaO_Plg","Na2O_Plg","K2O_Plg"]  # vs An
        t_name = "An"

    else:  # Cpx
        features = ["SiO2_Cpx","TiO2_Cpx","Al2O3_Cpx","FeOt_Cpx","MnO_Cpx","MgO_Cpx","CaO_Cpx","Na2O_Cpx","Cr2O3_Cpx"]
        mode_columns = ["MgO_Cpx","FeOt_Cpx","CaO_Cpx","TiO2_Cpx","Al2O3_Cpx","SiO2_Cpx"]
        trend_axes = ["MgO_Cpx","FeOt_Cpx","CaO_Cpx","TiO2_Cpx"]  # vs Mg#
        t_name = "Mg_num"

    return {
        "features": features,
        "mode_columns": mode_columns,
        "trend_axes": trend_axes,
        "t_name": t_name,
        "df": df,
    }


# -------- Smoothing --------
# --- Helper: quantile-median fallback smoother ---
def _default_quantile_median_smoother(y, x, frac=0.2):
    """Fallback smoother: bin x into ~1/frac quantile bins and use median(y) per bin."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n == 0:
        return y
    q = max(5, int(round(1.0 / max(1e-6, min(0.9, frac)))))
    bins = pd.qcut(x, q=q, duplicates="drop")
    y_fit = pd.Series(y).groupby(bins).transform("median").to_numpy()
    if np.isnan(y_fit).any():
        med = np.nanmedian(y)
        y_fit = np.where(np.isnan(y_fit), med, y_fit)
    return y_fit


def LOWESS(y, x, frac=0.5, it=1, **_):
    """LOWESS smoother. Uses statsmodels if available; otherwise a quantile-median fallback.
    Output order matches input order.
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        return _default_quantile_median_smoother(y, x, frac=frac)

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return y
    order = np.argsort(x)
    y_ord = y[order]
    x_ord = x[order]
    yfit_ord = lowess(y_ord, x_ord, frac=float(frac), it=int(it), return_sorted=False)
    yfit = np.empty_like(yfit_ord)
    yfit[order] = yfit_ord
    return yfit


def POLY(y, x, degree=2, **_):  # degree=2 by default (quadratic)
    """
    Polynomial fit helper on scaled x (≈[-1, 1]) to mitigate conditioning; returns fitted y.
    """
    # default degree=2 (quadratic)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0:
        return y
    # scale x to ~[-1, 1] to avoid ill-conditioning
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        z = x * 0.0
    else:
        z = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    deg = int(degree)
    coeffs = np.polyfit(z, y, deg=deg)
    yfit = np.polyval(coeffs, z)
    return yfit


def fit_trend_curves(df, t_name, trend_axes, smooth_fn=None, smooth_kwargs=None, min_points=20):
    """
    Fit y~t curves for each axis using the provided smoothing function.

    Parameters
    ----------
     df : DataFrame
     t_name : str
     trend_axes : list[str]
     smooth_fn : callable
     smooth_kwargs : dict | None
     min_points : int

    Returns
    -------
     dict[str, Series]
         Mapping of axis name -> fitted values (Series aligned to df.index).
    """
    if not isinstance(t_name, str):
        raise TypeError("t_name must be a string")
    if not isinstance(trend_axes, (list, tuple)):
        raise TypeError("trend_axes must be a list/tuple")
    if smooth_fn is None or not callable(smooth_fn):
        raise TypeError("smooth_fn must be callable (e.g., LOWESS)")
    if smooth_kwargs is None:
        smooth_kwargs = {}

    missing = [c for c in [t_name] + list(trend_axes) if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    x_all = df[t_name].to_numpy(dtype=float)
    fitted = {}

    for axis in trend_axes:
        y_all = df[axis].to_numpy(dtype=float)
        mask = np.isfinite(x_all) & np.isfinite(y_all)

        if mask.sum() < int(min_points):
            fitted_vals = np.full_like(y_all, np.nan, dtype=float)
        else:
            y_fit_masked = smooth_fn(y_all[mask], x_all[mask], **smooth_kwargs)
            fitted_vals = np.full_like(y_all, np.nan, dtype=float)
            fitted_vals[mask] = np.asarray(y_fit_masked, dtype=float)

        fitted[axis] = pd.Series(fitted_vals, index=df.index, name=f"{axis}_fit")

    return fitted


def compute_trend_residuals_and_z(
    df,
    trend_axes,
    t_name,
    p_neigh: float = 0.04,
    k_min: int = 10,
    k_max: int = 30,
    eps: float = 1e-9,
):
    """Local trend residuals and robust z per axis using k-nearest neighbours in t.
    Steps (per axis): sort by t → sliding window of size k → local median & MAD →
    residual r and z = r / (1.4826*MAD+eps). Also returns robust R² from MAD reduction.
    """
    if not isinstance(t_name, str):
        raise TypeError("t_name must be a string")

    need = [t_name] + list(trend_axes)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"missing required columns: {miss}")

    t_all = df[t_name].to_numpy(dtype=float)
    n_total = len(df)

    residuals, z_by_axis, r2_robust_by_axis = {}, {}, {}

    def _mad(a):
        a = np.asarray(a, dtype=float)
        m = np.nanmedian(a)
        return np.nanmedian(np.abs(a - m))

    for axis in trend_axes:
        y_all = df[axis].to_numpy(dtype=float)
        mask = np.isfinite(t_all) & np.isfinite(y_all)
        n = int(mask.sum())

        r_full = np.full(n_total, np.nan)
        z_full = np.full(n_total, np.nan)

        if n < max(3, k_min):
            residuals[axis] = pd.Series(r_full, index=df.index, name=f"{axis}_resid")
            z_by_axis[axis] = pd.Series(z_full, index=df.index, name=f"{axis}_z")
            r2_robust_by_axis[axis] = np.nan
            continue

        t = t_all[mask]; y = y_all[mask]
        order = np.argsort(t)
        t_s = t[order]; y_s = y[order]

        k = int(np.clip(int(round(p_neigh * n)), k_min, k_max))
        k = max(3, min(k, n))
        half = k // 2

        med = np.empty(n, dtype=float)
        mad = np.empty(n, dtype=float)

        for i in range(n):
            L = max(0, i - half)
            R = min(n, L + k)
            L = max(0, R - k)
            win = y_s[L:R]
            m = np.nanmedian(win)
            s = 1.4826 * np.nanmedian(np.abs(win - m))
            if not (np.isfinite(s) and s > 0):
                s = np.nanstd(win, ddof=0)
            if not (np.isfinite(s) and s > 0):
                s = 1.0
            med[i] = m
            mad[i] = s

        r_s = y_s - med

        mad_y = _mad(y_s)
        mad_r = _mad(r_s)
        r2r = 1.0 - (mad_r / mad_y) ** 2 if (np.isfinite(mad_y) and mad_y > 0) else np.nan
        r2_robust_by_axis[axis] = float(r2r)

        z_s = r_s / (mad + eps)

        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        r_masked = r_s[inv]
        z_masked = z_s[inv]

        r_full[mask] = r_masked
        z_full[mask] = z_masked

        residuals[axis] = pd.Series(r_full, index=df.index, name=f"{axis}_resid")
        z_by_axis[axis] = pd.Series(z_full, index=df.index, name=f"{axis}_z")

    trend_z = pd.DataFrame(z_by_axis).mean(axis=1, skipna=True).rename("trend_z")
    axis_z_df = pd.DataFrame(z_by_axis)

    TAU_R2 = 0.12
    axes_pass = [ax for ax, r2 in r2_robust_by_axis.items() if np.isfinite(r2) and r2 >= TAU_R2]
    if not axes_pass:
        axes_pass = list(axis_z_df.columns)

    if axes_pass:
        trend_axis_medianabs = axis_z_df[axes_pass].abs().median(axis=1).rename("trend_axis_medianabs")
    else:
        trend_axis_medianabs = axis_z_df.abs().median(axis=1).rename("trend_axis_medianabs")

    Z_CLIP = 8.0
    tam_clip = np.minimum(trend_axis_medianabs.astype(float), Z_CLIP)
    trend_score = np.exp(-0.5 * (tam_clip ** 2)).rename("trend_score")

    return {
        "residuals": residuals,
        "z_by_axis": z_by_axis,
        "trend_z": trend_z,
        "trend_axis_medianabs": trend_axis_medianabs,
        "trend_score": trend_score,
        "local_params": {"p_neigh": float(p_neigh), "k_min": int(k_min), "k_max": int(k_max)},
        "trend_axes_kept": axes_pass,
        "trend_r2_robust": r2_robust_by_axis,
    }


# -------- KDE modes --------

def KDE_mode_hits(df, mode_columns, return_matrix=False):
    """1D KDE per column → HDR intervals (keep ~90% mass) →
    (a) binary membership and (b) continuous proximity scores via peak distance.
    """
    if not isinstance(mode_columns, (list, tuple)):
        raise TypeError("mode_columns must be a list or tuple")

    present = set(df.columns)
    missing = [c for c in mode_columns if c not in present]
    if missing:
        raise KeyError(f"expected columns is missing {missing}")

    N = len(df)
    used, skipped = [], []
    y_thr, intervals, member = {}, {}, {}
    scores_cols = {}
    kde_xy = {}

    P_KEEP = 0.90
    EPS = 1e-9

    def _area_weights(x, y):
        dx = np.empty_like(x)
        dx[1:-1] = 0.5 * (x[2:] - x[:-2])
        dx[0] = x[1] - x[0]
        dx[-1] = x[-1] - x[-2]
        return y * np.clip(dx, 0.0, None)

    def _hdr_threshold(y, w, p_keep=P_KEEP):
        tot = float(np.sum(w))
        if not np.isfinite(tot) or tot <= 0:
            return np.nan
        order = np.argsort(-y)
        y_sorted = y[order]
        w_sorted = w[order]
        cum = np.cumsum(w_sorted)
        k = np.searchsorted(cum, p_keep * tot, side="left")
        k = min(max(k, 0), len(y_sorted) - 1)
        return float(y_sorted[k])

    def _runs_to_intervals(idx, x):
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)
        return [(float(x[r[0]]), float(x[r[-1]])) for r in runs]

    def _estimate_peaks(x_sorted, y_sorted):
        # local maxima indices
        if len(y_sorted) >= 3:
            is_peak = (y_sorted[1:-1] > y_sorted[:-2]) & (y_sorted[1:-1] > y_sorted[2:])
            peaks = np.where(is_peak)[0] + 1
        else:
            peaks = np.array([], dtype=int)
        if peaks.size == 0 and y_sorted.size:
            peaks = np.array([int(np.nanargmax(y_sorted))])
        return peaks

    for name in mode_columns:
        vals = df[name].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        n_fin = int(finite.sum())
        if n_fin < 50:
            skipped.append(name)
            continue

        xy = lwt.KDE(df.loc[finite, [name]], name)
        if not {"x", "y"}.issubset(xy.columns):
            raise ValueError("KDE must return 'x' and 'y'")

        x = xy["x"].to_numpy(dtype=float)
        y = xy["y"].to_numpy(dtype=float)

        # store grid for plotting
        xy_local = xy[["x", "y"]].copy(); xy_local["column"] = name
        kde_xy[name] = xy_local

        w = _area_weights(x, y)
        thr = _hdr_threshold(y, w)
        y_thr[name] = thr

        keep = (y >= thr)
        idx = np.flatnonzero(keep)
        ints = _runs_to_intervals(idx, x)
        intervals[name] = ints

        in_col = np.zeros(N, dtype=bool)
        for (L, R) in ints:
            in_col |= (vals >= L) & (vals <= R)
        member[name] = in_col

        # continuous proximity score via peaks
        order_xy = np.argsort(x)
        x_s = x[order_xy]
        y_s = y[order_xy]
        pk_idx = _estimate_peaks(x_s, y_s)

        x_span = float(np.nanmax(x_s) - np.nanmin(x_s)) if x_s.size else 0.0
        SIGMA_MAX_FRAC = 0.10
        dx_grid = np.clip(np.gradient(x_s), 0.0, None)
        peaks = []  # (mass, center, sigma, height)
        for pi in pk_idx:
            h = float(y_s[pi])
            if not (np.isfinite(h) and h > 0):
                continue
            half = 0.5 * h
            li = pi
            while li > 0 and y_s[li] >= half:
                li -= 1
            ri = pi
            nxy = len(y_s)
            while ri < nxy - 1 and y_s[ri] >= half:
                ri += 1
            left_x = x_s[max(li, 0)]
            right_x = x_s[min(ri, nxy - 1)]
            fwhm = max(right_x - left_x, 1e-9)
            sigma = max(fwhm / 2.355, 1e-6)
            SIGMA_MIN_FRAC = 0.005
            SIGMA_MIN_ABS = 0.05
            sigma = max(sigma, max(SIGMA_MIN_ABS, SIGMA_MIN_FRAC * x_span))
            sigma = min(sigma, max(SIGMA_MAX_FRAC * x_span, 1e-3))
            mass = float(np.nansum(y_s[max(li, 0):min(ri + 1, nxy)] * dx_grid[max(li, 0):min(ri + 1, nxy)]))
            peaks.append((mass, float(x_s[pi]), float(sigma), h))

        TOP_K = 2
        MIN_HEIGHT_FRAC = 0.25
        MIN_MASS_FRAC = 0.05
        if peaks:
            y_max = max(p[3] for p in peaks)
            peaks = [p for p in peaks if p[3] >= MIN_HEIGHT_FRAC * y_max]
            tot = float(np.sum(_area_weights(x_s, y_s)))
            peaks = [p for p in peaks if (tot > 0 and (p[0] / tot) >= MIN_MASS_FRAC)]
            peaks.sort(key=lambda t: t[0], reverse=True)
            peaks = peaks[:TOP_K]

        centers, sigmas = [], []
        if peaks:
            centers = [p[1] for p in peaks]
            sigmas = [p[2] for p in peaks]
        elif ints:
            for (L, R) in ints:
                c = 0.5 * (L + R)
                fwhm = max(R - L, 1e-9)
                s = max(fwhm / 2.355, 1e-6)
                SIGMA_MIN_FRAC = 0.005
                SIGMA_MIN_ABS = 0.05
                s = max(s, max(SIGMA_MIN_ABS, SIGMA_MIN_FRAC * x_span))
                s = min(s, max(SIGMA_MAX_FRAC * x_span, 1e-3))
                centers.append(float(c)); sigmas.append(float(s))

        if centers:
            C = np.asarray(centers, dtype=float)
            S = np.asarray(sigmas, dtype=float)
            d = np.abs(vals[:, None] - C[None, :]) / (S[None, :] + EPS)
            d_min = d.min(axis=1)
            D_CLIP = 8.0
            d_min = np.minimum(d_min, D_CLIP)
            s = np.exp(-0.5 * d_min ** 2)
            s = np.where(np.isfinite(s), s, 0.0)
        else:
            s = np.zeros(N, dtype=float)
        scores_cols[name] = s

        used.append(name)

    if used:
        membership_df = pd.DataFrame({c: member[c] for c in used}, index=df.index)
        mode_hits = membership_df.sum(axis=1).astype(int).rename("mode_hits")
        scores_df = pd.DataFrame({c: scores_cols[c] for c in used}, index=df.index)
        cluster_score = scores_df.median(axis=1, skipna=True).rename("cluster_score")
    else:
        membership_df = pd.DataFrame(index=df.index)
        mode_hits = pd.Series(np.nan, index=df.index, name="mode_hits")
        scores_df = pd.DataFrame(index=df.index)
        cluster_score = pd.Series(np.nan, index=df.index, name="cluster_score")

    return {
        "mode_hits": mode_hits,
        "membership": membership_df if return_matrix else None,
        "y_thresholds": y_thr,
        "intervals": intervals,
        "used_columns": used,
        "skipped_columns": skipped,
        "scores": scores_df,
        "cluster_score": cluster_score,
        "kde_xy": kde_xy,
    }


# -------- Main pipeline --------

def geoscore_filter(
    df,
    phase,
    smooth_fn=None,
    smooth_kwargs=None,
    return_membership=False
):
    """
    End-to-end geochemical filter pipeline for a given phase.
    Steps: build rules, fit trends, compute local z, run KDE/HDR, combine into a
    continuous geo_score and final_pass.

    Returns a dict with the filtered DataFrame, diagnostics, and intermediates.
    """
    rules = make_phase_rules(df, phase)
    df_phase = rules["df"].copy()
    t_name = rules["t_name"]
    trend_axes = rules["trend_axes"]
    mode_columns = rules["mode_columns"]

    # fit trends (LOWESS default)
    if smooth_fn is None:
        smooth_fn = LOWESS
    if smooth_kwargs is None:
        smooth_kwargs = {"frac": 0.35, "it": 1}
    fits = fit_trend_curves(df_phase, t_name, trend_axes,
                            smooth_fn=smooth_fn, smooth_kwargs=smooth_kwargs)
    # ---- Write each fitted series into df_phase for plotting access ----
    for ax_name, s in fits.items():
        df_phase[f"{ax_name}_fit"] = s

    # compute local residuals and robust z in t-neighborhoods
    trend_out = compute_trend_residuals_and_z(df_phase, trend_axes, t_name)
    df_phase["trend_z"] = trend_out["trend_z"]
    df_phase["trend_axis_medianabs"] = trend_out["trend_axis_medianabs"]
    df_phase["trend_score"] = trend_out["trend_score"]

    # KDE/HDR -> mode_hits (+ optional membership)
    kde_out = KDE_mode_hits(df_phase, mode_columns, return_matrix=return_membership)
    df_phase["mode_hits"] = kde_out["mode_hits"]
    df_phase["cluster_score"] = kde_out.get("cluster_score", pd.Series(np.nan, index=df_phase.index))

    # combine trend and cluster scores
    # Choose how to combine: "max" (generous) or "avg" (balanced)
    COMBINE_MODE = "max"   # or "avg".
    GEO_MIN = 0.4          # decision threshold.

    ts = df_phase["trend_score"].astype(float)
    cs = df_phase["cluster_score"].astype(float)

    if COMBINE_MODE == "max":
        geo_score = np.maximum(ts, cs)
    else:
        geo_score = 0.5 * (ts + cs)

    df_phase["geo_score"] = geo_score
    df_phase["geo_pass"] = (df_phase["geo_score"] >= GEO_MIN)

    # final decision based on geo_score
    df_phase["final_pass"] = df_phase["geo_pass"]

    # remove legacy binary gating columns if they exist (keep things clean)
    legacy_cols = ["trend_pass", "modes_pass", "lof_pass", "pass_2of3"]
    to_drop = [c for c in legacy_cols if c in df_phase.columns]
    if to_drop:
        df_phase.drop(columns=to_drop, inplace=True)

    # Stash trend selection into attrs so plotting can show only the used curves
    try:
        df_phase.attrs["trend_axes_kept"] = trend_out.get("trend_axes_kept", [])
        df_phase.attrs["trend_r2_robust"] = trend_out.get("trend_r2_robust", {})
    except Exception:
        pass

    return {
        "df": df_phase,
        "diagnostics": {
            "phase": phase,
            "t_name": t_name,
            "trend_axes": trend_axes,
            "mode_columns_used": kde_out["used_columns"],
            "mode_columns_skipped": kde_out["skipped_columns"],
            "kde_y_thresholds": kde_out["y_thresholds"],
            "kde_intervals": kde_out["intervals"],
            "trend_local_params": trend_out.get("local_params", {}),
            "geo_combine_mode": COMBINE_MODE,
            "geo_min": GEO_MIN,
            "trend_smoother": getattr(smooth_fn, "__name__", str(smooth_fn)),
            "trend_smoother_kwargs": (smooth_kwargs or {}),
            "kde_xy": kde_out.get("kde_xy", {}),
            "trend_axes_kept": trend_out.get("trend_axes_kept", []),
            "trend_r2_robust": trend_out.get("trend_r2_robust", {}),
        },
        "kde_scores": kde_out.get("scores"),
        "membership": (kde_out["membership"] if return_membership else None),
        "trend": trend_out,
        "fits": fits,
    }

def filter_fig(
    df,
    diagnostics=None,
    geo_min=None,
    combine_mode=None,
    t_col=None,
    figsize=(18, 15),
    label_fs=11,
    tick_fs=9,
):
    """
    Generate a multi-panel figure summarizing filter diagnostics for liquids.
    Layout: summary row (distribution, decision plot, |z| vs t, retention), Harker
    rows (with fitted trends), and KDE rows (HDR bands and thresholds).
    """

    # ------------------------- 1) PREPARE INPUTS ------------------------- #
    diagnostics = diagnostics or {}

    # Geo threshold and combine behaviour
    geo_min = diagnostics.get("geo_min", 0.4) if geo_min is None else geo_min
    combine_mode = diagnostics.get("geo_combine_mode", "max") if combine_mode is None else combine_mode

    # Determine progress variable column (t)
    if t_col is None:
        t_col = diagnostics.get("t_name", None)
        if t_col is None:
            for guess in ("MgO_Liq", "An", "Mg_num"):
                if guess in df.columns:
                    t_col = guess
                    break

    # Convenience accessor to pull arrays (optionally for an index subset)
    def _col(name, idx=None, default_nan=True):
        if name not in df.columns:
            if default_nan:
                return np.full(len(df) if idx is None else len(idx), np.nan)
            raise KeyError(f"Column '{name}' not found in DataFrame")
        arr = df[name].to_numpy()
        return arr if idx is None else arr[idx]

    # Keep flags and colours
    geo = df.get("geo_score", 0).astype(float).fillna(0).clip(0, 1).to_numpy()
    passed = df.get("geo_pass", True).astype(bool).fillna(False).to_numpy()
    keep_idx = np.where(passed)[0]
    remove_idx = np.where(~passed)[0]

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("viridis")
    face_all = cmap(norm(geo))

    # trend axes to plot
    trend_cols_kept = diagnostics.get("trend_axes_kept", [])
    if not trend_cols_kept:
        trend_cols_kept = ["SiO2_Liq", "FeOt_Liq", "CaO_Liq", "Al2O3_Liq", "K2O_Liq"]
    trend_cols_kept = [c for c in trend_cols_kept if c in df.columns][:8]

    # KDE inputs
    kde_xy = diagnostics.get("kde_xy", {})
    intervals = diagnostics.get("kde_intervals", {})
    ythr_map = diagnostics.get("kde_y_thresholds", {})
    used_cols = diagnostics.get("mode_columns_used", list(kde_xy.keys()))
    kde_cols = [c for c in used_cols if c in kde_xy][:8]

    # grid
    width_ratios = [1, 0.3, 1, 0.3, 1, 0.3, 1]
    height_ratios = [0.3, 0.08, 0.3, 0.08, 0.3, 0.08, 0.3, 0.08, 0.3]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows=len(height_ratios), ncols=len(width_ratios),
        width_ratios=width_ratios, height_ratios=height_ratios,
        hspace=0, wspace=0,
    )

    def _ax_at(row_block, col_block):
        """Return an axis anchored to the large logical cell (r,c)."""
        return fig.add_subplot(gs[2 * row_block, 2 * col_block])

    # summary row
    def _draw_summary_row():
        # Panel 1: Histogram of geo_score
        ax_hist = _ax_at(0, 0)
        ax_hist.hist(df['geo_score'].dropna(), bins=20, color='k')
        ax_hist.axvline(geo_min, color='r', ls='--', lw=1)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel('geo_score', fontsize=label_fs)
        ax_hist.tick_params(axis='y', left=False, labelleft=False)
        ax_hist.tick_params(labelsize=tick_fs)

        # decision plot (trend vs cluster)
        ax_dec = _ax_at(0, 1)
        ts = _col('trend_score')
        cs = _col('cluster_score')
        ax_dec.scatter(ts[keep_idx], cs[keep_idx], c=face_all[keep_idx], edgecolors='none', alpha=0.9, s=16)
        ax_dec.scatter(ts[remove_idx], cs[remove_idx], c=face_all[remove_idx], edgecolors='red', alpha=0.9, s=18, linewidths=0.8)
        ax_dec.set(xlabel='trend score', ylabel='cluster score', xlim=(0, 1), ylim=(0, 1))
        if combine_mode == 'avg':
            tgrid = np.linspace(0, 1, 200)
            ax_dec.plot(tgrid, 2 * geo_min - tgrid, 'r--', lw=1)
        else:
            ax_dec.axvline(geo_min, color='r', ls='--', lw=1)
            ax_dec.axhline(geo_min, color='r', ls='--', lw=1)
        ax_dec.grid(alpha=0.15, lw=0.5)
        ax_dec.tick_params(labelsize=tick_fs)

        # |z| (median across axes) vs t
        ax_tz = _ax_at(0, 2)
        ax_tz.scatter(_col(t_col, keep_idx), _col('trend_axis_medianabs', keep_idx), c=face_all[keep_idx], edgecolors='none', alpha=0.9, s=16)
        ax_tz.scatter(_col(t_col, remove_idx), _col('trend_axis_medianabs', remove_idx), c=face_all[remove_idx], edgecolors='red', alpha=0.9, s=18, linewidths=0.8)
        ax_tz.set(xlabel=t_col, ylabel='|z| (median across axes)')
        ax_tz.grid(alpha=0.15, lw=0.5)
        ax_tz.tick_params(labelsize=tick_fs)

        # retention curve vs threshold
        ax_keep = _ax_at(0, 3)
        scores_sorted = df['geo_score'].astype(float).dropna().sort_values().to_numpy()
        if scores_sorted.size:
            n = scores_sorted.size
            frac_kept = 1.0 - np.arange(n) / n
            ax_keep.plot(scores_sorted, frac_kept, lw=1.2)
            ax_keep.axvline(geo_min, color='r', ls='--', lw=1)
            ax_keep.set(xlabel='geo_score threshold', ylabel='fraction kept', xlim=(0, 1), ylim=(0, 1))
        ax_keep.grid(alpha=0.15, lw=0.5)
        ax_keep.tick_params(labelsize=tick_fs)

    # harker rows with trend overlays
    def _overlay_fit_vs_t(ax, x_name, y_name):
        fit_col = f"{y_name}_fit"
        if (x_name in df.columns) and (fit_col in df.columns):
            ok = df[[x_name, fit_col]].dropna().sort_values(x_name)
            if len(ok) > 1:
                ax.plot(ok[x_name], ok[fit_col], color='k', lw=1.1, alpha=0.85)

    def _draw_harker_rows():
        # Logical slots: 8 positions -> (row_block, col_block)
        slots = [(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]
        for oxide, (rb, cb) in zip(trend_cols_kept, slots):
            ax = _ax_at(rb, cb)
            ax.scatter(_col(t_col, keep_idx), _col(oxide, keep_idx), c=face_all[keep_idx], edgecolors='none', alpha=0.9, s=18)
            ax.scatter(_col(t_col, remove_idx), _col(oxide, remove_idx), c=face_all[remove_idx], edgecolors='red', alpha=0.9, s=20, linewidths=0.8)
            ax.set(xlabel=t_col, ylabel=oxide)
            _overlay_fit_vs_t(ax, t_col, oxide)
            ax.tick_params(labelsize=tick_fs)

    # KDE rows with HDR
    def _draw_kde_rows():
        slots = [(3,0),(3,1),(3,2),(3,3),(4,0),(4,1),(4,2),(4,3)]
        for name, (rb, cb) in zip(kde_cols, slots):
            ax = _ax_at(rb, cb)
            grid = kde_xy[name]
            x = grid['x'].to_numpy(); y = grid['y'].to_numpy()
            ax.plot(x, y, lw=1.2)
            # HDR bands
            for (left, right) in intervals.get(name, []):
                ax.axvspan(left, right, color='tab:blue', alpha=0.25, lw=0)
            # Threshold line
            ythr = ythr_map.get(name, None)
            if ythr is not None and np.isfinite(ythr):
                ax.axhline(ythr, color='red', ls='--', lw=1.0, alpha=0.7)
            # Rug of observed values for this oxide
            if name in df.columns:
                vals = df[name].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ymin, ymax = ax.get_ylim()
                    yr = ymin + 0.03 * (ymax - ymin)
                    ax.vlines(vals, ymin, yr, color='k', lw=0.5, alpha=0.25)
            ax.set_xlabel(name, fontsize=tick_fs)
            ax.set_ylabel('Density', fontsize=tick_fs)

    # assemble
    _draw_summary_row()
    _draw_harker_rows()
    _draw_kde_rows()

    # Colorbar
    cax = fig.add_axes([0.935, 0.10, 0.015, 0.78])
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label('geo_score (0 to 1)', fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    # Title
    kept = int(passed.sum()); tot = len(df)
    fig.suptitle(f"Kept: {kept} / {tot}  |  geo_min={geo_min}  |  combine_mode={combine_mode}", fontsize=12, y=0.06)

    return fig