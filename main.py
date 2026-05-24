"""
SOC analysis in alveolar crumb structures
Author: Jorge L. Pineda — CEMSC3, UNSAM
Course: Computational Neuroscience 2026
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from PIL import Image
import csv
from scipy import ndimage as ndi
from scipy.stats import lognorm

warnings.filterwarnings('ignore')

try:
    from skimage import measure, filters
    from skimage.morphology import closing, disk, remove_small_objects
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    import cv2
    import powerlaw
except ImportError as e:
    print(f"[ERROR] Dependencia faltante: {e}")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════=
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════=
BREAD_CONFIG = {
    'Sourdough':  {'gamma_molnar': 1.331, 'color': '#E74C3C', 'width_mm': 140.0, 'height_mm': 120.0},
    'Wheat':      {'gamma_molnar': 1.858, 'color': '#3498DB', 'width_mm': 115.0, 'height_mm': 115.0},
    'Brown':      {'gamma_molnar': 1.637, 'color': '#8E44AD', 'width_mm': 115.0, 'height_mm': 100.0},
    'Croissant':  {'gamma_molnar': 1.289, 'color': '#F39C12', 'width_mm': 120.0, 'height_mm':  65.0},
    'Baguette':   {'gamma_molnar': 1.265, 'color': '#27AE60', 'width_mm':  75.0, 'height_mm':  75.0},
    'Homemade':   {'gamma_molnar': 1.638, 'color': '#95A5A6', 'width_mm': 120.0, 'height_mm': 110.0},
}

MIN_BUBBLE_AREA_PX = 15
CLOSE_RADIUS       = 2
ROI_RADIUS_FRAC    = 0.44
N_BOOTSTRAP        = 200     # iteraciones bootstrap para CI de gamma
OUTPUT_DIR         = 'out'

# ═════════════════════════════════════════════════════════════════════════════=
# MODULE 1: IMAGE PROCESSING
# ═════════════════════════════════════════════════════════════════════════════=

def procesar_imagen(path: str, width_mm: float, height_mm: float) -> dict:
    """Watershed segmentation of bubbles in a slice."""
    arr_rgb = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
    gray    = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq  = clahe.apply(gray)

    thresh_val = filters.threshold_otsu(gray_eq)
    binary     = gray_eq < thresh_val
    binary     = closing(binary, disk(CLOSE_RADIUS))
    binary     = remove_small_objects(binary, min_size=MIN_BUBBLE_AREA_PX)

    dist   = ndi.distance_transform_edt(binary)
    coords = peak_local_max(dist, footprint=np.ones((3, 3)), labels=binary)

    h_px, w_px  = binary.shape
    scale_x     = width_mm  / w_px
    scale_y     = height_mm / h_px
    px_area_mm2 = scale_x * scale_y

    if coords.size == 0:
        return {
            'arr_rgb': arr_rgb, 'labeled': np.zeros_like(binary, dtype=int),
            'sizes_mm2': np.array([]), 'cx': w_px//2, 'cy': h_px//2,
            'r': int(min(h_px, w_px) * ROI_RADIUS_FRAC),
            'sx': scale_x, 'sy': scale_y,
            'thresh': thresh_val, 'porosidad': 0.0,
            'h_px': h_px, 'w_px': w_px,
        }

    mask_peaks = np.zeros(dist.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_peaks)
    labeled     = watershed(-dist, markers, mask=binary)

    cx_px = w_px // 2
    cy_px = h_px // 2
    r_px  = int(min(h_px, w_px) * ROI_RADIUS_FRAC)
    roi_mm2 = np.pi * (r_px * scale_x) * (r_px * scale_y)

    labeled_roi        = np.zeros_like(labeled)
    sizes_mm2          = []
    total_bubble_area  = 0.0
    new_id             = 1

    for p in measure.regionprops(labeled):
        ry, rx = p.centroid
        in_roi      = (rx - cx_px)**2 + (ry - cy_px)**2 <= r_px**2
        big_enough  = p.area >= MIN_BUBBLE_AREA_PX
        if in_roi and big_enough:
            labeled_roi[labeled == p.label] = new_id
            area = p.area * px_area_mm2
            sizes_mm2.append(area)
            total_bubble_area += area
            new_id += 1

    porosidad = (total_bubble_area / roi_mm2 * 100.0) if roi_mm2 > 0 else 0.0

    return {
        'arr_rgb': arr_rgb, 'labeled': labeled_roi,
        'sizes_mm2': np.array(sizes_mm2, dtype=np.float64),
        'cx': cx_px, 'cy': cy_px, 'r': r_px,
        'sx': scale_x, 'sy': scale_y,
        'thresh': thresh_val, 'porosidad': porosidad,
        'h_px': h_px, 'w_px': w_px,
    }


# ═════════════════════════════════════════════════════════════════════════════=
# MODULE 2: STATISTICAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════=

def ajustar_distribucion(sizes: np.ndarray, xmin_mm2: float,
                         n_boot: int = N_BOOTSTRAP) -> dict:
    """
    MLE estimation of gamma, Vuong LR test, and bootstrap CI.
    Bootstrap: resample with replacement N_boot times → 95% CI for gamma.
    """
    valid = sizes[sizes >= xmin_mm2]
    n     = len(valid)

    resultado = {
        'gamma': np.nan, 'gamma_ci_low': np.nan, 'gamma_ci_high': np.nan,
        'xmin': xmin_mm2, 'n_fit': n, 'robusto': n >= 50,
        'R_lr': np.nan, 'p_lr': np.nan,
        'D_ks': np.nan,
        'mu_ln': np.nan, 'sigma_ln': np.nan,
        'sigma_branching': np.nan,
    }

    if n < 10:
        return resultado

    fit = powerlaw.Fit(valid, discrete=False, xmin=xmin_mm2, verbose=False)
    resultado['gamma'] = fit.power_law.alpha

    try:
        resultado['D_ks'] = fit.power_law.D
    except Exception:
        pass

    try:
        R, p = fit.distribution_compare('power_law', 'lognormal',
                                        normalized_ratio=True)
        resultado['R_lr'] = R
        resultado['p_lr'] = p
    except Exception:
        pass

    try:
        shape, _, scale = lognorm.fit(valid, floc=0)
        resultado['sigma_ln'] = shape
        resultado['mu_ln']    = np.log(scale)
    except Exception:
        pass

    # Bootstrap CI para gamma (Clauset et al. 2009, §4)
    if n >= 50 and n_boot > 0:
        gammas_boot = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            sample = rng.choice(valid, size=n, replace=True)
            try:
                f_b = powerlaw.Fit(sample, discrete=False,
                                   xmin=xmin_mm2, verbose=False)
                gammas_boot.append(f_b.power_law.alpha)
            except Exception:
                pass
        if len(gammas_boot) >= 10:
            resultado['gamma_ci_low']  = float(np.percentile(gammas_boot, 2.5))
            resultado['gamma_ci_high'] = float(np.percentile(gammas_boot, 97.5))

    # Branching sigma: there is NO direct estimator from a static γ.
    # The formula 1-1/(γ-1) is a mean-field approximation without standard support.
    # For γ≤2 it yields negative/NaN values; deprecated. Requires time-series data.
    resultado['sigma_branching'] = np.nan  # Requires time-series estimation (Zeraati et al. 2022)

    return resultado


# ═════════════════════════════════════════════════════════════════════════════=
# MODULE 3: GREENBERG-HASTINGS (GH) MODEL
# ═════════════════════════════════════════════════════════════════════════════=

def greenberg_hastings(grid_size: int = 200,
                       n_steps:   int = 5000,
                       p_excite:  float = 0.002,
                       n_states:  int = 3,
                       seed:      int = 42) -> dict:
    """
    Greenberg-Hastings cellular automaton (1978) on a 2D lattice.

    States:
      0 → Quiescent (susceptible)
      1 → Excited   (firing)
      2..n_states-1 → Refractory (recovering)

    Rules (Moore neighbourhood):
      Quiescent → Excited if ≥1 neighbor is excited OR with probability p_excite (spontaneous)
      Excited   → Refractory (state 2)
      Refractory → decrements towards 0

    Returns:
      avalanche_sizes: array of avalanche sizes (number of excited cells per step)
      grid_history:    final grid for visualization
      gamma_gh:        exponent estimated by MLE
    """
    rng   = np.random.default_rng(seed)
    grid  = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Random initial seed
    init_mask = rng.random((grid_size, grid_size)) < 0.05
    grid[init_mask] = 1

    avalanche_sizes = []

    for step in range(n_steps):
        new_grid = grid.copy()

        # Excitados → Refractario
        exc_mask = (grid == 1)
        new_grid[exc_mask] = 2

        # Refractarios → decrementan (el último refractario → Quiescente)
        for s in range(2, n_states):
            ref_mask = (grid == s)
            new_grid[ref_mask] = s - 1 if s > 2 else 0  # FIX: R₂→Q, no R₂→E

        # Quiescentes → excitados si tienen vecino excitado
        quiescent = (grid == 0)

        # Convolución para detectar vecinos excitados (Moore 8-conectado)
        neighbor_exc = ndi.convolve(exc_mask.astype(np.float32),
                                    np.ones((3, 3), dtype=np.float32),
                                    mode='wrap') > 0

        spontaneous = rng.random((grid_size, grid_size)) < p_excite
        activate    = quiescent & (neighbor_exc | spontaneous)
        new_grid[activate] = 1

        grid = new_grid

            # Avalanche size = # excited cells at this time step
        size = int(np.sum(grid == 1))
        if size > 0:
            avalanche_sizes.append(size)

    avalanche_sizes = np.array(avalanche_sizes, dtype=np.float64)

    # Estimate gamma of the GH model
    gamma_gh   = np.nan
    ci_low_gh  = np.nan
    ci_high_gh = np.nan

    if len(avalanche_sizes) >= 50:
        xmin_gh = np.percentile(avalanche_sizes, 10)
        try:
            fit_gh   = powerlaw.Fit(avalanche_sizes[avalanche_sizes >= xmin_gh],
                                    discrete=True, xmin=xmin_gh, verbose=False)
            gamma_gh = fit_gh.power_law.alpha

            # Bootstrap CI (100 iteraciones)
            boot_g = []
            rng2   = np.random.default_rng(99)
            valid_gh = avalanche_sizes[avalanche_sizes >= xmin_gh]
            for _ in range(100):
                s = rng2.choice(valid_gh, size=len(valid_gh), replace=True)
                try:
                    f2 = powerlaw.Fit(s, discrete=True, xmin=xmin_gh, verbose=False)
                    boot_g.append(f2.power_law.alpha)
                except Exception:
                    pass
            if len(boot_g) >= 10:
                ci_low_gh  = float(np.percentile(boot_g, 2.5))
                ci_high_gh = float(np.percentile(boot_g, 97.5))
        except Exception:
            pass

    return {
        'avalanche_sizes': avalanche_sizes,
        'grid_final':      grid,
        'gamma_gh':        gamma_gh,
        'ci_low':          ci_low_gh,
        'ci_high':         ci_high_gh,
        'grid_size':       grid_size,
        'n_steps':         n_steps,
        'p_excite':        p_excite,
    }


def generar_figura_gh(gh_result: dict):
    """Figure for the Greenberg-Hastings model: grid + avalanche distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Greenberg-Hastings model — criticality and avalanche distribution',
                 fontsize=13, fontweight='bold')

    # Panel 1: Estado final del autómata
    ax = axes[0]
    cmap = mcolors.ListedColormap(['#2C3E50', '#E74C3C', '#F39C12'])
    im   = ax.imshow(gh_result['grid_final'], cmap=cmap,
                     vmin=0, vmax=2, interpolation='nearest')
    ax.set_title(f"Final State (t={gh_result['n_steps']})\n"
                 f"Black=Quiescent  Red=Excited  Orange=Refractory",
                 fontsize=10)
    ax.set_xlabel('Cell x', fontsize=9)
    ax.set_ylabel('Cell y', fontsize=9)
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2],
                 label='State: 0=Q, 1=E, 2=R')

    # Panel 2: Distribución de avalanchas en log-log
    ax = axes[1]
    sizes = gh_result['avalanche_sizes']
    if len(sizes) >= 20:
        bins   = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 30)
        counts, edges = np.histogram(sizes, bins=bins)
        ctrs   = (edges[:-1] + edges[1:]) / 2.0
        widths = edges[1:] - edges[:-1]
        mask   = counts > 0
        pdf    = counts / (counts.sum() * widths + 1e-30)

        ax.scatter(ctrs[mask], pdf[mask], s=50, color='#2980B9',
                   edgecolors='k', lw=0.5, label='Empirical GH', zorder=3)

        g = gh_result['gamma_gh']
        if not np.isnan(g):
            xmin_gh  = np.percentile(sizes, 10)
            x_fit    = np.logspace(np.log10(xmin_gh), np.log10(sizes.max()), 100)
            ref_mask = ctrs >= xmin_gh
            if ref_mask.any():
                ref_idx = np.where(ref_mask)[0][0]
                c_pl    = pdf[ref_idx] * (ctrs[ref_idx] ** g)
                ci_str  = ''
                if not np.isnan(gh_result['ci_low']):
                    ci_str = f' [{gh_result["ci_low"]:.2f}, {gh_result["ci_high"]:.2f}]'
                ax.plot(x_fit, c_pl * x_fit ** (-g), color='#C0392B', lw=2.2,
                        label=f'PL γ={g:.3f}{ci_str}', zorder=4)

        # Neural reference line γ=1.5
        ref_mask2 = ctrs >= sizes.min()
        if ref_mask2.any():
            ri  = np.where(ref_mask2)[0][0]
            c15 = pdf[ri] * (ctrs[ri] ** 1.5)
            x2  = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
            ax.plot(x2, c15 * x2 ** (-1.5), color='#27AE60', lw=1.8,
                    ls='--', label='Neural ref. γ=1.5 (Beggs & Plenz)', zorder=3)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Avalanche size S (cells)', fontsize=9)
    ax.set_ylabel('Density P(S)', fontsize=9)
    ax.set_title('GH avalanche distribution\n(log-log)', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3, ls=':')

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'greenberg_hastings.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {fname}")
    return fname


# ═════════════════════════════════════════════════════════════════════════════=
# MODULE 4: VISUALIZATION (3 Slices × Variety)
# ═════════════════════════════════════════════════════════════════════════════=

def _colorear_labels(labeled: np.ndarray) -> np.ndarray:
    """HSV colors for connected components."""
    n   = labeled.max()
    rgb = np.ones((*labeled.shape, 3), dtype=np.float32)
    if n == 0:
        return rgb
    np.random.seed(42)
    hues   = np.random.permutation(np.linspace(0, 1, n + 1)[:-1])
    colors = mcolors.hsv_to_rgb(
        np.column_stack([hues, np.full(n, 0.82), np.full(n, 0.92)])
    )
    for i, col in enumerate(colors):
        rgb[labeled == i + 1] = col
    return rgb


def generar_figura_variedad(variedad: str, rodajas_data: list, cfg: dict):
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 3, wspace=0.28, hspace=0.45,
                           left=0.06, right=0.96, top=0.90, bottom=0.04)

    # Two-line title to control style and spacing with panels.
    fig.text(0.5, 0.975, f'{variedad} — 3-slice analysis',
             ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.947,
             'Computational Neuroscience 2026 — CEMSC3 — UNSAM',
             ha='center', va='top', fontsize=12, fontweight='normal', alpha=0.72)

    for row, data in enumerate(rodajas_data):
        # Capture sx/sy by value to avoid closure bug
        sx_local = data['sx']
        sy_local = data['sy']

        # ── Col 0: Original image ──────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(data['arr_rgb'], interpolation='bilinear')

        # FIX: default arguments capture by value, not by reference
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, pos, s=sx_local: f"{v * s:.0f}"))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos, s=sy_local: f"{v * s:.0f}"))

        ax.set_xlabel('mm', fontsize=9)
        ax.set_ylabel('mm', fontsize=9)
        ax.set_title(f'Slice {row + 1} — Original', fontsize=11, fontweight='bold')
        ax.add_patch(mpatches.Circle(
            (data['cx'], data['cy']), data['r'],
            color=cfg['color'], fill=False, lw=2, ls='--'))

        # ── Col 1: Watershed ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(_colorear_labels(data['labeled']), interpolation='nearest')
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, pos, s=sx_local: f"{v * s:.0f}"))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos, s=sy_local: f"{v * s:.0f}"))
        ax.set_xlabel('mm', fontsize=9)
        ax.set_ylabel('mm', fontsize=9)

        n_bub = len(data['sizes_mm2'])
        ax.set_title(f'Slice {row + 1} — Watershed (N={n_bub})',
                     fontsize=11, fontweight='bold')
        ax.add_patch(mpatches.Circle(
            (data['cx'], data['cy']), data['r'],
            color='k', fill=False, lw=1.5))
        info_str = (f"Otsu = {data['thresh']:.0f}\n"
                    f"Porosity = {data['porosidad']:.1f}%")
        ax.text(0.03, 0.96, info_str, transform=ax.transAxes, fontsize=8,
                va='top', bbox=dict(fc='white', alpha=0.88, pad=3, ec='grey'))

        # ── Col 2: Log-log distribution ─────────────────────────────────────
        ax   = fig.add_subplot(gs[row, 2])
        sizes = data['sizes_mm2']

        if len(sizes) >= 10:
            smin = sizes.min()
            smax = sizes.max()
            if smin >= smax:
                smax = smin * 10.0

            bins   = np.logspace(np.log10(smin), np.log10(smax), 28)
            counts, edges = np.histogram(sizes, bins=bins)
            ctrs   = (edges[:-1] + edges[1:]) / 2.0
            widths = edges[1:] - edges[:-1]
            valid_mask = counts > 0
            pdf    = np.where(valid_mask,
                              counts / (counts.sum() * widths + 1e-30), np.nan)

            ax.scatter(ctrs[valid_mask], pdf[valid_mask],
                       s=40, color=cfg['color'], alpha=0.88,
                       edgecolors='k', lw=0.5, label='Empirical', zorder=3)

            # FIX: usar xmin del stat, no recalcular
            xmin_mm2 = data['stat']['xmin']
            x_fit    = np.logspace(np.log10(xmin_mm2), np.log10(smax), 100)
            stat     = data['stat']

            # Power Law MLE (rojo continuo)
            if not np.isnan(stat['gamma']):
                ref_mask = ctrs >= xmin_mm2
                if ref_mask.any():
                    ri   = np.where(ref_mask)[0][0]
                    c_pl = pdf[ri] * (ctrs[ri] ** stat['gamma'])
                    g    = stat['gamma']
                    ci_str = ''
                    if not np.isnan(stat['gamma_ci_low']):
                        ci_str = (f' [{stat["gamma_ci_low"]:.2f},'
                                  f'{stat["gamma_ci_high"]:.2f}]')
                    ax.plot(x_fit, c_pl * x_fit ** (-g),
                            color='#C0392B', lw=2.2,
                            label=f'PL (γ={g:.3f}{ci_str})', zorder=4)

            # Referencia Molnar (verde discontinuo)
            gm = cfg['gamma_molnar']
            if not np.isnan(gm):
                ref_mask = ctrs >= xmin_mm2
                if ref_mask.any():
                    ri   = np.where(ref_mask)[0][0]
                    c_m  = pdf[ri] * (ctrs[ri] ** gm)
                    ax.plot(x_fit, c_m * x_fit ** (-gm),
                            color='#27AE60', lw=1.8, ls='--',
                            label=f'Molnar (γ={gm:.3f})', zorder=4)

            # Log-Normal MLE (violeta punteado)
            if not np.isnan(stat['mu_ln']) and not np.isnan(stat['sigma_ln']):
                scale_ln = np.exp(stat['mu_ln'])
                ax.plot(x_fit,
                        lognorm.pdf(x_fit, s=stat['sigma_ln'], scale=scale_ln),
                        color='#8E44AD', lw=1.5, ls=':',
                        label=f'LN (μ={stat["mu_ln"]:.2f})', zorder=3)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Area S (mm²)', fontsize=9)
            ax.set_ylabel('Density P(S)', fontsize=9)
            ax.set_title(f'Slice {row + 1} — Distribution',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.95)
            ax.grid(True, which='both', alpha=0.3, ls=':')
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient N', transform=ax.transAxes,
                    ha='center', va='center', color='#7F8C8D', fontsize=10)
            ax.set_axis_off()

    fname = os.path.join(OUTPUT_DIR, f'{variedad.lower()}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: EXPORTACIÓN CSV
# ══════════════════════════════════════════════════════════════════════════════

def exportar_csv(resultados_por_variedad: dict):
    def f(v, d=4):
        try:
            return f"{float(v):.{d}f}" if v is not None and not np.isnan(float(v)) else 'N/A'
        except Exception:
            return 'N/A'

    # ── CSV PER-SLICE ────────────────────────────────────────────────────────
    campos_slice = [
        'Variety', 'Slice', 'N', 'Robust',
        'gamma_MLE', 'gamma_CI_low', 'gamma_CI_high',
        'gamma_Molnar', 'Delta_gamma',
        'D_KS', 'R_LR', 'p_LR', 'mu_LN', 'sigma_LN',
        'Porosity_pct', 'xmin_mm2',
        'scale_x_mm_px', 'scale_y_mm_px',
        'Regime', 'LR_Interpretation',
    ]

    def clasificar(gf):
        if np.isnan(gf):
            return 'N/A'
        if gf < 1.45:
            return 'Sub-critical'
        if gf <= 1.55:
            return 'Critical'
        return 'Supercritical'

    def interp_lr(R_lr, p_lr):
        if np.isnan(R_lr) or np.isnan(p_lr):
            return 'N/A'
        if p_lr >= 0.10:
            return 'Indistinguishable (PL ~ LN)'
        if R_lr > 0:
            return 'PL preferred'
        return 'Log-Normal preferred'

    path_slice = os.path.join(OUTPUT_DIR, 'results_per_slice.csv')
    with open(path_slice, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=campos_slice)
        writer.writeheader()
        for variedad, rodajas in sorted(resultados_por_variedad.items()):
            gm = BREAD_CONFIG[variedad]['gamma_molnar']
            for idx, data in enumerate(rodajas):
                st = data['stat']
                gf = st['gamma']
                dg = abs(gf - gm) if not np.isnan(gf) else np.nan
                writer.writerow({
                    'Variety': variedad,
                    'Slice': idx + 1,
                    'N': len(data['sizes_mm2']),
                    'Robust': 'Yes' if st['robusto'] else 'No',
                    'gamma_MLE': f(gf),
                    'gamma_CI_low': f(st['gamma_ci_low']),
                    'gamma_CI_high': f(st['gamma_ci_high']),
                    'gamma_Molnar': f(gm),
                    'Delta_gamma': f(dg),
                    'D_KS': f(st['D_ks']),
                    'R_LR': f(st['R_lr']),
                    'p_LR': f(st['p_lr']),
                    'mu_LN': f(st['mu_ln']),
                    'sigma_LN': f(st['sigma_ln']),
                    'Porosity_pct': f(data['porosidad'], 2),
                    'xmin_mm2': f(st['xmin']),
                    'scale_x_mm_px': f(data['sx'], 5),
                    'scale_y_mm_px': f(data['sy'], 5),
                    'Regime': clasificar(gf),
                    'LR_Interpretation': interp_lr(st['R_lr'], st['p_lr']),
                })
    print(f"  → {path_slice}")

    # ── CSV AGREGADO ─────────────────────────────────────────────────────────
    campos_agg = [
        'Variety', 'N_total', 'N_fit', 'Robust',
        'gamma_MLE', 'gamma_CI_low', 'gamma_CI_high',
        'gamma_Molnar', 'Delta_gamma',
        'D_KS', 'R_LR', 'p_LR', 'mu_LN', 'sigma_LN',
        'Porosity_mean_pct', 'Porosity_std_pct',
        'xmin_mm2_mean',
        'Regime', 'LR_Interpretation',
    ]

    path_agg = os.path.join(OUTPUT_DIR, 'results_aggregate.csv')
    with open(path_agg, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=campos_agg)
        writer.writeheader()
        for variedad, rodajas in sorted(resultados_por_variedad.items()):
            sizes_all = np.concatenate([r['sizes_mm2'] for r in rodajas])
            porosidades = [r['porosidad'] for r in rodajas]
            # FIX: xmin as mean of per-slice xmins
            xmins = [r['stat']['xmin'] for r in rodajas]
            xmin_media = float(np.mean(xmins))

            stat_agg = ajustar_distribucion(sizes_all, xmin_media, n_boot=N_BOOTSTRAP)

            gf = stat_agg['gamma']
            gm = BREAD_CONFIG[variedad]['gamma_molnar']
            dg = abs(gf - gm) if not np.isnan(gf) else np.nan

            writer.writerow({
                'Variety': variedad,
                'N_total': len(sizes_all),
                'N_fit': stat_agg['n_fit'],
                'Robust': 'Yes' if stat_agg['robusto'] else 'No',
                'gamma_MLE': f(gf),
                'gamma_CI_low': f(stat_agg['gamma_ci_low']),
                'gamma_CI_high': f(stat_agg['gamma_ci_high']),
                'gamma_Molnar': f(gm),
                'Delta_gamma': f(dg),
                'D_KS': f(stat_agg['D_ks']),
                'R_LR': f(stat_agg['R_lr']),
                'p_LR': f(stat_agg['p_lr']),
                'mu_LN': f(stat_agg['mu_ln']),
                'sigma_LN': f(stat_agg['sigma_ln']),
                'Porosity_mean_pct': f(np.mean(porosidades), 2),
                'Porosity_std_pct': f(np.std(porosidades), 2),
                'xmin_mm2_mean': f(xmin_media),
                'Regime': clasificar(gf),
                'LR_Interpretation': interp_lr(stat_agg['R_lr'], stat_agg['p_lr']),
            })
    print(f"  → {path_agg}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    img_dir    = os.path.join(script_dir, 'img')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  SOC Analysis — 3 Slices × Variety")
    print("=" * 70)

    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image folder not found: {img_dir}")
        print("        Create an 'img/' folder next to this script and place the images inside.")
        sys.exit(1)

    resultados_por_variedad = {}

    for nombre, cfg in BREAD_CONFIG.items():
        print(f"\n[{nombre}]  {cfg['width_mm']:.0f} × {cfg['height_mm']:.0f} mm")
        rodajas_data = []

        for slice_idx in range(1, 4):
            fname    = f"{nombre.lower()}_{slice_idx}.png"
            img_path = os.path.join(img_dir, fname)

            if not os.path.exists(img_path):
                print(f"  [Slice {slice_idx}] ⚠ Not found: {fname}")
                continue

            print(f"  [Slice {slice_idx}] Processing {fname}...")
            data     = procesar_imagen(img_path, cfg['width_mm'], cfg['height_mm'])
            xmin_mm2 = MIN_BUBBLE_AREA_PX * data['sx'] * data['sy']
            stat     = ajustar_distribucion(data['sizes_mm2'], xmin_mm2,
                                            n_boot=N_BOOTSTRAP)

            gf = stat['gamma']
            ci_str = ''
            if not np.isnan(stat['gamma_ci_low']):
                ci_str = f" CI=[{stat['gamma_ci_low']:.3f},{stat['gamma_ci_high']:.3f}]"
            print(f"       N={len(data['sizes_mm2'])} | "
                  f"Robust:{'✓' if stat['robusto'] else '⚠'} | "
                  f"Porosity={data['porosidad']:.1f}%")
            if not np.isnan(gf):
                r_lr_str = f"{stat['R_lr']:.3f}" if not np.isnan(stat['R_lr']) else 'N/A'
                print(f"       γ={gf:.4f}{ci_str} | "
                      f"R_LR={r_lr_str}")

            rodajas_data.append({**data, 'stat': stat})

        if rodajas_data:
            resultados_por_variedad[nombre] = rodajas_data
            generar_figura_variedad(nombre, rodajas_data, cfg)

    # ── Greenberg-Hastings model ────────────────────────────────────────────
    print("\n[GH] Simulating Greenberg-Hastings model...")
    gh_result = greenberg_hastings(grid_size=200, n_steps=5000,
                                   p_excite=0.002, n_states=3)
    print(f"     Avalanchas: {len(gh_result['avalanche_sizes'])}")
    if not np.isnan(gh_result['gamma_gh']):
        print(f"     γ_GH = {gh_result['gamma_gh']:.4f} "
              f"CI=[{gh_result['ci_low']:.3f},{gh_result['ci_high']:.3f}]")
    generar_figura_gh(gh_result)

    # ── Exportar CSVs ────────────────────────────────────────────────────────
    print("\n[CSV] Exporting results...")
    exportar_csv(resultados_por_variedad)

    print("\n" + "=" * 70)
    print("  ✓ ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print("  • [variety].png          — 3-slice figure per variety")
    print("  • greenberg_hastings.png  — GH model")
    print("  • results_per_slice.csv")
    print("  • results_aggregate.csv")
    print()


if __name__ == '__main__':
    main()