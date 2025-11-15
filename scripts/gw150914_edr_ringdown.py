# scripts/gw150914_gr_Ma_fit.py
"""
Ajuste del ringdown de GW150914 (H1) con GR puro, dejando libres
la masa final M_f y el espín a_f del agujero negro remanente.

Compara la frecuencia 220 obtenida con la frecuencia GR estándar
para M_f = 68 Msun y a_f = 0.67, y calcula un corrimiento efectivo:

  alpha_eff = f_220(M_f_fit, a_f_fit) / f_220(68, 0.67) - 1

para comparar con el alpha_flow ~ -0.5 obtenido en el modelo EDR.

Requiere que los archivos HDF5 estén en data/, generados por
scripts/download_data.py
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constantes físicas (SI)
G_SI    = 6.67430e-11
C_SI    = 2.99792458e8
MSUN_SI = 1.98847e30

# Valores "estándar" de M_f y a_f usados antes
M_REF_SOLAR = 68.0
A_REF       = 0.67


def geom_time_from_mass_solar(m_solar: float) -> float:
    """Convierte masa en Msun a tiempo geométrico M = G M / c^3 (s)."""
    return G_SI * m_solar * MSUN_SI / C_SI**3


def qnm_220_GR(M_final_solar: float, a_final: float) -> tuple[float, float]:
    """
    Frecuencia (Hz) y tiempo de decaimiento (s) del modo (2,2,0) en GR,
    usando los ajustes de Berti et al.
    """
    # Coeficientes Berti para (2,2,0)
    f1, f2, f3 = 1.5251, -1.1568, 0.1292
    g1, g2, g3 = 0.0889, -0.0260, 0.3350

    M_geom = geom_time_from_mass_solar(M_final_solar)
    one_minus_a = 1.0 - a_final

    omegaR_M = f1 + f2 * (one_minus_a**f3)
    omegaI_M = g1 + g2 * (one_minus_a**g3)

    omegaR_SI = omegaR_M / M_geom
    omegaI_SI = omegaI_M / M_geom

    f_Hz  = omegaR_SI / (2.0 * np.pi)
    tau_s = 1.0 / abs(omegaI_SI)
    return f_Hz, tau_s


def ringdown_model_GR_220_Ma(t: np.ndarray,
                             M_final_solar: float,
                             a_final: float,
                             A220: float,
                             phi220: float) -> np.ndarray:
    """
    Modelo GR puro para el ringdown:

      h(t) = A220 * exp(-t / tau_220) * cos(2π f_220 t + phi220)

    donde f_220 y tau_220 dependen de M_f y a_f.
    """
    f220_GR, tau220_GR = qnm_220_GR(M_final_solar, a_final)
    omega220 = 2.0 * np.pi * f220_GR
    return A220 * np.exp(-t / tau220_GR) * np.cos(omega220 * t + phi220)


def load_strain_from_hdf5(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga el strain y los tiempos desde un archivo HDF5 LOSC."""
    with h5py.File(path, "r") as f:
        dset = f["strain/Strain"]
        strain = dset[()]
        t0 = dset.attrs["Xstart"]
        dt = dset.attrs["Xspacing"]

    n = len(strain)
    times = t0 + np.arange(n) * dt
    return times, strain


def chi2_stats(y: np.ndarray, y_model: np.ndarray, n_params: int) -> tuple[float, float, float, float]:
    """
    Devuelve S, chi2_red, AIC y BIC para un modelo dado (sigma = 1).
    """
    n = len(y)
    resid = y - y_model
    S = np.sum(resid**2)
    chi2 = S
    dof = n - n_params
    chi2_red = chi2 / dof if dof > 0 else np.nan
    AIC = 2 * n_params + n * np.log(S / n)
    BIC = n_params * np.log(n) + n * np.log(S / n)
    return S, chi2_red, AIC, BIC


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")
    h1_file = os.path.join(data_dir, "H-H1_LOSC_4_V1-1126259446-32.hdf5")

    if not os.path.exists(h1_file):
        raise FileNotFoundError(
            f"No se encontró {h1_file}. Ejecuta primero scripts/download_data.py"
        )

    # 1. Cargar señal H1
    t, h = load_strain_from_hdf5(h1_file)
    h = h - np.mean(h)

    # 2. Ventana de ringdown
    idx_peak = np.argmax(np.abs(h))
    t_peak = t[idx_peak]

    t_start = t_peak + 0.002  # 2 ms después del pico
    t_end   = t_peak + 0.030  # hasta 30 ms después del pico

    mask = (t >= t_start) & (t <= t_end)
    t_rd = t[mask]
    h_rd = h[mask]

    t_rel = t_rd - t_rd[0]

    # 3. Normalizar señal
    h_rms = np.sqrt(np.mean(h_rd**2))
    h_norm = h_rd / h_rms if h_rms != 0 else h_rd

    print(f"Ventana de ringdown: duración ~ {t_rel[-1] - t_rel[0]:.6f} s, puntos = {len(t_rel)}")
    print(f"RMS de la ventana: {h_rms:.3e}")

    # 4. Ajuste GR con M_f y a_f libres
    def model_to_fit(t_arr, M_solar, a_final, A220, phi220):
        return ringdown_model_GR_220_Ma(t_arr, M_solar, a_final, A220, phi220)

    # Estimaciones iniciales
    M0   = M_REF_SOLAR       # punto de partida: 68 Msun
    a0   = A_REF             # 0.67
    A0   = np.max(h_norm)
    phi0 = 0.0

    p0 = [M0, a0, A0, phi0]

    # Cotas físicas razonables para M_f y a_f
    bounds = (
        [40.0, 0.1, -10 * abs(A0), -2 * np.pi],   # M_f >= 40 Msun, 0.1 <= a <= 0.99
        [100.0, 0.99, 10 * abs(A0),  2 * np.pi],
    )

    print("\n[Ajuste GR con M_f y a_f libres (sin alpha_flow, sin modo X)]")
    popt, pcov = curve_fit(
        model_to_fit,
        t_rel,
        h_norm,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))

    M_fit, a_fit, A220_fit, phi220_fit = popt
    dM, da, dA, dphi = perr

    print("\nResultados del ajuste GR libre (M_f, a_f, A220, phi220):")
    print(f"M_f^fit   = {M_fit:.2f} ± {dM:.2f} Msun")
    print(f"a_f^fit   = {a_fit:.3f} ± {da:.3f}")
    print(f"A220      = {A220_fit:.3e} ± {dA:.3e}")
    print(f"phi220    = {phi220_fit:.3f} ± {dphi:.3f} rad")

    # 5. Frecuencias 220 y corrimiento efectivo
    f_ref, tau_ref   = qnm_220_GR(M_REF_SOLAR, A_REF)
    f_fit, tau_fit   = qnm_220_GR(M_fit, a_fit)

    alpha_eff = f_fit / f_ref - 1.0

    print("\nFrecuencias 220 (GR):")
    print(f"f_220(M_ref={M_REF_SOLAR:.1f}, a_ref={A_REF:.2f}) = {f_ref:.1f} Hz")
    print(f"f_220(M_fit, a_fit)                       = {f_fit:.1f} Hz")
    print(f"alpha_eff (solo por M_f,a_f)             = {alpha_eff:.3f}")

    # 6. Estadísticos de ajuste
    h_model = model_to_fit(t_rel, *popt)
    S, chi2_red, AIC, BIC = chi2_stats(h_norm, h_model, n_params=4)

    print("\nEstadísticos del ajuste GR libre (ventana normalizada):")
    print(f"S = {S:.4e}, chi2_red = {chi2_red:.3f}, AIC = {AIC:.2f}, BIC = {BIC:.2f}")

    # 7. Figura simple: datos vs modelo GR libre
    h_model_phys = h_model * h_rms
    resid = h_norm - h_model

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t_rel * 1000, h_rd,          label="Datos H1 (ringdown)", lw=1, color="C0")
    ax1.plot(t_rel * 1000, h_model_phys,  label="GR libre (M_f,a_f)",   lw=2, color="C1")
    ax1.set_ylabel("Strain")
    ax1.set_title("GW150914 (H1): Ringdown GR con M_f,a_f libres")
    ax1.grid(True)
    ax1.legend()

    ax2.axhline(0.0, color="k", lw=1, ls="--")
    ax2.plot(t_rel * 1000, resid, label="Datos - GR libre", lw=1, color="C1")
    ax2.set_xlabel("Tiempo desde inicio de la ventana [ms]")
    ax2.set_ylabel("Residuo (norm.)")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    out_png = os.path.join(repo_root, "data", "gw150914_gr_Ma_fit.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nFigura guardada en {out_png}")


if __name__ == "__main__":
    main()
