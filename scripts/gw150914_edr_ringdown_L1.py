# scripts/gw150914_edr_ringdown_L1.py
"""
Comparación de modelos de ringdown para GW150914 (L1):

- Modelo 0: GR (solo modo 220)
- Modelo 1: GR + alpha_flow (220 corregido, sin modo X)
- Modelo 2: GR + modo X (220 GR, alpha_flow = 0)
- Modelo 3: EDR completo (220 corregido + X)

Calcula parámetros ajustados, S, chi2_red, AIC y BIC para cada modelo,
y genera una figura con datos, GR y EDR completo + residuales.

Requiere que los archivos HDF5 de L1 estén en data/, generados por
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

# Parámetros del remanente de GW150914 (aprox.)
M_FINAL_SOLAR = 68.0
A_FINAL       = 0.67


def geom_time_from_mass_solar(m_solar: float) -> float:
    return G_SI * m_solar * MSUN_SI / C_SI**3


def qnm_220_GR(m_final_solar: float, a_final: float) -> tuple[float, float]:
    """Frecuencia (Hz) y tiempo de decaimiento (s) del modo (2,2,0) en GR."""
    f1, f2, f3 = 1.5251, -1.1568, 0.1292
    g1, g2, g3 = 0.0889, -0.0260, 0.3350

    M_geom = geom_time_from_mass_solar(m_final_solar)
    one_minus_a = 1.0 - a_final

    omegaR_M = f1 + f2 * (one_minus_a**f3)
    omegaI_M = g1 + g2 * (one_minus_a**g3)

    omegaR_SI = omegaR_M / M_geom
    omegaI_SI = omegaI_M / M_geom

    f_Hz  = omegaR_SI / (2.0 * np.pi)
    tau_s = 1.0 / abs(omegaI_SI)
    return f_Hz, tau_s


def omega_horizon(m_final_solar: float, a_final: float) -> float:
    """Frecuencia angular del horizonte de Kerr (rad/s)."""
    M_geom = geom_time_from_mass_solar(m_final_solar)
    a = a_final
    rp_geom = M_geom * (1.0 + np.sqrt(1.0 - a**2))
    omega_H = a / (2.0 * M_geom * rp_geom)
    return omega_H


def qnm_EDR_parameters(alpha_flow: float = 0.0,
                       beta_flow: float = 0.0,
                       k_flow: float = 1.0,
                       gamma_flow: float = 10.0) -> tuple[float, float, float, float]:
    """
    Frecuencias (Hz) y tiempos (s) para:
      - 220^EDR
      - modo nuevo X
    """
    f220_GR, tau220_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)

    f220_EDR   = f220_GR   * (1.0 + alpha_flow)
    tau220_EDR = tau220_GR * (1.0 + beta_flow)

    omega_H = omega_horizon(M_FINAL_SOLAR, A_FINAL)
    fX_Hz   = (k_flow * omega_H) / (2.0 * np.pi)
    M_geom  = geom_time_from_mass_solar(M_FINAL_SOLAR)
    tauX_s  = gamma_flow * M_geom

    return f220_EDR, tau220_EDR, fX_Hz, tauX_s


def ringdown_model_GR_220(t: np.ndarray,
                          A220: float,
                          phi220: float) -> np.ndarray:
    """Solo modo 220 en GR (sin correcciones EDR)."""
    f220_GR, tau220_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)
    omega220 = 2.0 * np.pi * f220_GR
    return A220 * np.exp(-t / tau220_GR) * np.cos(omega220 * t + phi220)


def ringdown_model_EDR_full(t: np.ndarray,
                            A220: float, phi220: float,
                            AX: float, phiX: float,
                            alpha_flow: float,
                            k_flow: float = 1.0,
                            gamma_flow: float = 10.0) -> np.ndarray:
    """Modelo EDR completo: 220 corregido + modo X."""
    f220_EDR, tau220_EDR, fX_Hz, tauX_s = qnm_EDR_parameters(
        alpha_flow=alpha_flow,
        beta_flow=0.0,
        k_flow=k_flow,
        gamma_flow=gamma_flow,
    )

    omega220 = 2.0 * np.pi * f220_EDR
    omegaX   = 2.0 * np.pi * fX_Hz

    h220 = A220 * np.exp(-t / tau220_EDR) * np.cos(omega220 * t + phi220)
    hX   = AX   * np.exp(-t / tauX_s)    * np.cos(omegaX   * t + phiX)

    return h220 + hX


def ringdown_model_GR_plus_X(t: np.ndarray,
                             A220: float, phi220: float,
                             AX: float, phiX: float,
                             k_flow: float = 1.0,
                             gamma_flow: float = 10.0) -> np.ndarray:
    """Modelo 2: 220 GR + modo X (alpha_flow = 0)."""
    f220_GR, tau220_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)
    omega220 = 2.0 * np.pi * f220_GR

    _, _, fX_Hz, tauX_s = qnm_EDR_parameters(
        alpha_flow=0.0,
        beta_flow=0.0,
        k_flow=k_flow,
        gamma_flow=gamma_flow,
    )
    omegaX = 2.0 * np.pi * fX_Hz

    h220 = A220 * np.exp(-t / tau220_GR) * np.cos(omega220 * t + phi220)
    hX   = AX   * np.exp(-t / tauX_s)    * np.cos(omegaX   * t + phiX)
    return h220 + hX


def ringdown_model_GR_alpha(t: np.ndarray,
                            A220: float, phi220: float,
                            alpha_flow: float) -> np.ndarray:
    """Modelo 1: 220 corregido con alpha_flow, sin modo X."""
    f220_EDR, tau220_EDR, _, _ = qnm_EDR_parameters(
        alpha_flow=alpha_flow,
        beta_flow=0.0,
        k_flow=1.0,
        gamma_flow=10.0,
    )
    omega220 = 2.0 * np.pi * f220_EDR
    return A220 * np.exp(-t / tau220_EDR) * np.cos(omega220 * t + phi220)


def load_strain_from_hdf5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        dset = f["strain/Strain"]
        strain = dset[()]
        t0 = dset.attrs["Xstart"]
        dt = dset.attrs["Xspacing"]

    n = len(strain)
    times = t0 + np.arange(n) * dt
    return times, strain


def chi2_stats(y: np.ndarray, y_model: np.ndarray, n_params: int) -> tuple[float, float, float, float]:
    """Devuelve S, chi2_red, AIC y BIC para un modelo dado (sigma = 1)."""
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
    l1_file = os.path.join(data_dir, "L-L1_LOSC_4_V1-1126259446-32.hdf5")

    if not os.path.exists(l1_file):
        raise FileNotFoundError(
            f"No se encontró {l1_file}. Ejecuta primero scripts/download_data.py"
        )

    # 1. Cargar señal L1
    t, h = load_strain_from_hdf5(l1_file)
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

    print(f"Ventana de ringdown (L1): duración ~ {t_rel[-1] - t_rel[0]:.6f} s, puntos = {len(t_rel)}")
    print(f"RMS de la ventana (L1): {h_rms:.3e}")

    # ==========
    # Modelo 0: GR puro (220)
    # ==========
    def m0_gr(t_arr, A220, phi220):
        return ringdown_model_GR_220(t_arr, A220, phi220)

    A220_0 = np.max(h_norm)
    phi220_0 = 0.0
    p0_m0 = [A220_0, phi220_0]
    bounds_m0 = ([-10 * abs(A220_0), -2 * np.pi],
                 [ 10 * abs(A220_0),  2 * np.pi])

    print("\n[Ajuste Modelo 0 (L1): GR (solo 220)]")
    popt_m0, pcov_m0 = curve_fit(
        m0_gr, t_rel, h_norm, p0=p0_m0, bounds=bounds_m0, maxfev=20000
    )
    perr_m0 = np.sqrt(np.diag(pcov_m0))
    A220_m0, phi220_m0 = popt_m0
    h0 = m0_gr(t_rel, *popt_m0)

    # ==========
    # Modelo 1: GR + alpha_flow (220 corregido, sin X)
    # ==========
    def m1_alpha(t_arr, A220, phi220, alpha_flow):
        return ringdown_model_GR_alpha(t_arr, A220, phi220, alpha_flow)

    p0_m1 = [A220_m0, phi220_m0, 0.0]
    bounds_m1 = ([-10 * abs(A220_m0), -2 * np.pi, -0.5],
                 [ 10 * abs(A220_m0),  2 * np.pi,  0.5])

    print("\n[Ajuste Modelo 1 (L1): GR + alpha_flow (220 corregido, sin X)]")
    popt_m1, pcov_m1 = curve_fit(
        m1_alpha, t_rel, h_norm, p0=p0_m1, bounds=bounds_m1, maxfev=20000
    )
    perr_m1 = np.sqrt(np.diag(pcov_m1))
    A220_m1, phi220_m1, alpha_m1 = popt_m1
    h1 = m1_alpha(t_rel, *popt_m1)

    # ==========
    # Modelo 2: GR + modo X (alpha_flow = 0)
    # ==========
    k_flow_example     = 1.0
    gamma_flow_example = 10.0

    def m2_grX(t_arr, A220, phi220, AX, phiX):
        return ringdown_model_GR_plus_X(
            t_arr, A220, phi220, AX, phiX,
            k_flow=k_flow_example, gamma_flow=gamma_flow_example
        )

    AX_0 = 0.1 * A220_m0
    phiX_0 = 0.0
    p0_m2 = [A220_m0, phi220_m0, AX_0, phiX_0]
    bounds_m2 = ([-10 * abs(A220_m0), -2 * np.pi, -10 * abs(A220_m0), -2 * np.pi],
                 [ 10 * abs(A220_m0),  2 * np.pi,  10 * abs(A220_m0),  2 * np.pi])

    print("\n[Ajuste Modelo 2 (L1): GR + modo X (alpha_flow = 0)]")
    popt_m2, pcov_m2 = curve_fit(
        m2_grX, t_rel, h_norm, p0=p0_m2, bounds=bounds_m2, maxfev=20000
    )
    perr_m2 = np.sqrt(np.diag(pcov_m2))
    A220_m2, phi220_m2, AX_m2, phiX_m2 = popt_m2
    h2 = m2_grX(t_rel, *popt_m2)

    # ==========
    # Modelo 3: EDR completo (220 corregido + X)
    # ==========
    def m3_edr(t_arr, A220, phi220, AX, phiX, alpha_flow):
        return ringdown_model_EDR_full(
            t_arr, A220, phi220, AX, phiX, alpha_flow,
            k_flow=k_flow_example, gamma_flow=gamma_flow_example
        )

    p0_m3 = [A220_m1, phi220_m1, AX_m2, phiX_m2, alpha_m1]
    bounds_m3 = ([-10 * abs(A220_m1), -2 * np.pi, -10 * abs(A220_m1), -2 * np.pi, -0.5],
                 [ 10 * abs(A220_m1),  2 * np.pi,  10 * abs(A220_m1),  2 * np.pi,  0.5])

    print("\n[Ajuste Modelo 3 (L1): EDR completo (220 corregido + X)]")
    popt_m3, pcov_m3 = curve_fit(
        m3_edr, t_rel, h_norm, p0=p0_m3, bounds=bounds_m3, maxfev=20000
    )
    perr_m3 = np.sqrt(np.diag(pcov_m3))
    A220_m3, phi220_m3, AX_m3, phiX_m3, alpha_m3 = popt_m3
    h3 = m3_edr(t_rel, *popt_m3)

    # 4. Estadísticos para cada modelo
    S0, chi2r0, AIC0, BIC0 = chi2_stats(h_norm, h0, n_params=2)
    S1, chi2r1, AIC1, BIC1 = chi2_stats(h_norm, h1, n_params=3)
    S2, chi2r2, AIC2, BIC2 = chi2_stats(h_norm, h2, n_params=4)
    S3, chi2r3, AIC3, BIC3 = chi2_stats(h_norm, h3, n_params=5)

    print("\n=== Parámetros clave (L1) ===")
    print(f"Modelo 1 (alpha_flow): alpha_flow = {alpha_m1:.3e} ± {perr_m1[2]:.3e}")
    print(f"Modelo 2 (modo X):     AX        = {AX_m2:.3e} ± {perr_m2[2]:.3e}")
    print(f"Modelo 3 (EDR full):   alpha_flow = {alpha_m3:.3e} ± {perr_m3[4]:.3e}, AX = {AX_m3:.3e} ± {perr_m3[2]:.3e}")

    print("\n=== Comparación de modelos (L1, ventana normalizada) ===")
    print(f"M0 GR       : S = {S0:.4e}, chi2_red = {chi2r0:.3f}, AIC = {AIC0:.2f}, BIC = {BIC0:.2f}")
    print(f"M1 GR+alpha : S = {S1:.4e}, chi2_red = {chi2r1:.3f}, AIC = {AIC1:.2f}, BIC = {BIC1:.2f}")
    print(f"M2 GR+X     : S = {S2:.4e}, chi2_red = {chi2r2:.3f}, AIC = {AIC2:.2f}, BIC = {BIC2:.2f}")
    print(f"M3 EDR full : S = {S3:.4e}, chi2_red = {chi2r3:.3f}, AIC = {AIC3:.2f}, BIC = {BIC3:.2f}")

    # 5. Figura (datos L1 + M0 + M3 + residuales)
    h0_phys = h0 * h_rms
    h3_phys = h3 * h_rms

    resid_m0 = h_norm - h0
    resid_m3 = h_norm - h3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(t_rel * 1000, h_rd,      label="Datos L1 (ringdown)", lw=1, color="C0")
    ax1.plot(t_rel * 1000, h0_phys,   label="Modelo 0: GR (220)",  lw=2, color="C1")
    ax1.plot(t_rel * 1000, h3_phys,   label="Modelo 3: EDR (220+X)", lw=2, color="C2")
    ax1.set_ylabel("Strain")
    ax1.set_title("GW150914 (L1): Datos vs GR y EDR")
    ax1.grid(True)
    ax1.legend()

    ax2.axhline(0.0, color="k", lw=1, ls="--")
    ax2.plot(t_rel * 1000, resid_m0, label="Datos - M0 (GR)",  lw=1, color="C1")
    ax2.plot(t_rel * 1000, resid_m3, label="Datos - M3 (EDR)", lw=1, color="C2")
    ax2.set_xlabel("Tiempo desde inicio de la ventana [ms]")
    ax2.set_ylabel("Residuo (norm.)")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    out_png = os.path.join(repo_root, "data", "gw150914_L1_edr_ringdown.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nFigura guardada en {out_png}")


if __name__ == "__main__":
    main()
