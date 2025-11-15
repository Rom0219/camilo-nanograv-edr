# scripts/gw150914_edr_windows_H1.py
"""
Prueba la robustez de alpha_flow frente a la elección de ventana de ringdown,
usando GW150914 (H1) y el modelo M1: GR + alpha_flow (modo 220 corregido, sin modo X).

Para cada ventana definida detrás del pico:
  [t_start_offset, t_end_offset] (en segundos)

calcula:
  - alpha_flow ± error
  - S, chi2_red, AIC, BIC

y escribe los resultados en la terminal.
"""

import os
import h5py
import numpy as np
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


def qnm_EDR_parameters(alpha_flow: float = 0.0,
                       beta_flow: float = 0.0,
                       k_flow: float = 1.0,
                       gamma_flow: float = 10.0) -> tuple[float, float, float, float]:
    """Frecuencias (Hz) y tiempos para 220^EDR y modo X (aunque aquí solo usamos 220)."""
    f220_GR, tau220_GR = qnm_220_GR(M_FINAL_SOLAR, A_FINAL)

    f220_EDR   = f220_GR   * (1.0 + alpha_flow)
    tau220_EDR = tau220_GR * (1.0 + beta_flow)

    omega_H = 0.0  # no se necesita aquí
    fX_Hz   = 0.0
    M_geom  = geom_time_from_mass_solar(M_FINAL_SOLAR)
    tauX_s  = gamma_flow * M_geom

    return f220_EDR, tau220_EDR, fX_Hz, tauX_s


def ringdown_model_GR_alpha(t: np.ndarray,
                            A220: float, phi220: float,
                            alpha_flow: float) -> np.ndarray:
    """Modelo M1: 220 corregido con alpha_flow, sin modo X."""
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
    """Devuelve S, chi2_red, AIC y BIC (sigma = 1, datos normalizados)."""
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
    # Ruta de datos H1
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")
    h1_file = os.path.join(data_dir, "H-H1_LOSC_4_V1-1126259446-32.hdf5")

    if not os.path.exists(h1_file):
        raise FileNotFoundError(
            f"No se encontró {h1_file}. Ejecuta primero scripts/download_data.py"
        )

    # Cargar señal H1 completa
    t, h = load_strain_from_hdf5(h1_file)
    h = h - np.mean(h)

    # Pico global
    idx_peak = np.argmax(np.abs(h))
    t_peak = t[idx_peak]

    print(f"Pico global en t = {t_peak:.6f} s (índice {idx_peak})")

    # Definir ventanas detrás del pico (offsets en segundos)
    windows = [
        (0.001, 0.030),  # 1 ms a 30 ms
        (0.002, 0.030),  # 2 ms a 30 ms (ventana original)
        (0.003, 0.030),  # 3 ms a 30 ms
        (0.004, 0.030),  # 4 ms a 30 ms
        (0.002, 0.025),  # 2 ms a 25 ms
    ]

    print("\nVentanas a probar (offsets en ms):")
    for i, (dt_start, dt_end) in enumerate(windows):
        print(f"  W{i}: [{dt_start*1e3:.1f}, {dt_end*1e3:.1f}] ms")

    print("\nResultados (modelo M1: GR + alpha_flow, sin modo X):")
    print("idx | t_start_ms t_end_ms | alpha_flow ± err | S      chi2_red  AIC     BIC")

    for i, (dt_start, dt_end) in enumerate(windows):
        # 1. Construir máscara de ventana
        t_start = t_peak + dt_start
        t_end   = t_peak + dt_end

        mask = (t >= t_start) & (t <= t_end)
        t_rd = t[mask]
        h_rd = h[mask]

        if len(t_rd) < 20:
            print(f"{i:3d} | {dt_start*1e3:8.3f} {dt_end*1e3:8.3f} | ventana muy corta ({len(t_rd)} puntos)")
            continue

        t_rel = t_rd - t_rd[0]

        # 2. Normalizar
        h_rms = np.sqrt(np.mean(h_rd**2))
        h_norm = h_rd / h_rms if h_rms != 0 else h_rd

        # 3. Ajuste M1
        def m1_alpha(t_arr, A220, phi220, alpha_flow):
            return ringdown_model_GR_alpha(t_arr, A220, phi220, alpha_flow)

        A220_0 = np.max(h_norm)
        phi220_0 = 0.0
        p0_m1 = [A220_0, phi220_0, 0.0]
        bounds_m1 = ([-10 * abs(A220_0), -2 * np.pi, -0.5],
                     [ 10 * abs(A220_0),  2 * np.pi,  0.5])

        try:
            popt_m1, pcov_m1 = curve_fit(
                m1_alpha, t_rel, h_norm,
                p0=p0_m1, bounds=bounds_m1, maxfev=20000
            )
            perr_m1 = np.sqrt(np.diag(pcov_m1))
        except Exception as e:
            print(f"{i:3d} | {dt_start*1e3:8.3f} {dt_end*1e3:8.3f} | error en ajuste: {e}")
            continue

        A220_fit, phi220_fit, alpha_fit = popt_m1
        alpha_err = perr_m1[2]

        h_model = m1_alpha(t_rel, *popt_m1)
        S, chi2_red, AIC, BIC = chi2_stats(h_norm, h_model, n_params=3)

        print(f"{i:3d} | {dt_start*1e3:8.3f} {dt_end*1e3:8.3f} | "
              f"{alpha_fit: .3e} ± {alpha_err: .3e} | "
              f"{S:6.1f}  {chi2_red:7.3f}  {AIC:6.2f}  {BIC:6.2f}")


if __name__ == "__main__":
    main()

