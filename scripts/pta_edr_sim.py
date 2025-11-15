# scripts/pta_edr_sim.py
"""
Simulación simple de un Pulsar Timing Array (PTA) para comparar
GR (curva de Hellings–Downs) con una deformación tipo EDR.

- Se generan N_puls pulsars en el cielo con posiciones aleatorias.
- Se calculan las separaciones angulares zeta_ij entre todas las parejas.
- Se construye una correlación "verdadera" para cada pareja:
      C_true(ζ) = A_true * [ HD(ζ) + delta_true * (ζ/π - 0.5) ]
  donde HD(ζ) es la curva estándar de Hellings–Downs.
- Se añaden errores gaussianos para simular medidas con incertidumbre.
- Se ajustan dos modelos:
    M0 (GR)   : C(ζ) = A * HD(ζ)
    M1 (EDR)  : C(ζ) = A * [ HD(ζ) + delta * (ζ/π - 0.5) ]
- Se comparan S, chi2_red, AIC y BIC para ambos modelos, y se genera
  una figura con las correlaciones simuladas y las curvas mejor ajustadas.

Esto es el análogo PTA de los modelos GR vs EDR que usaste en los ringdowns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def hellings_downs(zeta: np.ndarray) -> np.ndarray:
    """
    Curva de Hellings–Downs estándar (normalizada a 1 en ζ=0).

    Fórmula típica:
      Γ(ζ) = 0.5 + (3/2) * x * ln(x) - 0.25 * x,   donde x = (1 - cos ζ) / 2

    Luego se normaliza para que Γ(0) = 1.
    """
    cosz = np.cos(zeta)
    x = (1.0 - cosz) / 2.0
    # Evitar problemas numéricos en x=0
    x = np.clip(x, 1e-12, 1.0)
    gamma = 0.5 + 1.5 * x * np.log(x) - 0.25 * x
    # Normalizar a Γ(0) = 1
    gamma0 = 0.5 + 1.5 * (0.0) * np.log(1e-12) - 0.25 * 0.0
    gamma_norm = gamma / gamma0
    return gamma_norm


def chi2_stats(y: np.ndarray, y_model: np.ndarray, n_params: int) -> tuple[float, float, float, float]:
    """
    S, chi2_red, AIC y BIC con errores gaussianos homogéneos (sigma absorbida en S).
    """
    n = len(y)
    resid = y - y_model
    S = float(np.sum(resid**2))
    chi2 = S
    dof = n - n_params
    chi2_red = chi2 / dof if dof > 0 else np.nan
    AIC = 2 * n_params + n * np.log(S / n)
    BIC = n_params * np.log(n) + n * np.log(S / n)
    return S, chi2_red, AIC, BIC


def main() -> None:
    # Parámetros de la simulación
    N_puls = 20
    A_true = 1.0       # amplitud verdadera del fondo
    delta_true = -0.3  # deformación EDR verdadera (análogo de alpha_flow)

    sigma_noise = 0.2  # dispersión de ruido en las correlaciones

    rng = np.random.default_rng(12345)

    # 1. Generar posiciones aleatorias en la esfera para los púlsares
    #    (usamos distribución isotrópica)
    cos_theta = rng.uniform(-1.0, 1.0, size=N_puls)
    theta = np.arccos(cos_theta)           # colatitud
    phi = rng.uniform(0.0, 2.0*np.pi, size=N_puls)  # longitud

    # Direcciones unitarias
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    n_vecs = np.vstack([nx, ny, nz]).T

    # 2. Calcular separaciones angulares zeta_ij para todas las parejas i<j
    zetas = []
    C_true = []
    C_err  = []

    for i in range(N_puls):
        for j in range(i+1, N_puls):
            ni = n_vecs[i]
            nj = n_vecs[j]
            cos_zeta = np.clip(np.dot(ni, nj), -1.0, 1.0)
            zeta = np.arccos(cos_zeta)
            zetas.append(zeta)

            # Correlación "verdadera" GR+EDR
            hd = hellings_downs(np.array([zeta]))[0]
            deform = (zeta / np.pi) - 0.5
            C_ij = A_true * (hd + delta_true * deform)

            # Añadir ruido gaussiano
            noise = rng.normal(0.0, sigma_noise)
            C_obs = C_ij + noise

            C_true.append(C_obs)
            C_err.append(sigma_noise)

    zetas = np.array(zetas)
    C_true = np.array(C_true)
    C_err = np.array(C_err)

    # Ordenar por ángulo para graficar más claro
    order = np.argsort(zetas)
    zetas = zetas[order]
    C_true = C_true[order]
    C_err = C_err[order]

    # 3. Ajuste Modelo 0: GR puro (C = A * HD)
    hd_all = hellings_downs(zetas)
    # Ajuste lineal simple en mínimos cuadrados con errores constantes
    # C_obs ≈ A * HD => A_fit = sum(HD*C) / sum(HD^2)
    A_m0 = np.sum(hd_all * C_true) / np.sum(hd_all**2)
    C_m0 = A_m0 * hd_all

    S0, chi2r0, AIC0, BIC0 = chi2_stats(C_true, C_m0, n_params=1)

    # 4. Ajuste Modelo 1: GR + deformación EDR (C = A * [HD + delta*(ζ/π - 0.5)])
    deform_all = (zetas / np.pi) - 0.5

    # Ajuste lineal en parámetros (A, A*delta) reescribiendo:
    # C = A * HD + (A*delta) * deform
    # Definimos p1 = A, p2 = A*delta.
    X = np.vstack([hd_all, deform_all]).T  # diseño n×2
    # Solución de mínimos cuadrados (sin pesos o con pesos iguales)
    # p = (X^T X)^(-1) X^T y
    XT_X = X.T @ X
    XT_y = X.T @ C_true
    p = np.linalg.solve(XT_X, XT_y)
    A_m1 = p[0]
    A_delta = p[1]
    delta_m1 = A_delta / A_m1 if A_m1 != 0 else 0.0

    C_m1 = A_m1 * (hd_all + delta_m1 * deform_all)

    S1, chi2r1, AIC1, BIC1 = chi2_stats(C_true, C_m1, n_params=2)

    print("=== Parámetros simulación PTA (verdaderos) ===")
    print(f"A_true      = {A_true:.3f}")
    print(f"delta_true  = {delta_true:.3f}")
    print(f"sigma_noise = {sigma_noise:.3f}")

    print("\n=== Ajuste Modelo 0 (GR puro: C = A*HD) ===")
    print(f"A_fit       = {A_m0:.3f}")
    print(f"S0          = {S0:.4e}, chi2_red = {chi2r0:.3f}, AIC = {AIC0:.2f}, BIC = {BIC0:.2f}")

    print("\n=== Ajuste Modelo 1 (GR+EDR: C = A[HD + delta*(ζ/π-0.5)]) ===")
    print(f"A_fit       = {A_m1:.3f}")
    print(f"delta_fit   = {delta_m1:.3f}")
    print(f"S1          = {S1:.4e}, chi2_red = {chi2r1:.3f}, AIC = {AIC1:.2f}, BIC = {BIC1:.2f}")

    # 5. Figura comparativa
    fig, ax = plt.subplots(figsize=(8, 5))

    zeta_deg = zetas * 180.0 / np.pi

    ax.errorbar(zeta_deg, C_true, yerr=C_err, fmt="o", color="k",
                ms=4, alpha=0.7, label="Datos simulados")

    zeta_grid = np.linspace(0.0, np.pi, 200)
    hd_grid = hellings_downs(zeta_grid)
    deform_grid = (zeta_grid / np.pi) - 0.5

    C_gr_grid = A_m0 * hd_grid
    C_edr_grid = A_m1 * (hd_grid + delta_m1 * deform_grid)

    ax.plot(zeta_grid * 180.0 / np.pi, C_gr_grid, label="Modelo 0: GR (HD)", color="C1", lw=2)
    ax.plot(zeta_grid * 180.0 / np.pi, C_edr_grid, label="Modelo 1: GR+EDR", color="C2", lw=2)

    ax.set_xlabel("Separación angular ζ [grados]")
    ax.set_ylabel("Correlación normalizada")
    ax.set_title("PTA simulado: GR vs GR+EDR en correlaciones tipo Hellings–Downs")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    repo_root = os.path.dirname(os.path.dirname(__file__))
    out_png = os.path.join(repo_root, "data", "pta_edr_sim.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nFigura guardada en {out_png}")


if __name__ == "__main__":
    main()
