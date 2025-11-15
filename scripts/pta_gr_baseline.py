#!/usr/bin/env python
"""
Análisis base NANOGrav 15-yr: modelo estándar GR
(HD + power-law GWB) usando enterprise + enterprise_extensions.

Este script:
- Carga un subconjunto de pulsars NG15yr desde data/ng15yr/par y data/ng15yr/tim
- Construye el modelo estándar GR (HD + power-law GWB) vía models.model_2a
- Lanza un MCMC corto de prueba para verificar que todo funciona
"""

import glob
import os
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise_extensions import models, model_utils

# Opcional: usa la_forge si está instalada para inspeccionar cadenas
try:
    from la_forge.core import Core
    HAVE_LAFORGE = True
except ImportError:
    HAVE_LAFORGE = False


def load_ng15_pulsars(par_dir="data/ng15yr/par",
                      tim_dir="data/ng15yr/tim",
                      max_psrs=5):
    """
    Carga pulsars NG15yr desde carpetas .par y .tim, usando solo max_psrs púlsares.

    Espera que:
    - par_dir contenga archivos *.par
    - tim_dir contenga archivos *.tim
    - Haya el mismo número de .par y .tim y que correspondan uno a uno.
    """
    par_files = sorted(glob.glob(os.path.join(par_dir, "*.par")))
    tim_files = sorted(glob.glob(os.path.join(tim_dir, "*.tim")))

    if len(par_files) == 0 or len(tim_files) == 0:
        raise RuntimeError(
            f"No se encontraron .par/.tim en {par_dir} y {tim_dir}. "
            "Copia ahí los archivos reales de NANOGrav 15-yr."
        )

    if len(par_files) != len(tim_files):
        raise RuntimeError(
            f"Número distinto de .par ({len(par_files)}) y .tim ({len(tim_files)}). "
            "Asegúrate de que corresponden uno a uno."
        )

    # Limitar al máximo deseado de púlsares para que corra rápido
    par_files = par_files[:max_psrs]
    tim_files = tim_files[:max_psrs]

    psrs = []
    for par, tim in zip(par_files, tim_files):
        print(f"Cargando pulsar: {os.path.basename(par)} / {os.path.basename(tim)}")
        # Ephemeris típica de NG15yr; ajusta si hace falta
        psrs.append(Pulsar(par, tim, ephem="DE440"))

    print(f"Total pulsars cargados: {len(psrs)}")
    return psrs


def build_gr_model(psrs):
    """
    Modelo estándar NANOGrav:
    - ruido individual por pulsar
    - fondo GWB isótropo con correlaciones Hellings–Downs (HD)
    - espectro de potencia simple (power-law) para el GWB
    """
    pta = models.model_2a(
        psrs,
        psd="powerlaw",          # ley de potencia para el GWB
        noisedict=None,          # puedes pasar un dict de ruido si lo tienes
        components=["gw"],       # solo fondo GWB
        upper_limit=False,
    )
    return pta


def run_mcmc(pta, outdir="chains/gr_baseline", nsteps=500):
    """
    Lanza un MCMC corto de prueba.

    nsteps se mantiene pequeño para que el script sea rápido en Codespaces.
    Para análisis reales, sube nsteps (p. ej. 1e5 o más).
    """
    os.makedirs(outdir, exist_ok=True)

    sampler = model_utils.setup_sampler(
        pta,
        outdir=outdir,
        resume=False,
    )

    # Punto inicial: muestrea todos los parámetros una vez
    x0 = np.hstack([p.sample() for p in pta.params])

    print(f"Iniciando MCMC con {nsteps} pasos...")
    sampler.sample(
        x0,
        nsteps,
        SCAMweight=30,
        AMweight=15,
        DEweight=50,
    )
    print("MCMC terminado.")

    # Si tienes la_forge, crea un Core para inspeccionar las cadenas
    if HAVE_LAFORGE:
        print("Creando objeto Core de la_forge...")
        core = Core(label="gr_baseline", outdir=outdir)
        print(core)
        print("la_forge Core creado.")


def main():
    # Cambia max_psrs si quieres usar más púlsares
    psrs = load_ng15_pulsars(max_psrs=5)
    pta = build_gr_model(psrs)
    run_mcmc(pta)


if __name__ == "__main__":
    main()
