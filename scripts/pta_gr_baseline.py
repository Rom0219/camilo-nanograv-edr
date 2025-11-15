#!/usr/bin/env python
"""
Análisis base NANOGrav 15-yr: modelo estándar GR
(HD + power-law GWB) usando enterprise + enterprise_extensions.
"""

import glob
import os
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise_extensions import models, model_utils


def load_ng15_pulsars(par_dir="data/ng15yr/par", tim_dir="data/ng15yr/tim"):
    """Carga pulsars NG15yr desde carpetas .par y .tim."""
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

    psrs = []
    for par, tim in zip(par_files, tim_files):
        print(f"Cargando pulsar: {os.path.basename(par)} / {os.path.basename(tim)}")
        psrs.append(Pulsar(par, tim, ephem="DE440"))

    print(f"Total pulsars cargados: {len(psrs)}")
    return psrs


def build_gr_model(psrs):
    """
    Modelo estándar NANOGrav:
    - ruido por pulsar
    - fondo GWB isótropo con HD
    - espectro de potencia power-law
    """
    pta = models.model_2a(
        psrs,
        psd="powerlaw",
        noisedict=None,
        components=["gw"],
        gamma_gw_prior="uniform",
        upper_limit=False,
    )
    return pta


def run_mcmc(pta, outdir="chains/gr_baseline", nsteps=2000):
    """MCMC corto de prueba (2000 pasos) solo para verificar que todo corre."""
    os.makedirs(outdir, exist_ok=True)

    sampler = model_utils.setup_sampler(
        pta,
        outdir=outdir,
        resume=False,
    )

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


def main():
    psrs = load_ng15_pulsars()
    pta = build_gr_model(psrs)
    run_mcmc(pta)


if __name__ == "__main__":
    main()
