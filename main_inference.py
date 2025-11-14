# ================================================================
#  MAIN_INFERENCE.py: ENSAMBLAJE DE MODELOS EDR Y RG
#  Repositorio: Rom0219/camilo-nanograv-edr
#  Objetivo: Preparar M_EDR y M_RG para el cálculo del Factor de Bayes.
# ================================================================

# --- 1. CONFIGURACIÓN E INSTALACIÓN (Ejecutar en el terminal del Codespace) ---
# Ejecuta en el terminal:
# pip install enterprise-pulsar enterprise_extensions PTMCMCSampler

import numpy as np
import enterprise.signals.signal_base as base 
from enterprise.pulsar import Pulsar
from enterprise.signals import parameter as param
from enterprise.signals import gp_signals, deterministic_signals, utils
from enterprise.signals import white_signals # Necesario para compatibilidad
import glob
import os
# from PTMCMCSampler.PTMCMCSampler import PTSampler # Descomentar para la inferencia real

# --- CONSTANTES FÍSICAS ---
F_REF = 1.0e-8    
H0 = 70e3 / 3.086e22 # Constante de Hubble [s^-1]

# ----------------------------------------------------------------
# 2. FUNCIONES FÍSICAS (TU TEORÍA EDR)
# ----------------------------------------------------------------

# PARÁMETROS GLOBALES DE BÚSQUEDA (Necesarios para la inferencia)
log10_A = param.Uniform(-16, -14)('gwb_log10_A')
gamma = param.Uniform(3.0, 5.0)('gwb_gamma')
QNM_amp = param.Uniform(-10, -3)('gwb_qnm_amp')
log10_QNM_f0 = param.Uniform(-9.5, -8.5)('gwb_qnm_f0')
atenuacion_corr_param = param.Uniform(-10, -1)('log10_atenuacion_corr')

# FUNCIÓN 2.1: ESPECTRO EDR/QNM
def edr_qnm_corrected_spectrum(f, log10_A, gamma, QNM_amp, log10_QNM_f0):
    """
    Espectro de Potencia EDR, combinando ley de potencia RG y pico QNM Gaussiano.
    """
    A = 10 ** log10_A
    h_c_rg_sq = A**2 * (f / F_REF)**(3.0 - gamma) / (12 * np.pi**2 * f**3)

    f_qnm_edr = 10 ** log10_QNM_f0
    sigma_qnm = 3.0e-10 # Amortiguamiento fijo
    QNM_term = (10 ** QNM_amp) * np.exp(-(f - f_qnm_edr)**2 / (2 * sigma_qnm**2))

    return np.clip(h_c_rg_sq + QNM_term, 1e-40, None)

# FUNCIÓN 2.2: CORRELACIÓN EDR (HD MODIFICADO)
def edr_hellings_downs_corr(theta, log10_atenuacion_edr):
    """
    Curva Hellings–Downs ajustada por viscosidad del vacío (EDR).
    """
    hd_rg = utils.hd_corr(theta)
    atenuacion_factor = 1.0 - 10 ** log10_atenuacion_edr
    return hd_rg * atenuacion_factor

# ----------------------------------------------------------------
# 3. ENSAMBLAJE DE MODELOS (M_RG Y M_EDR)
# ----------------------------------------------------------------

def assemble_models():
    """Ensambla el Modelo Nulo (RG) y el Modelo de Prueba (EDR)."""
    
    # --- 3.1. Componentes Comunes ---
    # Ruido Galáctico (GNN) – Se recomienda para un análisis completo
    gamma_gnn = param.Uniform(3.5, 5.5)('gnn_gamma')
    log10_A_gnn = param.Uniform(-16, -12)('gnn_log10_A')
    gnn_basis = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw, components=30, Tspan=1e8, name='GNN_Noise'
    )
    gnn_signal = base.Signal(gnn_basis(
        log10_A=log10_A_gnn, gamma=gamma_gnn
    ))
    
    # Correlación Hellings-Downs (Necesaria en ambos modelos como prueba geométrica)
    corr_basis_std = gp_signals.HellingsDowns(name='HD_Correlacion_RG')()
    
    # -----------------------------------------------------------
    # A. MODELO NULO: M_RG (RG Pura + HD Estándar)
    # -----------------------------------------------------------
    gwb_basis_rg = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw, components=30, Tspan=1e8, name='RG_GWB'
    )
    gwb_signal_rg = base.Signal(gwb_basis_rg(
        log10_A=log10_A, 
        gamma=param.Constant(13/3.0)('gwb_gamma_rg') # Gamma fijo por RG
    ))

    # M_RG: GWB RG (PowerLaw) + GNN + HD Estándar
    model_rg = gwb_signal_rg + gnn_signal + corr_basis_std
    print("✅ Modelo Nulo RG (PowerLaw) ensamblado.")
    
    # -----------------------------------------------------------
    # B. MODELO DE PRUEBA: M_EDR (Tu Espectro EDR/QNM + Tu Correlación EDR Modificada)
    # -----------------------------------------------------------
    
    # 1. Base Fourier para el espectro EDR
    gwb_basis_edr = gp_signals.FourierBasisGP(
        spectrum=edr_qnm_corrected_spectrum,
        components=30,
        Tspan=1e8,  
        name='EDR_QNM_Spectrum'
    )
    
    # 2. Ensamblaje de la Señal GWB (Usando sintaxis de inyección forzada)
    gwb_signal_base = base.Signal(gwb_basis_edr)
    params_edr_gwb = {
        'gwb_log10_A': log10_A, 
        'gwb_gamma': gamma, 
        'gwb_qnm_amp': QNM_amp, 
        'gwb_qnm_f0': log10_QNM_f0 
    }
    gwb_signal_edr = gwb_signal_base(**params_edr_gwb) # ¡Inyección forzada de parámetros!

    # 3. Señal de Correlación EDR (Deterministic)
    corr_signal_edr = deterministic_signals.Deterministic(
        edr_hellings_downs_corr,
        name='EDR_Correlacion'
    )(log10_atenuacion_edr=atenuacion_corr_param) 

    # M_EDR: GWB EDR/QNM + GNN + HD Modificado (Tu Correlación)
    model_edr = gwb_signal_edr + gnn_signal + corr_signal_edr
    print("✅ Modelo EDR (QNM y Atenuación) ensamblado.")
    
    return model_rg, model_edr

# ----------------------------------------------------------------
# 4. FUNCIÓN DE CARGA DE DATOS (ESQUELÉTICA)
# ----------------------------------------------------------------

def load_nanograv_pulsars(data_dir='nanograv_15yr_data'):
    """Carga los objetos Pulsar desde los archivos .par y .tim."""
    if not os.path.isdir(data_dir):
        print(f"\n¡ALERTA!: Carpeta de datos '{data_dir}' no encontrada.")
        print("Antes de la inferencia, debes crear esta carpeta y añadir los archivos .tim y .par de NANOGrav 15-yr.")
        return []

    parfiles = sorted(glob.glob(os.path.join(data_dir, '*.par')))
    timfiles = sorted(glob.glob(os.path.join(data_dir, '*.tim')))
    
    psrs = []
    # Aquí iría el bucle de carga de Pulsar(p, t)
    # psrs = [Pulsar(p, t) for p, t in zip(parfiles, timfiles)]
    
    print(f"Púlsares listos para cargar: {len(parfiles)}") 
    return psrs

# ----------------------------------------------------------------
# 5. EJECUCIÓN PRINCIPAL
# ----------------------------------------------------------------

if __name__ == '__main__':
    
    # 5.1. Cargar Púlsares (Solo verifica si están los archivos)
    psrs = load_nanograv_pulsars()
    
    # 5.2. Ensamblar los modelos
    model_rg, model_edr = assemble_models()
    
    print("\n========================================================")
    print("  VERIFICACIÓN DE ENSAMBLAJE (Codespaces)")
    print("========================================================")
    print("Parámetros totales del Modelo EDR:")
    # Imprimir los nombres de los parámetros para confirmar que se inyectaron
    print([p.name for p in model_edr.param_names]) 
    
    # El siguiente paso sería inicializar el Likelihood y el PTSampler aquí.
    # likelihood_rg = model_rg.get_likelihood(psrs)
    # sampler_rg = PTSampler(...) 
    # sampler_rg.sample(N_steps)
    
    print("\n✅ ¡El Código está listo para la Inferencia Bayesiana con Sampler!")
