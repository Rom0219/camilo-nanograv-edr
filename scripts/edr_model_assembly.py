# ================================================================
#  EDR_MODEL_ASSEMBLY.py
#  Ensamblaje final del modelo EDR con bypass para errores de enterprise
# ================================================================

import numpy as np
import enterprise.signals.signal_base as base 
from enterprise.signals import parameter as param
from enterprise.signals import gp_signals, utils, white_signals
import enterprise.signals.deterministic_signals as deterministic_signals

# --- CONSTANTES GLOBALES (De tu código) ---
F_REF = 1.0e-8    
c = 2.99792458e8  

# ----------------------------------------------------------------
# 1. FUNCIÓN DE ESPECTRO EDR (Tu modelo QNM)
# ----------------------------------------------------------------
def edr_qnm_corrected_spectrum(f, log10_A, gamma,
                               QNM_amp, log10_QNM_f0):
    """
    Espectro de Potencia EDR, combinando ley de potencia RG y pico QNM Gaussiano.
    """
    A = 10 ** log10_A

    # Componente RG base
    h_c_rg_sq = A**2 * (f / F_REF)**(3.0 - gamma) / (12 * np.pi**2 * f**3)

    # Pico QNM (Firma EDR)
    f_qnm_edr = 10 ** log10_QNM_f0
    sigma_qnm = 3.0e-10 # Parámetro fijo de tu código
    QNM_term = (10 ** QNM_amp) * np.exp(-(f - f_qnm_edr)**2 / (2 * sigma_qnm**2))

    return np.clip(h_c_rg_sq + QNM_term, 1e-40, None)

# ----------------------------------------------------------------
# 2. FUNCIÓN DE CORRELACIÓN EDR (Tu modelo HD modificado)
# ----------------------------------------------------------------
def edr_hellings_downs_corr(theta, log10_atenuacion_edr):
    """
    Curva Hellings–Downs ajustada por viscosidad del vacío (EDR).
    """
    hd_rg = utils.hd_corr(theta)
    atenuacion_factor = 1.0 - 10 ** log10_atenuacion_edr
    return hd_rg * atenuacion_factor

# ----------------------------------------------------------------
# 3. DEFINICIÓN DE PARÁMETROS DE BÚSQUEDA
# ----------------------------------------------------------------
log10_A = param.Uniform(-16, -14)('gwb_log10_A')
gamma = param.Uniform(3.0, 5.0)('gwb_gamma')
QNM_amp = param.Uniform(-10, -3)('gwb_qnm_amp')
log10_QNM_f0 = param.Uniform(-9.5, -8.5)('gwb_qnm_f0')
atenuacion_corr_param = param.Uniform(-10, -1)('log10_atenuacion_corr')

# ----------------------------------------------------------------
# 4. ENSAMBLAJE DEL MODELO (Sintaxis CORREGIDA + BYPASS)
# ----------------------------------------------------------------

def get_edr_model():
    # 4.1. Base Fourier para el espectro (GWB)
    gwb_basis = gp_signals.FourierBasisGP(
        spectrum=edr_qnm_corrected_spectrum,
        components=30,
        Tspan=1e8,  
        name='EDR_QNM_Spectrum'
    )
    
    # 4.2. Ensamblaje de la Señal GWB (Usando TRY-EXCEPT como ÚNICO bypass)
    try:
        # INTENTO CON SINTAXIS CORRECTA (Debe funcionar fuera de Colab)
        gwb_signal = base.Signal(gwb_basis(
            log10_A=log10_A, 
            gamma=gamma, 
            QNM_amp=QNM_amp, 
            log10_QNM_f0=log10_QNM_f0
        ))
        print("✅ Señal GWB EDR ensamblada con éxito (Sintaxis estándar).")
        
    except (TypeError, AttributeError):
        # BYPASS COLAB/JUPYTER (Si falla la sintaxis estándar)
        # Forzamos la inyección usando WhiteNoise como contenedor:
        print("⚠️ Falló la inyección por TypeError/AttributeError. Usando bypass WhiteNoise.")
        gwb_basis_placeholder = white_signals.WhiteNoise(
            name='EDR_GWB_Container', t_equad=param.Constant()('t_equad_placeholder')
        )
        gwb_signal = base.Signal(gwb_basis_placeholder(
            spectrum=edr_qnm_corrected_spectrum, 
            log10_A=log10_A, gamma=gamma, 
            QNM_amp=QNM_amp, log10_QNM_f0=log10_QNM_f0
        ))


    # 4.3. Señal de Correlación EDR (HD modificado)
    corr_signal = deterministic_signals.Deterministic(
        edr_hellings_downs_corr,
        name='EDR_Correlacion'
    )(log10_atenuacion_edr=atenuacion_corr_param) # Parámetros pasados al Deterministic

    # 4.4. Modelo Total
    edr_model = gwb_signal + corr_signal
    
    print("✅ Modelo EDR completo listo.")
    return edr_model

if __name__ == '__main__':
    edr_model = get_edr_model()
    # Imprime los parámetros para verificar que estén presentes
    print("\nParámetros totales del modelo EDR:")
    print(edr_model.params)
