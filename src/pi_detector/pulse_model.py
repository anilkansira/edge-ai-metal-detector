from dataclasses import dataclass
from math import exp
import random
from .materials import MaterialProfile
from .soil import SoilProfile

@dataclass(frozen=True)
class PulseConfig:
    pulse_current_a: float = 2.8
    pulse_width_us: float = 160.0
    pulse_repetition_hz: float = 600.0
    coil_inductance_mh: float = 0.45
    coil_resistance_ohm: float = 1.20
    blanking_time_us: float = 25.0
    sample_times_us: tuple = (35, 55, 85, 130, 200, 320, 500, 750)
    adc_bits: int = 12
    preamp_noise: float = 0.010
    base_emi_noise: float = 0.010

def clamp(x, a, b):
    return max(a, min(b, x))

def quantize_adc(value, bits):
    value = clamp(value, 0.0, 1.0)
    levels = (2 ** bits) - 1
    return round(value * levels) / levels

def pulse_energy_factor(config: PulseConfig):
    energy = config.pulse_current_a * (config.pulse_width_us / 160.0)
    coil_factor = clamp(config.coil_inductance_mh / max(0.10, config.coil_resistance_ohm), 0.35, 1.35)
    return clamp(energy * coil_factor, 0.25, 1.60)

def simulate_ground_response(soil: SoilProfile, config: PulseConfig, randomize=True):
    mag, cond, moist = soil.magnetic_susceptibility, soil.electrical_conductivity, soil.moisture
    sal, iron = soil.salinity, soil.iron_oxide_level
    amp = 0.016 + mag*0.038 + cond*0.032 + moist*0.018 + sal*0.030 + iron*0.034 + soil.false_positive_risk*0.012
    amp *= 1.0 - soil.magnetite_compensation * 0.28
    amp *= pulse_energy_factor(config)
    out=[]
    for i,t_us in enumerate(config.sample_times_us):
        t_ms=t_us/1000.0
        v=amp*exp(-t_ms/max(0.04, soil.ground_decay_tau))
        v += (cond*0.010 + sal*0.013) * exp(-t_ms/0.90)
        v += (mag*0.013 + iron*0.016) * exp(-t_ms/0.36) * (1.0-soil.magnetite_compensation*0.30)
        if randomize:
            v += random.gauss(0, soil.noise_floor*(0.15-min(0.07,i*0.009)))
            v += random.gauss(0, config.base_emi_noise*0.18)
            v += random.gauss(0, config.preamp_noise*0.12)
        out.append(quantize_adc(v, config.adc_bits))
    return out

def simulate_target_response(material: MaterialProfile, soil: SoilProfile, config: PulseConfig, depth_m, horizontal_distance_m, coil_height_cm=7.0, sweep_quality=1.0, orientation_factor=1.0, size_factor=1.0, corrosion=0.0):
    cond, perm, ferro = material.conductivity, material.relative_permeability, material.ferrous_level
    proximity = exp(-((horizontal_distance_m/4.4)**2)*2.45)
    height_loss = exp(-(coil_height_cm/100.0)/0.17)
    effective_att = clamp(soil.attenuation + soil.magnetite_compensation*0.18, 0.48, 0.98)
    depth_loss = exp(-depth_m/max(0.18, 0.98*effective_att))
    material_gain = 0.40 + cond*0.42 + perm*0.24
    soil_penalty = 1.0 - soil.magnetic_susceptibility*0.065 - max(0.0, soil.moisture-0.55)*0.070 - soil.salinity*0.060 + soil.magnetite_compensation*0.10
    soil_penalty = clamp(soil_penalty, 0.56, 1.08)
    amp = proximity*depth_loss*height_loss*material_gain*clamp(orientation_factor,0.38,1.0)*(1.0-corrosion*0.18)*size_factor*soil_penalty*pulse_energy_factor(config)*(0.70+0.30*clamp(sweep_quality,0,1))
    tau = material.decay_tau_base*(1+cond*0.12)*(1-ferro*0.10)*(1-soil.magnetic_susceptibility*0.030)*(1+soil.magnetite_compensation*0.04)
    tau = clamp(tau, 0.05, 1.40)
    out=[]
    for t_us in config.sample_times_us:
        t_ms=max(1.0,t_us-config.blanking_time_us)/1000.0
        v = amp*exp(-t_ms/tau) + amp*ferro*0.12*exp(-t_ms/0.13) + amp*cond*0.10*exp(-t_ms/0.80)
        out.append(clamp(v, 0.0, 1.0))
    meta={'amplitude':amp,'tau':tau,'phase_proxy':cond*48.0-ferro*52.0+soil.magnetic_susceptibility*7.0,'ferrous_index':ferro,'conductivity_response':cond,'depth_m':depth_m}
    return out, meta

def combine_waveforms(*waveforms):
    if not waveforms: return []
    out=[0.0]*len(waveforms[0])
    for wf in waveforms:
        for i,v in enumerate(wf): out[i]+=v
    return [clamp(v,0.0,1.0) for v in out]
