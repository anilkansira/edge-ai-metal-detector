from dataclasses import dataclass
from math import log
from .pulse_model import PulseConfig, clamp

FEATURE_NAMES = ['signal','phase_proxy','tau_estimate','ferrous_index','conductivity_response','depth_m','signal_depth_ratio','nonferrous_score','soil_magnetic_susceptibility','soil_electrical_conductivity','soil_moisture','soil_salinity','soil_iron_oxide','soil_noise','soil_difficulty','sample_1','sample_2','sample_3','sample_4','sample_5','sample_6','sample_7','sample_8','decay_area','decay_slope','early_late_ratio','snr_norm','sweep_quality','signal_stability','mixed_ratio','ground_balance_error','coil_height_norm','emi_noise','electronics_noise','battery_norm','pulse_width_norm','pulse_current_norm','blanking_time_norm']

@dataclass(frozen=True)
class FeatureVector:
    values: list[float]
    names: list[str]

def estimate_tau(samples, config: PulseConfig):
    if len(samples)<2: return 0.0
    s0=max(samples[0],1e-4); sl=max(samples[-1],1e-4)
    if s0 <= sl: return 1.20
    t0=config.sample_times_us[0]/1000.0; tl=config.sample_times_us[-1]/1000.0
    return clamp(-(tl-t0)/log(sl/s0),0.0,1.50)

def extract_waveform_features(target_samples, config: PulseConfig, phase_proxy, ferrous_index, conductivity_response, depth_m, soil_features, ground_balance_error, sweep_quality=1.0, signal_stability=1.0, mixed_ratio=0.0, coil_height_cm=7.0, emi_noise=0.010, electronics_noise=0.010, battery_percent=100.0):
    s=list(target_samples)
    while len(s)<8: s.append(0.0)
    s=s[:8]
    signal=clamp(sum(s)/len(s)*2.4,0.0,1.0)
    area=sum(s)/len(s); slope=s[0]-s[-1]
    early=sum(s[:3])/3.0; late=sum(s[-3:])/3.0
    ratio=early/(late+0.025)
    tau=estimate_tau(s, config)
    soil_noise=soil_features['noise_floor']
    snr=signal/(soil_noise+electronics_noise+emi_noise+0.015)
    snr_norm=clamp(snr/12.0,0.0,1.0)
    soil_difficulty=soil_features['magnetic_susceptibility']*0.24 + soil_features['electrical_conductivity']*0.20 + soil_features['salinity']*0.18 + soil_features['iron_oxide_level']*0.18 + soil_noise*2.40
    values=[signal,phase_proxy,tau,clamp(ferrous_index,0,1),clamp(conductivity_response,0,1.2),clamp(depth_m,0,2.5),signal/(depth_m+0.20),conductivity_response*(1.0-ferrous_index),soil_features['magnetic_susceptibility'],soil_features['electrical_conductivity'],soil_features['moisture'],soil_features['salinity'],soil_features['iron_oxide_level'],soil_noise,soil_difficulty,*s,area,slope,ratio,snr_norm,clamp(sweep_quality,0,1),clamp(signal_stability,0,1),clamp(mixed_ratio,0,1),clamp(ground_balance_error,0,1),clamp(coil_height_cm/20.0,0,1),emi_noise,electronics_noise,clamp(battery_percent/100.0,0,1),clamp(config.pulse_width_us/300.0,0,1),clamp(config.pulse_current_a/5.0,0,1),clamp(config.blanking_time_us/80.0,0,1)]
    return FeatureVector(values=values, names=FEATURE_NAMES)
