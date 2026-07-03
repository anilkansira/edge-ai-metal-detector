from dataclasses import dataclass

@dataclass(frozen=True)
class SoilProfile:
    name: str
    magnetic_susceptibility: float
    electrical_conductivity: float
    moisture: float
    salinity: float
    iron_oxide_level: float
    ground_decay_tau: float
    noise_floor: float
    attenuation: float
    false_positive_risk: float
    magnetite_compensation: float = 0.0

SOILS = {
    'Dry Sandy Soil': SoilProfile('Dry Sandy Soil', 0.12, 0.12, 0.18, 0.03, 0.10, 0.20, 0.018, 0.96, 0.12),
    'Normal Soil': SoilProfile('Normal Soil', 0.30, 0.32, 0.38, 0.06, 0.25, 0.30, 0.030, 0.88, 0.22),
    'Wet Clay Soil': SoilProfile('Wet Clay Soil', 0.45, 0.75, 0.82, 0.10, 0.35, 0.44, 0.052, 0.74, 0.40, 0.05),
    'Highly Mineralized Soil': SoilProfile('Highly Mineralized Soil', 0.78, 0.60, 0.36, 0.10, 0.78, 0.40, 0.064, 0.66, 0.62, 0.20),
    'Salty Conductive Soil': SoilProfile('Salty Conductive Soil', 0.28, 0.82, 0.70, 0.78, 0.20, 0.58, 0.060, 0.68, 0.55, 0.03),
    'Black Sand / Magnetite Soil': SoilProfile('Black Sand / Magnetite Soil', 0.86, 0.34, 0.28, 0.08, 0.88, 0.43, 0.066, 0.72, 0.62, 0.34),
}
