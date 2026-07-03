from dataclasses import dataclass

@dataclass(frozen=True)
class MaterialProfile:
    name: str
    conductivity: float
    relative_permeability: float
    ferrous_level: float
    decay_tau_base: float

MATERIALS = {
    'Iron': MaterialProfile('Iron', 0.35, 0.95, 0.95, 0.38),
    'Aluminum': MaterialProfile('Aluminum', 0.65, 0.05, 0.05, 0.55),
    'Copper': MaterialProfile('Copper', 1.00, 0.02, 0.02, 0.88),
    'Brass': MaterialProfile('Brass', 0.78, 0.03, 0.03, 0.72),
    'Steel': MaterialProfile('Steel', 0.48, 0.75, 0.75, 0.46),
}
