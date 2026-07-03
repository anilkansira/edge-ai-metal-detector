import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector.feature_extraction import extract_waveform_features
from pi_detector.pulse_model import PulseConfig
from pi_detector.soil import SOILS

def test_feature_vector_length():
    soil=SOILS['Normal Soil']; cfg=PulseConfig()
    f=extract_waveform_features([0.2,0.18,0.15,0.11,0.08,0.05,0.03,0.02], cfg, 10.0, 0.2, 0.7, 0.5, {'magnetic_susceptibility':soil.magnetic_susceptibility,'electrical_conductivity':soil.electrical_conductivity,'moisture':soil.moisture,'salinity':soil.salinity,'iron_oxide_level':soil.iron_oxide_level,'noise_floor':soil.noise_floor}, 0.02)
    assert len(f.values)==38
