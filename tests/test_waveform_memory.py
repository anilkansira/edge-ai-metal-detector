import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector.waveform_memory import VerifiedWaveformMemory

def test_waveform_memory_match():
    mem=VerifiedWaveformMemory(['Iron','Copper','Ground Noise'])
    s=[0.0]*38; s[15:23]=[0.4,0.35,0.30,0.22,0.16,0.10,0.06,0.03]
    mem.add_sample(s,'Copper')
    match=mem.match(s, lambda x:x)
    assert match.best_label=='Copper'
    assert match.strength>0
