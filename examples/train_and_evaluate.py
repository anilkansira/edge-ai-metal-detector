import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector import PulseInductionPipeline

def main():
    pipeline=PulseInductionPipeline()
    acc=pipeline.train(samples_per_material=700, noise_samples=700, seed=42)
    print('Training completed')
    print(f'Validation accuracy: {acc*100:.2f}%')
if __name__=='__main__': main()
