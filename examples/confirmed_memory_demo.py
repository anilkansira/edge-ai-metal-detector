import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector import PulseInductionPipeline

def show(title,result):
    print(title); print(f'  Prediction : {result.predicted_class}'); print(f'  Confidence : {result.confidence*100:.2f}%'); print(f'  Memory     : {result.memory_match.best_label}, similarity={result.memory_match.best_similarity:.3f}, strength={result.memory_match.strength:.3f}')
    for n,p in sorted(result.probabilities.items(), key=lambda i:i[1], reverse=True)[:5]: print(f'    {n:<14}: {p*100:6.2f}%')
    print()

def main():
    pipeline=PulseInductionPipeline(); pipeline.train(samples_per_material=450, noise_samples=450)
    material='Copper'; soil='Black Sand / Magnetite Soil'; depth=0.70
    before=pipeline.simulate_and_predict(material,soil,depth,0.25); show('Before verified waveform memory:',before)
    pipeline.add_verified_sample(before.feature, material)
    after=pipeline.simulate_and_predict(material,soil,depth,0.25); show('After adding verified Copper waveform:',after)
if __name__=='__main__': main()
