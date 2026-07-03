import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector import PulseInductionPipeline

def main():
    pipeline=PulseInductionPipeline()
    print('Training classifier...')
    acc=pipeline.train(samples_per_material=450, noise_samples=450)
    print(f'Validation accuracy: {acc*100:.2f}%\n')
    material='Copper'; soil='Black Sand / Magnetite Soil'; depth=0.65
    result=pipeline.simulate_and_predict(material,soil,depth,0.20)
    print('Input target:')
    print(f'  Material : {material}')
    print(f'  Soil     : {soil}')
    print(f'  Depth    : {depth:.2f} m\n')
    print('Prediction:')
    print(f'  Class      : {result.predicted_class}')
    print(f'  Confidence : {result.confidence*100:.2f}%')
    print(f'  SNR norm   : {result.snr_norm:.3f}')
    print(f'  Tau        : {result.tau_estimate:.3f}')
    print(f'  Memory     : {result.memory_match.best_label}, similarity={result.memory_match.best_similarity:.3f}, strength={result.memory_match.strength:.3f}\n')
    print('Class probabilities:')
    for n,p in sorted(result.probabilities.items(), key=lambda item:item[1], reverse=True): print(f'  {n:<14}: {p*100:6.2f}%')
if __name__=='__main__': main()
