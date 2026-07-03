import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pi_detector.classifier import GaussianProbabilisticClassifier

def test_classifier_predicts_known_cluster():
    clf=GaussianProbabilisticClassifier(['A','B'])
    clf.fit([[0.1,0.1],[0.12,0.09],[0.9,0.9],[0.88,0.91]], ['A','A','B','B'])
    assert clf.predict([0.11,0.10])=='A'
