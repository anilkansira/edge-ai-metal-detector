from dataclasses import dataclass
from .classifier import GaussianProbabilisticClassifier
from .dataset_generator import CLASS_NAMES, generate_dataset, train_validation_split, soil_features_dict
from .waveform_memory import VerifiedWaveformMemory, MemoryMatch
from .pulse_model import PulseConfig, simulate_ground_response, simulate_target_response, combine_waveforms
from .ground_balance import apply_ground_balance
from .feature_extraction import extract_waveform_features
from .materials import MATERIALS
from .soil import SOILS

@dataclass(frozen=True)
class PredictionResult:
    probabilities: dict
    predicted_class: str
    confidence: float
    memory_match: MemoryMatch
    snr_norm: float
    tau_estimate: float
    feature: list

class PulseInductionPipeline:
    def __init__(self, config=None):
        self.config=PulseConfig() if config is None else config
        self.classifier=GaussianProbabilisticClassifier(CLASS_NAMES)
        self.memory=VerifiedWaveformMemory(CLASS_NAMES)
        self.is_trained=False
    def train(self, samples_per_material=800, noise_samples=800, seed=42):
        x,y=generate_dataset(self.config,samples_per_material,noise_samples,seed)
        tx,ty,vx,vy=train_validation_split(x,y)
        self.classifier.fit(tx,ty); self.is_trained=True
        return self.classifier.evaluate(vx,vy)
    def _calibrate(self, probs, feature):
        signal, depth, soil_noise, soil_diff, snr, gb = feature[0], feature[5], feature[13], feature[14], feature[26], feature[30]
        temp=1.55 + soil_noise*5.5 + depth*0.08 + soil_diff*0.25 + gb*0.35
        soft={k:v**(1.0/temp) for k,v in probs.items()}; total=sum(soft.values()); soft={k:v/total for k,v in soft.items()}
        soft['Unknown'] += max(0,0.16-signal)*0.95 + max(0,0.35-snr)*0.20 + max(0,depth-1.25)*0.10
        soft['Ground Noise'] += soil_noise*1.10 + gb*0.18 + max(0,0.12-signal)*0.45
        total=sum(soft.values()); return {k:v/total for k,v in soft.items()}
    def _fuse_with_memory(self, calibrated, feature):
        match=self.memory.match(feature,self.classifier.normalize_one)
        if match.strength<=0.03: return calibrated,match
        blend=min(0.62,match.strength)
        fused={c:calibrated.get(c,0)*(1-blend)+match.probabilities.get(c,0)*blend for c in CLASS_NAMES}
        total=sum(fused.values()); return {c:v/total for c,v in fused.items()},match
    def predict_from_feature(self, feature):
        if not self.is_trained: raise RuntimeError('pipeline must be trained')
        raw=self.classifier.predict_proba(feature); cal=self._calibrate(raw,feature); probs,match=self._fuse_with_memory(cal,feature)
        pred=max(probs,key=probs.get); q=(max(0.001,feature[0])*max(0.001,feature[26])*max(0.001,1-feature[30]))**(1/3)
        return PredictionResult(probs,pred,probs[pred]*q,match,feature[26],feature[2],feature)
    def simulate_and_predict(self, material_name, soil_name, depth_m, horizontal_distance_m=0.25):
        if not self.is_trained: raise RuntimeError('pipeline must be trained')
        mat=MATERIALS[material_name]; soil=SOILS[soil_name]
        ground=simulate_ground_response(soil,self.config); target,meta=simulate_target_response(mat,soil,self.config,depth_m,horizontal_distance_m)
        measured=combine_waveforms(ground,target); comp,err=apply_ground_balance(measured,ground,soil,True)
        fv=extract_waveform_features(comp,self.config,meta['phase_proxy'],meta['ferrous_index'],meta['conductivity_response'],meta['depth_m'],soil_features_dict(soil),err)
        return self.predict_from_feature(fv.values)
    def add_verified_sample(self, feature, label):
        self.memory.add_sample(feature,label)
