from dataclasses import dataclass
from math import exp, sqrt

@dataclass(frozen=True)
class MemoryMatch:
    probabilities: dict
    strength: float
    best_label: str
    best_similarity: float

class VerifiedWaveformMemory:
    def __init__(self, class_names, max_samples=2200):
        self.class_names=class_names; self.max_samples=max_samples; self.samples=[]; self.labels=[]; self.memory_counts={c:0 for c in class_names}
    def add_sample(self, feature, label):
        if label not in self.class_names: raise ValueError(f'unknown label: {label}')
        if label in {'Unknown','Ground Noise','Mixed Target'}: return
        self.samples.append(feature[:]); self.labels.append(label); self.memory_counts[label]+=1
        if len(self.samples)>self.max_samples:
            over=len(self.samples)-self.max_samples; self.samples=self.samples[over:]; self.labels=self.labels[over:]
    def match(self, feature, normalize_fn):
        if not self.samples:
            return MemoryMatch({c:0.0 for c in self.class_names},0.0,'None',0.0)
        x=normalize_fn(feature); scores={c:0.0 for c in self.class_names}; counts={c:0 for c in self.class_names}
        weights=[]
        for i in range(len(feature)):
            if 15<=i<=22: weights.append(3.60)
            elif i in {0,2,3,4,23,24,25,26}: weights.append(1.90)
            elif i in {8,9,10,11,12,13,14}: weights.append(0.55)
            else: weights.append(1.00)
        sigma=2.95; best_label='None'; best_sim=0.0
        for sample,label in zip(self.samples,self.labels):
            s=normalize_fn(sample); dist=0.0; wsum=0.0
            for a,b,w in zip(x,s,weights): dist += w*((a-b)**2); wsum += w
            nd=sqrt(dist/max(wsum,1e-9)); sim=exp(-(nd**2)/(2*sigma*sigma))
            if sim<0.22: continue
            scores[label]+=sim**2.05; counts[label]+=1
            if sim>best_sim: best_sim=sim; best_label=label
        total=sum(scores.values())
        if total<=0: return MemoryMatch({c:0.0 for c in self.class_names},0.0,'None',0.0)
        probs={c:scores[c]/total for c in self.class_names}; bc=counts.get(best_label,0)
        sf=max(0.0,min(1.0,(best_sim-0.22)/0.62)); cf=min(1.0,bc/8.0); tf=min(1.0,total/8.0)
        strength=max(0.0,min(0.76,0.66*sf+0.24*cf+0.10*tf))
        if bc<=1: strength=min(strength,0.24)
        elif bc<=3: strength=min(strength,0.40)
        elif bc<=6: strength=min(strength,0.58)
        else: strength=min(strength,0.76)
        return MemoryMatch(probs,strength,best_label,best_sim)
