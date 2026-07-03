from math import exp, log, pi, sqrt

class GaussianProbabilisticClassifier:
    def __init__(self, class_names):
        self.class_names=class_names; self.feature_mean=[]; self.feature_std=[]; self.means={}; self.vars={}; self.priors={}; self.trained=False
    def fit_normalizer(self, samples):
        n=len(samples[0]); self.feature_mean=[]; self.feature_std=[]
        for j in range(n):
            vals=[r[j] for r in samples]; mean=sum(vals)/len(vals); var=sum((v-mean)**2 for v in vals)/len(vals)
            self.feature_mean.append(mean); self.feature_std.append(sqrt(max(var,1e-8)))
    def normalize_one(self, sample):
        return [(v-m)/s for v,m,s in zip(sample,self.feature_mean,self.feature_std)]
    def fit(self, samples, labels):
        if len(samples)!=len(labels): raise ValueError('samples and labels length mismatch')
        self.fit_normalizer(samples); norm=[self.normalize_one(x) for x in samples]
        grouped={c:[] for c in self.class_names}
        for x,y in zip(norm,labels):
            if y in grouped: grouped[y].append(x)
        total=len(norm)
        for c in self.class_names:
            data=grouped[c] or [[0.0 for _ in samples[0]]]
            self.priors[c]=max(1,len(data))/max(1,total)
            means=[]; vars_=[]
            for j in range(len(data[0])):
                vals=[r[j] for r in data]; mean=sum(vals)/len(vals); var=sum((v-mean)**2 for v in vals)/len(vals)
                means.append(mean); vars_.append(max(var,0.080))
            self.means[c]=means; self.vars[c]=vars_
        self.trained=True
    def predict_proba(self, sample):
        if not self.trained: raise RuntimeError('classifier must be fitted')
        x=self.normalize_one(sample); logs={}
        for c in self.class_names:
            lp=log(self.priors.get(c,1e-12)+1e-12)
            for v,m,var in zip(x,self.means[c],self.vars[c]):
                lp += -0.5*log(2*pi*var) - ((v-m)**2)/(2*var)
            logs[c]=lp
        mx=max(logs.values()); ex={c:exp(logs[c]-mx) for c in self.class_names}; total=sum(ex.values())
        return {c:ex[c]/total for c in self.class_names}
    def predict(self, sample):
        p=self.predict_proba(sample); return max(p,key=p.get)
    def evaluate(self, samples, labels):
        return sum(self.predict(x)==y for x,y in zip(samples,labels))/len(samples) if samples else 0.0
