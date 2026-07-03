from .soil import SoilProfile

def apply_ground_balance(measured, ground_baseline, soil: SoilProfile, use_ground_balance=True):
    if len(measured) != len(ground_baseline):
        raise ValueError('measured and ground_baseline must have same length')
    if use_ground_balance:
        gain = 0.84 if 'Black Sand' in soil.name else 0.96
        gain -= soil.magnetite_compensation * 0.08
        gain = max(0.76, min(0.98, gain))
        target=[max(0.0, measured[i]-ground_baseline[i]*gain) for i in range(len(measured))]
        err=sum(abs(measured[i]-ground_baseline[i]) for i in range(len(measured)))/len(measured)
    else:
        gain = 0.66 if 'Black Sand' in soil.name else 0.50
        target=[max(0.0, measured[i]-ground_baseline[i]*gain) for i in range(len(measured))]
        err=min(1.0, soil.false_positive_risk*0.32 + soil.noise_floor*1.25)
    return target, err
