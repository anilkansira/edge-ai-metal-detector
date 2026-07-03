import random
from .materials import MATERIALS
from .soil import SOILS, SoilProfile
from .pulse_model import PulseConfig, simulate_ground_response, simulate_target_response, combine_waveforms
from .ground_balance import apply_ground_balance
from .feature_extraction import extract_waveform_features

CLASS_NAMES=['Iron','Aluminum','Copper','Brass','Steel','Unknown','Mixed Target','Ground Noise']

def soil_features_dict(soil: SoilProfile):
    return {'magnetic_susceptibility':soil.magnetic_susceptibility,'electrical_conductivity':soil.electrical_conductivity,'moisture':soil.moisture,'salinity':soil.salinity,'iron_oxide_level':soil.iron_oxide_level,'noise_floor':soil.noise_floor}

def generate_material_sample(material_name, config, soil_name=None, depth_m=None, horizontal_distance_m=None):
    mat=MATERIALS[material_name]; soil=SOILS[soil_name] if soil_name else random.choice(list(SOILS.values()))
    depth_m=random.uniform(0.08,2.0) if depth_m is None else depth_m
    horizontal_distance_m=random.uniform(0.0,4.2) if horizontal_distance_m is None else horizontal_distance_m
    ground=simulate_ground_response(soil,config)
    target,meta=simulate_target_response(mat,soil,config,depth_m,horizontal_distance_m,coil_height_cm=random.uniform(4.5,11.0),sweep_quality=random.uniform(0.45,1.0),orientation_factor=random.uniform(0.55,1.0),size_factor=random.uniform(0.75,1.25),corrosion=random.uniform(0,0.35))
    measured=combine_waveforms(ground,target); comp,err=apply_ground_balance(measured,ground,soil,True)
    fv=extract_waveform_features(comp,config,meta['phase_proxy'],meta['ferrous_index'],meta['conductivity_response'],meta['depth_m'],soil_features_dict(soil),err,sweep_quality=random.uniform(0.45,1.0),signal_stability=random.uniform(0.45,1.0),coil_height_cm=random.uniform(4.5,11.0),emi_noise=config.base_emi_noise,electronics_noise=config.preamp_noise,battery_percent=random.uniform(70,100))
    return fv.values, mat.name

def generate_noise_sample(label, config):
    soil=random.choice(list(SOILS.values())); ground=simulate_ground_response(soil,config); mixed_ratio=0.0
    if label=='Mixed Target':
        ma=random.choice(list(MATERIALS.values())); mb=random.choice(list(MATERIALS.values()))
        ta,aa=simulate_target_response(ma,soil,config,random.uniform(0.15,1.5),random.uniform(0,1.2))
        tb,bb=simulate_target_response(mb,soil,config,random.uniform(0.15,1.5),random.uniform(0.2,1.5))
        measured=combine_waveforms(ground,ta,tb); phase=(aa['phase_proxy']+bb['phase_proxy'])/2; ferro=(aa['ferrous_index']+bb['ferrous_index'])/2; cond=(aa['conductivity_response']+bb['conductivity_response'])/2; depth=(aa['depth_m']+bb['depth_m'])/2; mixed_ratio=random.uniform(0.55,1.0)
    elif label=='Ground Noise':
        measured=ground; phase=random.uniform(-20,35); ferro=min(1.0,soil.magnetic_susceptibility*0.45+soil.iron_oxide_level*0.25); cond=min(1.0,soil.electrical_conductivity*0.65+soil.salinity*0.25); depth=random.uniform(0,0.3)
    else:
        measured=ground; phase=random.uniform(-45,50); ferro=random.uniform(0,1); cond=random.uniform(0.05,1.1); depth=random.uniform(0.2,2.2); mixed_ratio=random.uniform(0,0.65)
    comp,err=apply_ground_balance(measured,ground,soil,True)
    fv=extract_waveform_features(comp,config,phase,ferro,cond,depth,soil_features_dict(soil),err,sweep_quality=random.uniform(0.25,1.0),signal_stability=random.uniform(0.20,1.0),mixed_ratio=mixed_ratio,coil_height_cm=random.uniform(4,16),emi_noise=random.uniform(0.006,0.05),electronics_noise=random.uniform(0.006,0.03),battery_percent=random.uniform(50,100))
    return fv.values,label

def generate_dataset(config=None, samples_per_material=800, noise_samples=800, seed=42):
    random.seed(seed); config=PulseConfig() if config is None else config; samples=[]; labels=[]
    for m in MATERIALS:
        for _ in range(samples_per_material):
            x,y=generate_material_sample(m,config); samples.append(x); labels.append(y)
    for lab in ['Unknown','Ground Noise','Mixed Target']:
        for _ in range(noise_samples):
            x,y=generate_noise_sample(lab,config); samples.append(x); labels.append(y)
    combo=list(zip(samples,labels)); random.shuffle(combo); return [x for x,_ in combo],[y for _,y in combo]

def train_validation_split(samples, labels, validation_ratio=0.2, seed=7):
    random.seed(seed); combo=list(zip(samples,labels)); random.shuffle(combo); split=int(len(combo)*(1-validation_ratio)); tr=combo[:split]; va=combo[split:]
    return [x for x,_ in tr],[y for _,y in tr],[x for x,_ in va],[y for _,y in va]
