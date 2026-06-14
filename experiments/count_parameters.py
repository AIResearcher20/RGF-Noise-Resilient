import sys
sys.path.append('..')
from models.rgf import RGF

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = RGF(1703, 64, 5)
params = count_parameters(model)
print(f"RGF Parameters: {params:,}")
