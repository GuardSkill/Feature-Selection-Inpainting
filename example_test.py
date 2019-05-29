import torch
import torch.nn as nn
input=torch.ones(1, 1, 5, 5)
m = nn.UpsamplingNearest2d(scale_factor=2)
print(m(input))