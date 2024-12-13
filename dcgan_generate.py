import torch
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Generator

import os

seed = 420

lr = 2e-4
beta1 = 0.5
epoch_num = 32
batch_size = 8
nz = 100  # length of noise
ngpu = 0
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#load netG from pkl file
netG = Generator(nz)
netG.load_state_dict(torch.load('./nets/dcgan_netG.pkl', weights_only=True))
netG.eval()

torch.manual_seed(seed)
# used for visualzing training process
fixed_noise = torch.randn(16, nz, 1, device=device)

print(fixed_noise)

fake = netG(fixed_noise).detach().cpu()

f, a = plt.subplots(2)
a[0].plot(fixed_noise[0].view(-1))
a[0].set_title('Fixed Noise')
a[1].plot(fake[0].view(-1))
a[1].set_title('Generated Signal')
f.tight_layout(pad=5.0)
plt.show()