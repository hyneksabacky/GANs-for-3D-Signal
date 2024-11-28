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

# # save training process
# with torch.no_grad():
#     fake = netG(fixed_noise).detach().cpu()
#     f, a = plt.subplots(4, 4, figsize=(8, 8))
#     for i in range(4):
#         for j in range(4):
#             a[i][j].plot(fake[i * 4 + j].view(-1))
#             a[i][j].set_xticks(())
#             a[i][j].set_yticks(())
#     #while file exists, keep increasing the number
#     i = 0
#     while os.path.exists('./img/dcgan_generated{}.png'.format(i)):
#         i += 1
#     plt.savefig('./img/dcgan_generated{}.png'.format(i))
#     plt.close()

fake = netG(fixed_noise).detach().cpu()

f, a = plt.subplots(2)
a[0].plot(fixed_noise[0].view(-1))
a[0].set_title('Fixed Noise')
a[1].plot(fake[0].view(-1))
a[1].set_title('Generated Signal')
plt.show()