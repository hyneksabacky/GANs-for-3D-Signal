import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Discriminator, Generator, weights_init
from preprocessing import Dataset
from tqdm import tqdm

lr = 2e-4
beta1 = 0.5
epoch_num = 128
batch_size = 128
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nd = 6 # number of dimensions

activities = {'"walk"' : 0, '"sit"' : 1, '"car"' : 2}#, '"bus"' : 3, '"lie"' : 4}  # List of activities

def main():
    # activities = ['"walk"', '"lie"', '"car"', '"bus"', '"sit"']  # List of activities
    trainset = Dataset('./data/acce_data_xyz.h5', activities)

    # print shape of dataset
    print(f"Loaded dataset shape: {trainset.dataset.shape}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    criterion = nn.BCELoss()

    # used for visualzing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    G_losses = []
    D_losses = []

    axes_names = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']


    loop = tqdm(range(epoch_num), total=epoch_num, leave=False)
    for epoch in loop:
        for step, (data, labels) in enumerate(trainloader):
            labels = labels.to(device)
            labels_one_hot = torch.zeros(labels.size(0), len(activities.keys()), device=device)
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            # train netD
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            netD.zero_grad()
            output = netD(real_cpu, labels_one_hot).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise, labels_one_hot)
            label.fill_(fake_label)
            output = netD(fake.detach(), labels_one_hot).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train netG
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake, labels_one_hot).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (epoch, epoch_num, step, len(trainloader),
            #          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # # save training process
        # with torch.no_grad():
        #     fake = netG(fixed_noise).detach().cpu()
        #     f, axes = plt.subplots(nd, 4, figsize=(12, 9))
        #     for i in range(nd):
        #         for j in range(4):
        #             axes[i, j].plot(fake[j, i, :].view(-1).numpy())
        #             axes[i, j].set_xticks(())
        #             axes[i, j].set_yticks(())
        #             axes[i, j].set_title(f'{axes_names[i]}')
        #     plt.savefig('./img/dcgan_epoch_%d.png' % epoch)
        #     plt.close()

        
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # save models
    torch.save(netG.state_dict(), './nets/dcgan_netG_more_activities.pkl')
    torch.save(netD, './nets/dcgan_netD.pkl')

if __name__ == '__main__':
    main()