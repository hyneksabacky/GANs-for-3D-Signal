import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input size: 256
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 128
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 64
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 32
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: 16
            nn.Conv1d(512, 1, kernel_size=16, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            # Input: latent vector (nz)
            nn.ConvTranspose1d(nz, 512, kernel_size=16, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            # State size: 16
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            # State size: 32
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # State size: 64
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # State size: 128
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output size: 256
        )

    def forward(self, x):
        x = self.main(x)
        return x
