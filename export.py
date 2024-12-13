import torch

from dcgan import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nz = 100  # length of noise

# Create the VAE model
netG = Generator(nz).to(device)

netG.load_state_dict(torch.load('vae_model.pth', weights_only=True))
netG.to(device)
netG.eval()

# Prepare a dummy input for ONNX export
dummy_input = torch.randn(1, timesteps, input_dim).to(device)

# Export the loaded model to ONNX format
torch.onnx.export(
    vae_model,
    dummy_input,
    "vae_model.onnx",
    input_names=['input'],
    output_names=['output', 'mu', 'logvar'],
    dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                  'output': {0: 'batch_size', 1: 'sequence_length'}}
)

print("Trained model exported to vae_model.onnx")