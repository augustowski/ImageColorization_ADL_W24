import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
import os
import requests

# GitHub release URL for res18-unet pretrained generator
GITHUB_API_URL = "https://github.com/augustowski/ImageColorization_ADL_W24/releases/download/trained/"
PRETRAINED_GENERATOR = "res18-unet.pt"

# Model Initialization Function
def init_model(model, device):
    model = model.to(device)
    return model

# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = nn.BCEWithLogitsLoss() if gan_mode == 'vanilla' else nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        return self.loss(preds, labels)

# Build ResNet18 UNet
def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18(pretrained=True), n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G

# MainModel
class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lambda_L1=100.0):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        # Initialize generator
        if net_G is None:
            self.net_G = init_model(build_res_unet(n_input=1, n_output=2, size=256), self.device)
        else:
            self.net_G = net_G.to(self.device)

        # Define L1 Loss
        self.L1criterion = nn.L1Loss()

    def forward(self, x):
        return self.net_G(x)

# Load pretrained generator
def load_pretrained_generator(save_dir="models2"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, PRETRAINED_GENERATOR)

    # Download if not available locally
    if not os.path.exists(model_path):
        print(f"Downloading pretrained generator from {GITHUB_API_URL}{PRETRAINED_GENERATOR}...")
        response = requests.get(f"{GITHUB_API_URL}{PRETRAINED_GENERATOR}", stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded pretrained generator to {model_path}")
        else:
            raise RuntimeError(f"Failed to download pretrained generator. HTTP Status Code: {response.status_code}")

    # Load the generator weights
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return net_G

# Filter State Dict for Inference
def filter_state_dict(state_dict, prefix="net_G"):
    """
    Filters a state_dict to include only keys starting with a specific prefix.
    """
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix)}

# Load MainModel for Inference
def load_main_model_for_inference(model_path, device):
    """
    Loads the MainModel for inference, ensuring only the generator is initialized.
    """
    # Initialize model
    model = MainModel()

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = filter_state_dict(state_dict, prefix="net_G")
    model.net_G.load_state_dict(filtered_state_dict)

    model.to(device)
    model.eval()
    return model