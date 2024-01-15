import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Lambda, Resize, ToTensor
from torchvision.utils import save_image

from diffusion import GaussianDiffusion, linear_beta_schedule
from src.fashion_mnist_dataset import DDPMFashionMNIST
from src.unet import Unet


def loss_fn(model: nn.Module, diffusion: GaussianDiffusion, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """DDPM loss function.

    Args:
        model: Model to generate eps_theta(x_t, t).
        diffusion: Gaussian diffusion object.
        x0: Initial image: [b, c, h w].
        t: Timesteps batch: [b, 1].

    Returns:
        Smooth L1 loss between predicted and actual noise applied to the input image.
    """
    eps = torch.randn_like(x0)

    x_noisy = diffusion.q_sample(x0=x0, t=t, eps=eps)
    predicted_noise = model(x_noisy, t)

    return F.smooth_l1_loss(eps, predicted_noise)


def train(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataloader: DataLoader,
    optimizer,
    device,
    num_epochs: int,
    save_path: Path,
):
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

            loss = loss_fn(model=model, diffusion=diffusion, x0=batch, t=t)

            if step % 100 == 0:
                print(f"loss: {loss.item()}")

            loss.backward()
            optimizer.step()

        samples = diffusion.sample(model=model, image_size=28, batch_size=4, channels=1)

        all_images_to_save = (samples[-1] + 1) * 0.5
        save_image(all_images_to_save, save_path / f"{epoch}-samples.png", nrow=1)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        torch.save(checkpoint, save_path / f"{epoch}-checkpoint.pth")


parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=28)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=6)
parser.add_argument("--diffusion_timesteps", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cuda:1")
if __name__ == "__main__":
    args = parser.parse_args()

    # Setup dataloader
    transform = Compose(
        [
            Resize(args.image_size),
            CenterCrop(args.image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    dataset = DDPMFashionMNIST("./data", train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Setup model and optimizer
    model = Unet(channels=1, dim_mults=(1, 2, 4), dim=args.image_size)
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Setup diffusion process
    diffusion = GaussianDiffusion(noise_schedule=linear_beta_schedule, timesteps=args.diffusion_timesteps)

    save_path = "results"
    os.makedirs(save_path, exist_ok=True)

    train(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        save_path=Path(save_path),
    )
