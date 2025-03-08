import torch, torchvision, os, PIL, pdb
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

root_path='./data/'

def show(tensor, num=25):
  data = tensor.detach().cpu()
  grid = make_grid(data[:num], nrow=5).permute(1,2,0)
  plt.imshow(grid.clip(0,1))
  plt.show()


# initialising weights in different ways
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
      torch.nn.init.normal_(m.weight, 0.0, 0.02)
      torch.nn.init.constant_(m.bias,0)

    if isinstance(m,nn.BatchNorm2d):
      torch.nn.init.normal_(m.weight, 0.0, 0.02)
      torch.nn.init.constant_(m.bias,0)

# gen=gen.apply(init_weights)
# crit=crit.apply(init_weights)


# gradient penalty calculation
def get_gp(real, fake, crit, alpha, gamma=10):
  mix_images = real * alpha + fake * (1-alpha) # 128 x 3 x 128 x 128
  mix_scores = crit(mix_images) # 128 x 1

  gradient = torch.autograd.grad(
      inputs = mix_images,
      outputs = mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True,
  )[0] # 128 x 3 x 128 x 128

  gradient = gradient.view(len(gradient), -1)   # 128 x 49152
  gradient_norm = gradient.norm(2, dim=1)
  gp = gamma * ((gradient_norm-1)**2).mean()
  return gp


def save_checkpoint(epoch, num):
  torch.save({
    'epoch': epoch,
    'model_state_dict': gen.state_dict(),
    'optimizer_state_dict': gen_opt.state_dict()
  }, f"{root_path}Gen-{epoch}-{num}.pkl")

  torch.save({
    'epoch': epoch,
    'model_state_dict': crit.state_dict(),
    'optimizer_state_dict': crit_opt.state_dict()
  }, f"{root_path}Critic-{epoch}-{num}.pkl")

  print("Saved checkpoint")


def load_checkpoint(epoch, num):
  checkpoint = torch.load(f"{root_path}Gen-{epoch}-{num}.pkl")
  gen.load_state_dict(checkpoint['model_state_dict'])
  gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  checkpoint = torch.load(f"{root_path}Critic-{epoch}-{num}.pkl")
  crit.load_state_dict(checkpoint['model_state_dict'])
  crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  print("Loaded checkpoint")
