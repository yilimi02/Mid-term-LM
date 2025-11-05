import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    # state: arbitrary dict
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)

def plot_curves(losses, val_losses, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training Curves')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
