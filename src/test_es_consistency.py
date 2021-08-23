import torch
import torch.tensor as tt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from net import CAModel
import numpy as np
from utils import to_rgb, load_emoji
from torchvision.utils import save_image
import argparse
import json
import glob
import os
import shutil

torch.set_num_threads(1)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def train_step_es(model_try, x, iterations):
    iter_n = iterations
    for i in range(iter_n):
        if i == 39:
            x_at_39 = x.clone()
        x = model_try(x)
    return x, x_at_39

SAVE_FOLDER = "../results" + "/es_consistency_over_generations"

if os.path.isdir(SAVE_FOLDER):
    shutil.rmtree(SAVE_FOLDER)
os.mkdir(SAVE_FOLDER)

PRE_TRAINED_ROOT = "../models/final/es/train_log_es"

pt_args = argparse.Namespace()
with open(PRE_TRAINED_ROOT + '/commandline_args.txt', 'r') as f: # Load configurations used
    pt_args.__dict__ = json.load(f)

BATCH_SIZE = 1

model_try = CAModel(channel_n=pt_args.channels, fire_rate=pt_args.fire_rate, new_size_pad=pt_args.size,
                        hidden_size=pt_args.hidden_size, batch_size=BATCH_SIZE, use_hebb=pt_args.hebb)

h = pt_args.size + pt_args.pad * 2
w = pt_args.size + pt_args.pad * 2
seed = np.zeros([h, w, pt_args.channels], np.float64)
seed[h // 2, w // 2, 3:] = 1.0
x0 = tt(np.repeat(seed[None, ...], BATCH_SIZE, 0))

models = glob.glob(PRE_TRAINED_ROOT + "/models/saved_model_*.pt")
models = sorted(models, key=lambda x: int(x.split("_")[-1][:-3]))

print(len(models))
num_models = len(models)
# models = models[:num_models]
models = models[:65]

print(models)

model_try.load_state_dict(torch.load(models[-1]))
target_img = load_emoji(pt_args.emoji, pt_args.size)
p = pt_args.pad
pad_target = F.pad(tt(target_img), (0, 0, p, p, p, p))

run_per_model = 10
losses = np.empty((len(models), run_per_model))
losses_at_39 = np.empty(losses.shape)

xs = []
xs_at_39 = []
update_steps = 200
for i, m in enumerate(models):
    model_try.load_state_dict(torch.load(m)) # Load model
    for j in range(run_per_model):        
        x, x_at_39 = train_step_es(model_try, x0.clone(), update_steps)
        curr_loss = model_try.loss_f(x, pad_target).mean().item()
        curr_loss_at_39 = model_try.loss_f(x_at_39, pad_target).mean().item()
        losses[i, j] = curr_loss
        losses_at_39[i, j] = curr_loss_at_39

    xs.append(x)
    xs_at_39.append(x_at_39)

def get_betweens(lines):
    lines_between = np.empty((len(models), 2))
    for i in range(len(lines)):
        std = np.std(np.log10(lines[i, :]))
        std = np.abs(std)
        avg = np.mean(np.log10(lines[i, :]))
        lines_between[i, 0] = avg - std
        lines_between[i, 1] = avg + std
    return lines_between

losses_between = get_betweens(losses)
losses_between_at_39 = get_betweens(losses_at_39)

# Plot loss for t=40 and t=update_steps
x_range = np.arange(0, len(models)*1000, 1000) #[int(x.split("_")[-1][:-3]) for x in models]
plt.fill_between(x_range, losses_between[:, 0], losses_between[:, 1], facecolor="#353b48", alpha=0.15)
plt.plot(x_range, np.log10(np.mean(losses, axis=1)), color="#353b48")

plt.fill_between(x_range, losses_between_at_39[:, 0], losses_between_at_39[:, 1], facecolor="#e41a1c", alpha=0.15)
plt.plot(x_range, np.log10(np.mean(losses_at_39, axis=1)), color="#e41a1c")

plt.legend(["t=%d"%update_steps, "t=40"])
plt.yticks(np.arange(-2, 7), np.arange(-2, 7))
plt.suptitle("$ \log_{10} $(loss) over first %d generations" % (len(models)*1000), fontsize=20)
plt.savefig(SAVE_FOLDER + "/consistency_first%d.png" % (len(models)*1000))  

# Visualize images at t=40 and t=update_steps
vis = torch.cat([val for pair in zip(xs_at_39, xs) for val in pair], dim=0)
print(vis.shape)
save_image(to_rgb(vis).permute(0, 3, 1, 2), SAVE_FOLDER + "/image.png", nrow=2, padding=0)
