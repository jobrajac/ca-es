import logging
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
import torch
import time
import torch.tensor as tt
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from google.google import CAModelGoogle
from net_modeltest import CAModel
from utils import load_emoji, to_rgb, get_visualization_of_channels
from weight_updates import hebbian_update

reload(logging)

size = 9
TARGET_IMG = load_emoji("🍉", size)
torch.set_num_threads(1)

def mse(y_hat, y):
    """returns the mean squared error between the two inputs"""
    return ((y_hat - y) ** 2).mean()

def dmg(img, a,b,c, onlyTop=True, onlyBottom=False): #,rnd,rnd2,rnd3, 
    """Returns the input image, damaged in a rectangle of random size"""
    max_dmg = 5 # 4 px wide
    if onlyTop and onlyBottom:
        raise ValueError("Can't have exclusive damage on both sides")
    x_size = np.random.randint(2, 5) # 2, 3 or 4
    y_size = np.random.randint(2, 5)
    half_size = size // 2
    if onlyTop:
        x_start = np.random.randint(0, size-x_size - half_size)
    elif onlyBottom:
        x_start = np.random.randint(half_size + 1, size-x_size + 1)
    else:
        x_start = np.random.randint(0, size-x_size + 1)
    y_start = np.random.randint(0, size-y_size  + 1)
    return img

def get_rand_nums():
    """return three unique random numbers between 4 and 11 (inclusive)"""
    rnd = np.random.randint(4, 12)
    rnd2 = np.random.randint(4, 12)
    rnd3 = np.random.randint(4, 12)
    while rnd == rnd2:
        rnd2 = np.random.randint(4, 12)
    while rnd3 == rnd2 or rnd3 == rnd:
        rnd3 = np.random.randint(4, 12)

    return rnd, rnd2, rnd3


def dmg_noise(img):
    """returns the input image multiplied with random noise"""
    start = 0
    width = 9
    stop = start + width
    channels = 12
    update_mask = img[..., 3:4] > 0.1
    noise = np.random.randn(width, width, channels) * 0.03 * update_mask
    img[:, start:stop, start:stop, 4:] += noise
    return img


def train_step_hebb(model_try, coeffs_try, x, iter_n):
    """Perform an episode using Hebbian"""
    weights1_2, weights2_3 = list(model_try.parameters())
    weights1_2 = weights1_2.detach().numpy()
    weights2_3 = weights2_3.detach().numpy()

    loss_log = []
    xs = []
    dxs = []
    rnd1, rnd2, rnd3 = get_rand_nums()
    for i in range(iter_n):
        if i in dmg_steps:
            x = dmg_func(x.numpy(), rnd1, rnd2, rnd3)
            x = tt(x)

        l = model_try.loss_f(x, TARGET_IMG)
        loss_log.append(l.item())

        xs.append(x.clone().numpy())
        o0, o1, x, dx = model_try(x)
        dxs.append(dx.clone())
        weights1_2, weights2_3 = hebbian_update(coeffs_try, weights1_2, weights2_3, o0.numpy(),
                                                o1.numpy(), x.numpy())

        (a, b) = (0, 1)
        list(model_try.parameters())[a].data /= list(model_try.parameters())[a].__abs__().max()
        list(model_try.parameters())[b].data /= list(model_try.parameters())[b].__abs__().max()
        list(model_try.parameters())[a].data *= 0.4
        list(model_try.parameters())[b].data *= 0.4

    return loss_log, xs, dxs


def train_step_es(model_try, x, iter_n):
    """Perform a trainstep using ES"""
    loss_log = []
    xs = []
    dxs = []

    rnd1, rnd2, rnd3 = get_rand_nums()
    tic = time.time()
    for i in range(iter_n):
        if i in dmg_steps:
            x = dmg_func(x.numpy(), rnd1, rnd2, rnd3)
            x = tt(x)

        loss_log.append(model_try.loss_f(x, TARGET_IMG))
        xs.append(x.clone().numpy())
        x, dx = model_try(x)
        # x = model_try(x)
        dxs.append(dx.clone())


    toc = time.time()
    return loss_log, xs, dxs


def appendPlot(x, y, color):
    """Append a line with points x and y to a plot with color 'color'"""
    ax1.plot(x, y, color=color)


def visualize_batch(xs, fname, nrow=1):
    """Save visualization of input x's to file"""
    for i in range(len(xs)):
        xs[i] = to_rgb(xs[i]).permute(0, 3, 1, 2)
    save_image(torch.cat(xs, dim=0), log_folder + fname + '.png', nrow=nrow, padding=0)


def visualize_iters(xs, iters, fname):
    """Save visualizations of input x's on selected iterations to file"""
    show = []
    for i in range(len(xs)):
        if i in iters:
            show.append(to_rgb(xs[i]).permute(0, 3, 1, 2))

    fname = log_folder + fname + '.png'
    save_image(torch.cat(show, dim=0), fname, nrow=len(show), padding=0)
    with Image.open(fname) as im:
        w, h = im.size
        scale_f = 10
        im = im.resize((w * scale_f, h * scale_f), resample=Image.BOX)
        top_border = 12
        w, h = im.size
        res = Image.new(im.mode, (w, h + top_border), color=(220, 220, 220))
        res.paste(im, (0, top_border))
        im = res
        d = ImageDraw.Draw(im)
        for i, it in enumerate(iters):
            d.text((i * (w // len(show)) + 33, 0), "i=" + str(it), fill=(0, 0, 0))

        im.save(fname)


HIDDEN_SIZE = 32


def weights_init(m):
    """Initialize a network with uniform values between -0.1 and 0.1"""
    if isinstance(m, torch.nn.Linear) and m.in_features != HIDDEN_SIZE:
        torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)


def test_hebb(path, hs):
    """Start growing processes for Hebb and return loss, dx's and x's"""
    lines = []
    coeffs = np.load(path)
    xss = []
    dxss = []
    for i in range(tests_per_model):
        model_try_hebb = CAModel(channel_n=channels, fire_rate=0.5, new_size_pad=size,
                                 disable_grad=True, hidden_size=hs, batch_size=1, use_hebb=True)
        model_try_hebb.apply(weights_init)
        losses_hebb, xs, dxs = train_step_hebb(model_try_hebb, coeffs.copy(), x0.detach().clone(), iter_n)
        lines.append(losses_hebb)
        dxss.append(dxs)
        xss.append(xs)

    return lines, xss, dxss


def test_es(path, hs):
    """Start growing processes for ES and return loss, dx's and x's"""
    lines = []
    model_try_es = CAModel(channel_n=channels, fire_rate=0.5, new_size_pad=size,
                           disable_grad=True, hidden_size=hs, batch_size=1, use_hebb=False)
    model_try_es.load_state_dict(torch.load(path))
    model_try_es.double()
    xss = []
    dxss = []
    for i in range(tests_per_model):
        x_temp = x0.detach().clone()
        losses_es, xs, dxs = train_step_es(model_try_es, x_temp, iter_n)
        lines.append(losses_es)
        xss.append(xs)
        dxss.append(dxs)

    return lines, xss, dxss


def start_tests(tests):
    for i, test in enumerate(tests):
        if test['method'] == "es":
            lines, xss, dxss = test_es(test['path'], test['hs'])
        elif test['method'] == "bp":
            lines, xss, dxss = test_es(test['path'], test['hs'])
        elif test['method'] == "hebb":
            lines, xss, dxss = test_hebb(test['path'], test['hs'])
        else:
            print("Invalid method")
            return
        plot(lines, test['color'], test['name'])
        plotDX(dxss, test['color'], test['name'])
        visualize_iters([tt(x) for x in xss[-1]], vis_iters, test['name'])
        # visualize_iters([tt(x) for x in xss[-2]], vis_iters, test['name'] + "2")
        plot_channels(xss[-1], test['name'])


legend_lines = []
def plot(lines, color, label):
    """plot losses of a model with color 'color' and name 'label'"""
    print(label)
    lines = np.array(lines)
    lines = np.log10(lines)
    l_min = np.zeros(iter_n)
    l_max = np.zeros(iter_n)
    avg = np.zeros(iter_n)

    for i in range(iter_n):
        std = np.std(lines[:, i])
        std = np.abs(std)
        avg[i] = np.average(lines[:, i])
        l_min[i] = avg[i] - std
        l_max[i] = avg[i] + std

    ax1.fill_between(iters, l_min, l_max, facecolor=color, alpha=0.15)
    ax1.set_ylim([-3, -0.5])
    avgline, = ax1.plot(iters, avg, color=color, alpha=1, label=label)
    legend_lines.append(avgline)


legend_lines2 = []
def plotDX(dxs, color, label):
    """plot dx's of a model with color 'color' and name 'label'"""
    lines = []
    l_min = np.zeros(iter_n)
    l_max = np.zeros(iter_n)
    avg = np.zeros(iter_n)
    for dx in dxs:
        line = []
        for d in dx:
            dif = np.absolute(d).sum()
            line.append(dif)
        line = np.array(line)
        lines.append(line)

    lines = np.array(lines)
    lines = np.log10(lines)
    for i in range(iter_n):
        std = np.std(lines[:, i])
        std = np.abs(std)
        avg[i] = np.average(lines[:, i])
        l_min[i] = avg[i] - std
        l_max[i] = avg[i] + std

    ax2.fill_between(iters, l_min, l_max, facecolor=color, alpha=0.2)
    avgline, = ax2.plot(iters, avg, color=color, alpha=1, label=label)
    legend_lines2.append(avgline)


def plot_channels(ch, name):
    """Save visualization of model channels on selected iterations to file"""
    iin = torch.squeeze(tt(ch))
    vis = iin[vis_iters]
    chs = get_visualization_of_channels(vis)
    fname = log_folder + "channels_" + name + ".png"
    save_image(chs, fname, nrow=channels + 1, padding=0)
    logging.info(name + ', hc max %s, hc min %s', chs[..., 4:].max().item(), chs[..., 4:].min().item())

    with Image.open(fname) as im:
        w, h = im.size
        scale_f = 10
        im = im.resize((w * scale_f, h * scale_f), resample=Image.BOX)
        left_border = 40
        w, h = im.size
        res = Image.new(im.mode, (left_border + w, h), color=(220, 220, 220))
        res.paste(im, (left_border, 0))
        im = res

        fnt = ImageFont.truetype("arial.ttf", 20)

        d = ImageDraw.Draw(im)

        b_size = h // len(vis_iters)
        for i, it in enumerate(vis_iters):
            fw, fh = d.textsize(str(it))
            d.text(((left_border - fw - 10) / 2, (i * b_size + (b_size - fh) / 2) - 7), str(it), fill=(0, 0, 0),
                   font=fnt)

        im.save(fname)


log_folder = "../results/testing/"
if os.path.isdir(log_folder):
    shutil.rmtree(log_folder)
os.mkdir(log_folder)


logging.basicConfig(filename=log_folder + 'info.log', level=logging.INFO, format='%(message)s', filemode="w")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

channels = 16
iter_n = 70
tests_per_model = 10
dmg_steps = [35]
dmg_func = dmg
iters = np.arange(iter_n)
seed = np.zeros([size, size, channels], np.float64)
seed[size // 2, size // 2, 3:] = 1.0
x0 = tt(np.repeat(seed[None, ...], 1, 0))
ext_model = CAModel(channel_n=channels, fire_rate=0.5, new_size_pad=size, disable_grad=False, hidden_size=42,
                    batch_size=1)

vis_iters = [0, 1, 2, 5, 15, 25, 35]
epsilon = 0.05

tests = []

# Models trained to grow
tests.append({
    "method":"bp",
    "path": "../models/final/adam/bp2.5adam.pt",
    "color": "#377eb8",
    "name": "Adam",
    "hs": 32
})
tests.append({
    "method":"bp",
    "path": "../models/final/sgdm/bp2.5sgdm.pt",
    "color": "#984ea3",
    "name": "SGDM",
    "hs": 32
})
tests.append({
    "method": "es",
    "path": "../models/final/es/train_log_es/models/saved_model_440000.pt",
    "color": "#e41a1c",
    "name": "ES",
    "hs": 32
})
tests.append({
    "method":"hebb",
    "path": "../models/final/hebbian/train_log_hebb_s9_2/models/999000.npy",
    "color": "#4daf4a",
    "name": "Hebbian",
    "hs": 32
})


# dmg random rectangular

# tests.append({
#     "method":"bp",
#     "path": "../models/final/adam/bp2.4Adam_regen.pt",
#     "color": "#377eb8",
#     "name": "Adam",
#     "hs": 32
# })
# tests.append({
#     "method":"bp",
#     "path": "../models/final/sgdm/bp2.3SGDM_regen_moment95.pt",
#     "color": "#984ea3",
#     "name": "SGDM",
#     "hs": 32
# })
# tests.append({
#     "method": "es",
#     "path": "../models/final/es/train_log_es_pool_regen_2/models/saved_model_2000.pt",
#     "color": "#e41a1c",
#     "name": "ES",
#     "hs": 32
# })
# tests.append({
#     "method":"hebb",
#     "path": "../logs/hebbian/train_log_hebb_regen_fixed/models/15000.npy",
#     "color": "#4daf4a",
#     "name": "Hebbian",
#     "hs": 32
# })

# dmg green

# tests.append({
#     "method":"bp",
#     "path": "../models/final/adam/bp2.3adam_green.pt",
#     "color": "#377eb8",
#     "name": "Adam",
#     "hs": 32
# })
# tests.append({
#     "method":"bp",
#     "path": "../models/final/sgdm/bp2.3sgdm_green.pt",
#     "color": "#984ea3",
#     "name": "SGDM",
#     "hs": 32
# })
# tests.append({
#     "method":"es",
#     "path": "../models/final/es/train_log_es_dmggreen/models/saved_model_100000.pt",
#     "color": "#e41a1c",
#     "name": "ES",
#     "hs": 32
# })
# tests.append({
#     "method":"hebb",
#     "path": "../models/final/hebbian/hebb_green_ny/models/295000.npy",
#     "color": "#4daf4a",
#     "name": "Hebbian",
#     "hs": 32
# })

# dmg channels

# tests.append({
#     "method":"bp",
#     "path": "../models/final/adam/bp2.7adam_4ch.pt",
#     "color": "#377eb8",
#     "name": "Adam",
#     "hs": 32
# })
# tests.append({
#     "method":"bp",
#     "path": "../models/final/sgdm/bp2.7sgdm_4ch.pt",
#     "color": "#984ea3",
#     "name": "SGDM",
#     "hs": 32
# })
# tests.append({
#     "method":"es",
#     "path": "../models/final/es/train_log_es_chdmg/models/saved_model_600000.pt",
#     "color": "#e41a1c",
#     "name": "ES",
#     "hs": 32
# })
# tests.append({
#     "method":"hebb",
#     "path": "../models/final/hebbian/train_log_hebb_chdmg/models/999000.npy",
#     "color": "#4daf4a",
#     "name": "Hebbian",
#     "hs": 32
# })


for test in tests:
    logging.info("Running: " + test["name"] + " from " + test["path"])

logging.info(str(tests_per_model) + " times per model")

start_tests(tests)

ax1.legend(handles=legend_lines)
fig1.suptitle("$ \log_{10} $(loss) over update steps", fontsize=20)
fig1.savefig(log_folder + "results.png", dpi=300)

ax2.legend(handles=legend_lines2)
fig2.suptitle("$ \log_{10} (dx)$ over update steps", fontsize=20)
fig2.savefig(log_folder + "dxchange.png", dpi=300)
