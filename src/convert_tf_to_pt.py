import torch
import torch.tensor as tt
import tensorflow as tf
from tensorflow import keras
from google.google import CAModelGoogle
import numpy as np
from net import CAModel
from utils import to_rgb
from torchvision.utils import save_image
import os
import shutil
import glob


def train_step_es(model_try, x, iterations):
    iter_n = iterations
    xs = [x.clone()]
    for i in range(iter_n):
        x = model_try(x)
        xs.append(x.clone())
    return xs

FOLDER_PATH = "../models/final/sgdm"
to_convert = glob.glob(FOLDER_PATH + "/*.index")

if os.path.isdir(FOLDER_PATH + "/converted"):
    shutil.rmtree(FOLDER_PATH + "/converted")
os.mkdir(FOLDER_PATH + "/converted")
os.mkdir(FOLDER_PATH + "/converted/test_growths")

for e in to_convert:
    name = e.split("/")[-1][:-6] # name without .index
    hidden_size = 32
    tf_model = CAModelGoogle(16, 0.5, hidden_size)
    try: 
        tf_model.load_weights(FOLDER_PATH + "/" + name)
    except ValueError:
        hidden_size = 42
        tf_model = CAModelGoogle(16, 0.5, hidden_size)
        tf_model.load_weights(FOLDER_PATH + "/" + name)

    weights = []
    for layer in tf_model.dmodel.layers:
        weights.append(layer.weights[-1].numpy()[0, 0, :, :])

    pt_model = CAModel(16, 0.5, hidden_size, 9, 1)
    for i, p in enumerate(pt_model.parameters()):        
        p.data = torch.tensor(weights[i].T.astype(np.float64))

    torch.save(pt_model.state_dict(), FOLDER_PATH + "/converted/%s.pt" %name)

    # Test that the translated model works
    h, w = 9, 9
    ch = 16
    seed = np.zeros([h, w, ch], np.float64)
    seed[h // 2, w // 2, 3:] = 1.0
    x0 = tt(np.repeat(seed[None, ...], 1, 0))
    xs = train_step_es(pt_model, x0, 35)
    save_image(to_rgb(xs[-1]).permute(0, 3, 1, 2), FOLDER_PATH + "/converted/test_growths/%s.png"%name)


