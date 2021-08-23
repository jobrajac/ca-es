import io
import PIL.Image, PIL.ImageDraw
import numpy as np
import base64
import requests
import csv
import torch
from torchvision.utils import save_image
import glob
from PIL import Image


def load_emoji(emoji_code, img_size):
    """Loads image of emoji with code 'emoji' from google's emojirepository"""
    emoji_code = hex(ord(emoji_code))[2:].lower()
    url = 'https://raw.githubusercontent.com/googlefonts/noto-emoji/main/png/128/emoji_u%s.png' % emoji_code
    req = requests.get(url)
    img = PIL.Image.open(io.BytesIO(req.content))
    img.thumbnail((img_size, img_size), PIL.Image.ANTIALIAS)
    img = np.float64(img) / 255.0
    img[..., :3] *= img[..., 3:]
    return img


def to_rgba(x):
    """Return the four first channels (RGBA) of an image."""
    return x[..., :4]


def to_alpha(x):
    """Return the alpha channel of an image."""
    return torch.clamp(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    """Return the three first channels (RGB) with alpha deducted."""
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def export_model(ca, base_fn):
    """Save a PyTorch model to a specific path."""
    torch.save(ca.state_dict(), base_fn)


def visualize_batch(xs, step_i, log_folder, nrow=1):
    """Save a batch of multiple x's to file"""
    for i in range(len(xs)):
        xs[i] = to_rgb(xs[i]).permute(0, 3, 1, 2)
    save_image(torch.cat(xs, dim=0), log_folder + '/progress/batches_%04d.png' % step_i, nrow=nrow, padding=0)


def squash_tensor(x, min_val, max_val):
    """Normalize a tensor to a specific range."""
    x_np = x.clone().numpy()
    sq = np.interp(x_np, (x_np.min(), x_np.max()), (min_val, max_val))
    return torch.from_numpy(sq)
    
def squash_tensor_sign_exclusive(x, min_val, max_val):
    """Normalize a tensor to a specific range, but retain 0 as 0."""
    x = x.clone()
    if not torch.nonzero(x).any():
        return x
    positives = torch.abs((x > 0).double() * x)
    indices_positives = torch.nonzero(positives > 0, as_tuple=True)
    negatives = torch.abs((x < 0).double() * x)
    indices_negatives = torch.nonzero(negatives > 0, as_tuple=True)

    max_positives = positives.max().item()
    max_negatives = negatives.max().item()
    max_overall = max(max_positives, max_negatives)

    positives = squash_tensor(positives, min_val, max_val) * (max_positives / max_overall)
    negatives = squash_tensor(negatives, min_val, max_val) * (max_negatives / max_overall)

    x[indices_positives] = positives[indices_positives]
    x[indices_negatives] = - negatives[indices_negatives]
    return x

def write_file(filename, points_x, points_y):
    """Overwrite csv file with vectors x and y"""
    # This one overwrites
    with open(filename, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(len(points_x)):
            writer.writerow([points_x[i], points_y[i]])


def append_file(filename, points_x, points_y):
    """Append csv file with values x and y"""
    with open(filename, mode='a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(len(points_x)):
            writer.writerow([points_x[i], points_y[i]])


def read_file(filename):
    """Read csv file with rows: x, y"""
    px = []
    py = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            px.append(float(row[0]))
            py.append(float(row[1]))

    return px, py


def get_visualization_of_channels(xs, squash_across="none"):
    """Save visualization of the channels of a batch"""
    batch, h, w, ch = xs.shape
    print("Visualizing channels with squash_across = %s" % squash_across)
    if squash_across not in ["none", "evolution", "channels"]:
        raise NotImplementedError("Value %s for squash_across is not supported" % squash_across)
    if squash_across == "channels":
        for i in range(batch):
            xs[i, :, :, 4:] = squash_tensor_sign_exclusive(xs[i, :, :, 4:], 0, 1)
    elif squash_across == "evolution":
        for i in range(4, ch):
            xs[:, :, :, i] = squash_tensor_sign_exclusive(xs[:, :, :, i], 0, 1)
    elif squash_across == "none":
        xs[:, ..., 4:] = squash_tensor_sign_exclusive(xs[:, ..., 4:], 0, 1)

    newx = torch.zeros(ch * batch, 3, h, w)
    for i, x in enumerate(xs):
        x = x.unsqueeze(0).permute(3, 0, 1, 2)  # Channels = new batch_size
        for j in range(x.size(0)):
            filling_one = torch.ones(x[j].shape)
            if j < 3:
                newx[i * ch + j] = torch.cat([*[1 - x[j]] * j, filling_one, *[1 - x[j]] * (2 - j)], dim=0)
            elif j == 3:
                newx[i * ch + j] = 1 - torch.cat([x[j]] * 3, dim=0)
            else:
                positives = torch.abs((x[j] > 0).double() * x[j])
                negatives = torch.abs((x[j] < 0).double() * x[j])

                newx[i * ch + j] = torch.cat([1 - negatives, 1 - negatives - positives, 1 - positives], dim=0)

    # Add normal image beside channels [normal image, r, g, b, a, ch1, ch2....]
    rgb = to_rgb(xs).permute(0, 3, 1, 2)
    batches = newx.split(ch)
    image = []
    for i in range(len(batches)):
        if i == 0:
            image = torch.cat([rgb[i].unsqueeze(0), batches[i]], dim=0)
        else:
            image = torch.cat([image, rgb[i].unsqueeze(0), batches[i]], dim=0)
    return image


def scale_image(filepath, scale):
    """Scale a saved image"""
    for f in sorted(glob.glob(filepath)):
        with Image.open(f) as im:
            orig_w, orig_h = im.size
            im = im.resize((scale * orig_w, scale * orig_h), resample=Image.BOX)
            im.save(f)


def dmg(img, size, only_top=False, only_bottom=False):
    """Damage an image of size 'size' in a rectangle with lengths between 2 and 4 at random."""
    if only_top and only_bottom:
        raise ValueError("Can't have exclusive damage on both sides")
    x_size = np.random.randint(2, 5)  # 2, 3 or 4
    y_size = np.random.randint(2, 5)
    half_size = size // 2
    if only_top:
        x_start = np.random.randint(0, size - x_size - half_size)
    elif only_bottom:
        x_start = np.random.randint(half_size + 1, size - x_size + 1)
    else:
        x_start = np.random.randint(0, size - x_size + 1)
    y_start = np.random.randint(0, size - y_size + 1)

    img[x_start:x_start + x_size, y_start:y_start + y_size, :] = 0
    return img