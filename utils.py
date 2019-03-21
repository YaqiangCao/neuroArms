#3rd
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# hyperparameters
class hyperparameters:
    num_classes = 1
    batch_size = 32
    learning_rate = 1e-4
    dims = (224, 224, 1)
    weight_decay = 0.0005
    train_steps = 1000
    vali_steps = 100
    test_steps = 100
    epochs = 30
    gpus = 2


PARA = hyperparameters()


def readImg(f, norm=True, reshape=True):
    img = Image.open(f)
    if img.mode != 'L':
        img = img.convert("L")
    img = np.array(img.resize((PARA.dims[0], PARA.dims[1])))
    if reshape:
        img = img.reshape(PARA.dims[0], PARA.dims[1], 1)
    if norm:
        img = img / 255.0
        std = np.std(img, ddof=1)
        mean = np.mean(img)
        img = (img - mean) / max(std, 1. / PARA.dims[0])
    return img


def getdata(f):
    x, y = [], []
    for line in open(f):
        line = line.split("\n")[0].split("\t")
        if len(line) != 2:
            continue
        try:
            readImg(line[0])
            int(line[1])
        except:
            continue
        x.append(line[0])
        y.append(int(line[1]))
    x, y = np.array(x), np.array(y)
    return x, y


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
