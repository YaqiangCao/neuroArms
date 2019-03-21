# coding: utf-8
"""

"""

#sys
import os, random
from glob import glob
from copy import deepcopy
from datetime import datetime

#3rd
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

#global settings for tensorflow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# hyperparameters
class hyperparameters:
    num_classes = 1
    batch_size = 32
    learning_rate = 3 * 1e-4
    #last dim is channel
    #dims = (992, 1024, 1) #raw size, small image perfomrce better
    #dims = (496, 512, 1)
    dims = (224, 224, 1)
    weight_decay = 0.0005
    train_steps = 1560 / 2
    vali_steps = 156 / 2
    test_steps = 156 / 2
    epochs = 30
    gpus = 2


PARA = hyperparameters()

#modelNames = ["base", "vgg16", "vgg19", "inception", "resnet18", "resnet34", "resnet50", "resnet101", "densenet40","densenet121", "densenet161", "densenetI169", "densenet201", "densenet264"]
modelNames = ["base", "densenet40"]


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


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
        #if "images_aug" in line:
        #    continue
        if len(line) != 2:
            continue
        try:
            readImg(line[0])
            int(line[1])
        except:
            continue
        if int(line[1]) == -1:
            continue
        x.append(line[0])
        y.append(int(line[1]))
    x, y = np.array(x), np.array(y)
    return x, y


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getStat(model, x, y):
    yps = []
    nxs = list(chunks(x, 32))
    for nx in tqdm(nxs):
        #for nx in nxs:
        tmp = []
        for t in nx:
            mat = readImg(t)
            tmp.append(mat)
        tmp = np.array(tmp)
        yp = model.predict(tmp)
        yp = [np.argmax(nt) for nt in yp]
        yps.extend(list(yp))
    mae = mean_absolute_error(y, yps)
    #print("sklearn metrics accuracy MAE", mae)
    mse = mean_squared_error(y, yps)
    #print("sklearn metrics accuracy MSE", mse)
    nyps = [np.rint(yp) for yp in yps]
    acc = accuracy_score(y, nyps)
    #print("sklearn metrics accuracy acc", acc)
    return yps, mae, mse, acc


def test(pref="SDW_Hubor", testf="valiF.txt", suffix="valiF"):
    x, y = getdata(testf)
    #rs = {"x":x,"y_true":y}
    rs = {"y_true": y}
    for name in modelNames:
        cp = "models/%s_%s.h5" % (pref, name)
        if not os.path.isfile(cp):
            continue
        print("loading %s" % cp)
        model = load_model(cp)
        yps, mae, mse, acc = getStat(model, x, y)
        K.clear_session()
        print("\t".join(["model", "MAE", "MSE", "ACC"]))
        print("\t".join(list(map(str, [name, mae, mse, acc]))))
        print("\n")
        rs[name + "_y_pred"] = yps
    rs = pd.DataFrame(rs, index=x)
    rs.to_csv(
        "predictions/%s_%s.txt" % (pref, suffix),
        sep="\t",
        index_label="imageId")


test(pref="InitBce", testf="valiF.txt", suffix="valiF")
