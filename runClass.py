# coding: utf-8
"""
2019-01-24: running time record added
2019-01-24: model selection added
2019-01-25: treated as classification problem.

"""

#sys
import os, random
from glob import glob
from copy import deepcopy
from datetime import datetime

#3rd
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from arms.nasnet import NASNet
#networks
from arms.vgg19 import VGG19  #img_rows 224,img_cols 224,channel 3,num_classes if fine-tune,else could be adjusted
from arms.vgg16 import VGG16  #img_rows 224,img_cols 224,channel 3,num_classes
from arms.inception_v3 import InceptionV3
from arms.xception import Xception
from arms.wide_resnet import WideResidualNetwork
from arms.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from arms.densenet import DenseNet40, DenseNet121, DenseNet161, DenseNetI169, DenseNet201, DenseNet264

#global settings for tensorflow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# hyperparameters
class hyperparameters:
    num_classes = 21
    batch_size = 32
    learning_rate = 3 * 1e-4
    #last dim is channel
    #dims = (992, 1024, 1) #raw size, small image perfomrce better
    #dims = (496, 512, 1)
    dims = (224, 224, 1)
    weight_decay = 0.0005
    train_steps = 200
    vali_steps = 20
    test_steps = 20
    epochs = 30
    gpus = 2


PARA = hyperparameters()


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def buildModel(name):
    if name == "vgg16":
        model = VGG16(
            include_top=True,
            weights=None,
            input_shape=PARA.dims,
            #pooling='avg',
            classes=PARA.num_classes)
    if name == "vgg19":
        model = VGG19(
            include_top=True,
            weights=None,
            input_shape=PARA.dims,
            #pooling='avg',
            classes=PARA.num_classes)
    if name == "nasnet":
        model = NASNet(
            include_top=True,
            weights=None,
            input_shape=PARA.dims,
            #pooling="avg",
            classes=PARA.num_classes)
    if name == "inception":
        model = InceptionV3(
            include_top=True,
            weights=None,
            input_shape=PARA.dims,
            #pooling="avg",
            classes=PARA.num_classes)
    if name == "xception":
        model = Xception(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            #pooling="avg",
            classes=PARA.num_classes)
    if name == "resnet18":
        model = ResNet18(
            PARA.dims,
            PARA.num_classes,
            include_top=True,
        )
    if name == "resnet34":
        model = ResNet34(
            PARA.dims,
            PARA.num_classes,
            include_top=True,
        )
    if name == "resnet50":
        model = ResNet50(PARA.dims, PARA.num_classes, include_top=True)
    if name == "resnet101":
        model = ResNet101(PARA.dims, PARA.num_classes, include_top=True)
    if name == "resnet152":
        model = ResNet152(PARA.dims, PARA.num_classes, include_top=True)
    if name == "wideresnet":
        model = WideResidualNetwork(
            input_shape=PARA.dims, classes=PARA.num_classes, include_top=True)
    if name == "densenet40":
        model = DenseNet40(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)
    if name == "densenet121":
        model = DenseNet121(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)
    if name == "densenet161":
        model = DenseNet161(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)
    if name == "densenetI169":
        model = DenseNetI169(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)
    if name == "densenet201":
        model = DenseNet201(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)
    if name == "densenet264":
        model = DenseNet264(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=True,
            #pooling="avg",
            weights=None)

    # use multiple GPU
    model = multi_gpu_model(model, gpus=PARA.gpus)
    model.compile(
        loss="binary_crossentropy",
        #loss="mean_absolute_error",
        #loss= "categorical_crossentropy",
        optimizer=Adam(lr=PARA.learning_rate),
        #metrics=['accuracy'],
        metrics=['mae', "mse","acc"],
    )
    return model


def getModel(name=None, checkpoint=None):
    if os.path.isfile(checkpoint):
        print("loading existing model")
        model = load_model(checkpoint, )
    else:
        model = buildModel(name)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=5)
    callbacks = [reduce_lr]
    #print(model.summary())
    if checkpoint is not None:
        cp = ModelCheckpoint(
            checkpoint,
            monitor='val_loss',
            #monitor='val_acc',
            verbose=1,
            save_weights_only=False,
            save_best_only=True,
            mode='min')
        #mode='max')
        callbacks.append(cp)
    return model, callbacks


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
        #do not use augmentation data for testing
        if "images_aug" in line:
            continue
        line = line.split("\n")[0].split("\t")
        if len(line) != 2:
            continue
        try:
            int(line[1])
        except:
            continue
        x.append(line[0])
        y.append(int(line[1])+1) 
    x, y = np.array(x), np.array(y)
    return x, y
    """
    x_train, x_vali, y_train, y_vali = train_test_split(
        x, y, test_size=0.2, random_state=123)
    x_vali, x_test, y_vali, y_test = train_test_split(
        x_vali, y_vali, test_size=0.5, random_state=123)
    return x_train, x_vali, x_test, y_train, y_vali, y_test
    """


def generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, PARA.dims[0], PARA.dims[1],
                               PARA.dims[2]))
    batch_labels = np.zeros((batch_size, PARA.num_classes))
    while True:
        for i in range(batch_size):
            index = random.randint(0, len(features) - 1)
            x = readImg(features[index])
            if x is None:
                continue
            y = np.zeros(PARA.num_classes)
            y[labels[index]] = 1
            batch_features[i] = x
            batch_labels[i] = y
        #print(batch_features.shape,batch_labels.shape)
        yield batch_features, batch_labels


def train(pre="base"):
    x_train, y_train = getdata("trainF.txt")
    print(x_train[:10])
    print(y_train[:10])
    x_vali, y_vali = getdata("valiF.txt")
    x_test, y_test = getdata("testF.txt")
    #x_train, x_vali, x_test, y_train, y_vali, y_test = getdata(inputf)
    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(y_train),
                                                      y_train)
    #print(x_train.shape, x_vali.shape, x_test.shape)
    #print(class_weights)
    stats = {}
    for name in [
            "vgg16",
            "vgg19",
            #"nasnet",
            "inception",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            #"wideresnet",
            "xception",
            "densenet40",
            "densenet121",
            "densenet161", 
            "densenetI169", 
            "densenet201", 
            "densenet264"
    ]:
        s = datetime.now()
        cp = "models/%s_%s.h5" % (pre, name)
        print("starting ", name, cp)
        model, callbacks = getModel(name, cp)
        hist = model.fit_generator(
            generator(x_train, y_train, PARA.batch_size),
            callbacks=callbacks,
            epochs=PARA.epochs,
            steps_per_epoch=PARA.train_steps,
            shuffle=True,
            use_multiprocessing=True,
            class_weight=class_weights,
            validation_data=generator(x_vali, y_vali, PARA.batch_size),
            validation_steps=PARA.vali_steps)
        K.clear_session()
        usedTime = datetime.now() - s
        hist = pd.DataFrame(hist.history)
        hist.to_csv(
            "models/%s_%s_trainningHistroy.txt" % (pre, name),
            sep="\t",
            index_label="epoch")
        #print("------\n" * 3)
        print(name, cp)
        print("final metrics as following, used time:%s"%usedTime)
        model = load_model(cp)
        print("train data")
        mtrain = model.evaluate_generator(
            generator(x_train, y_train, PARA.batch_size), steps=PARA.train_steps)
        print("keras metrics", model.metrics_names, mtrain)
        print("vali data")
        mvali = model.evaluate_generator(
            generator(x_vali, y_vali, PARA.batch_size), steps=PARA.vali_steps)
        print("keras metrics", model.metrics_names, mvali)
        print("test data")
        mtest = model.evaluate_generator(
            generator(x_test, y_test, PARA.batch_size), steps=PARA.test_steps)
        print("keras metrics", model.metrics_names, mtest)
        stats[name] = {
            "train_loss": mtrain[0],
            "train_mae": mtrain[1],
            "train_mse": mtrain[2],
            "train_acc": mtrain[3],
            "vali_loss": mvali[0],
            "vali_mae": mvali[1],
            "vali_mse": mvali[2],
            "vali_acc": mvali[3],
            "test_loss": mtest[0],
            "test_mae": mtest[1],
            "test_mse": mtest[2],
            "test_acc": mtest[3],
            "trainning_time": usedTime,
        }
        print("------\n" * 2)
        K.clear_session()
    stats = pd.DataFrame(stats).T
    stats.to_csv(pre + "_model_loss_acc.txt", sep="\t")


def test(pref, model, sufix="", save=False):
    print(pref)
    x, y = [], []
    for line in open(pref):
        line = line.split("\n")[0].split("\t")
        x.append(line[0])
        try:
            y.append(int(line[1]))
        except:
            y.append(0)
    x = x[:10000]
    y = y[:10000]
    model = load_model(model)
    hist = {}
    for i, t in enumerate(tqdm(x)):
        try:
            mat = readImg(t)
        except:
            continue
        yp = model.predict(np.array([mat]))[0][0]
        n = "_".join(t.split("_")[:-1])
        #yp = np.rint(yp)
        if n not in hist:
            hist[n] = {"y_true": [], "y_pred": []}
        hist[n]["y_true"].append(y[i])
        hist[n]["y_pred"].append(yp)
    yps = []
    yts = []
    for key in hist.keys():
        yp = np.rint(np.mean(hist[key]["y_pred"]))
        yt = np.rint(np.mean(hist[key]["y_true"]))
        #print(key,yt,yp)
        yps.append(yp)
        yts.append(yt)
        hist[key]["y_pred"] = yp
        hist[key]["y_true"] = yt
    mae = mean_absolute_error(yts, yps)
    print("sklearn metrics accuracy MAE", mae)
    mse = mean_squared_error(yts, yps)
    print("sklearn metrics accuracy MSE", mse)
    hist = pd.DataFrame(hist).T
    if save:
        hist.to_csv("%s_prob.txt" % sufix, sep="\t")
    K.clear_session()


train(pre="base")
