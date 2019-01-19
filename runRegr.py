# coding: utf-8
"""

"""

#sys
import os, random
from glob import glob
from copy import deepcopy

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
from keras.layers import Flatten, Dense, Dropout
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from arms.nasnet import NASNet
#networks
from arms.vgg19 import VGG19  #img_rows 224,img_cols 224,channel 3,num_classes if fine-tune,else could be adjusted
from arms.vgg16 import VGG16  #img_rows 224,img_cols 224,channel 3,num_classes
from arms.inception_v3 import InceptionV3
from arms.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from arms.densenet import DenseNet40, DenseNet121, DenseNet161, DenseNetI169, DenseNet201, DenseNet264

#global settings for tensorflow
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# hyperparameters
class hyperparameters:
    num_classes = 1
    batch_size = 8
    learning_rate = 3 * 1e-4
    #last dim is channel
    #dims = (992, 1024, 1) #raw size, small image perfomrce better
    #dims = (496, 512, 1)
    dims = (224, 224, 1)
    weight_decay = 0.0005
    #train_steps = 200
    #epochs = 30
    train_steps = 10
    epochs = 1


PARA = hyperparameters()


def buildModel(name):
    if name == "vgg16":
        model = VGG16(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            #pooling='avg',
            classes=PARA.num_classes)
    if name == "vgg19":
        model = VGG19(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            #pooling='avg',
            classes=PARA.num_classes)
    if name == "nasnet":
        model = NASNet(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            #pooling="avg",
            classes=PARA.num_classes)
    if name == "inception":
        model = InceptionV3(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            #pooling="avg",
            classes=PARA.num_classes)
    if name == "resnet18":
        model = ResNet18(
            PARA.dims,
            PARA.num_classes,
            include_top=False,
        )
    if name == "resnet34":
        model = ResNet34(
            PARA.dims,
            PARA.num_classes,
            include_top=False,
        )
    if name == "resnet50":
        model = ResNet50(PARA.dims, PARA.num_classes, include_top=False)
    if name == "resnet101":
        model = ResNet101(PARA.dims, PARA.num_classes, include_top=False)
    if name == "resnet152":
        model = ResNet152(PARA.dims, PARA.num_classes, include_top=False)
    if name == "densenet40":
        model = DenseNet40(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    if name == "densenet121":
        model = DenseNet121(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    if name == "densenet161":
        model = DenseNet161(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    if name == "densenetI169":
        model = DenseNetI169(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    if name == "densenet201":
        model = DenseNet201(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    if name == "densenet264":
        model = DenseNet264(
            input_shape=PARA.dims,
            classes=PARA.num_classes,
            include_top=False,
            #pooling="avg",
            weights=None)
    #transfer learning part.
    #model.layers.pop()
    #model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []
    #model.add(Dense(units=1, name="dense_final"))
    x = model.output
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    predictions = Dense(1, name="dense_final")(x)

    # creating the final model
    model = Model(input=model.input, output=predictions)

    model.compile(
        #loss="binary_crossentropy",
        loss="mean_squared_error",
        #loss= "categorical_crossentropy",
        optimizer=Adam(lr=PARA.learning_rate),
        #metrics=['accuracy'],
        metrics=['mae', 'acc', "mse"],
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
        line = line.split("\n")[0].split("\t")
        x.append(line[0])
        y.append(int(line[1]))
    x, y = np.array(x), np.array(y)
    x_train, x_vali, y_train, y_vali = train_test_split(
        x, y, test_size=0.2, random_state=123)
    x_vali, x_test, y_vali, y_test = train_test_split(
        x_vali, y_vali, test_size=0.5, random_state=123)
    return x_train, x_vali, x_test, y_train, y_vali, y_test


def generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, PARA.dims[0], PARA.dims[1],
                               PARA.dims[2]))
    batch_labels = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            index = random.randint(0, len(features) - 1)
            x = readImg(features[index])
            if x is None:
                continue
            y = labels[index]
            batch_features[i] = x
            batch_labels[i] = y
        yield batch_features, batch_labels


def train(inputf, pre="base", blocks=7):
    x_train, x_vali, x_test, y_train, y_vali, y_test = getdata(inputf)
    #class_weights = class_weight.compute_class_weight("balanced",
    #                                                  np.unique(y_train),
    #                                                  y_train)
    #print(x_train.shape, x_vali.shape, x_test.shape)
    #print(class_weights)
    stats = {}
    for name in [
            "vgg16",
            "vgg19",
            "nasnet",
            "inception",
            "resnet18",
            "resnet34",
            "densenet40",
            #"resnet50", "resnet101", "resnet152", "densenet40", "densenet121",
            #"densenet161", "densenetI169", "densenet201", "densenet264"
    ]:
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
            #class_weight=class_weights,
            validation_data=generator(x_vali, y_vali, PARA.batch_size),
            validation_steps=20)
        K.clear_session()
        hist = pd.DataFrame(hist.history)
        hist.to_csv(
            "models/%s_%s_trainningHistroy.txt" % (pre, name),
            sep="\t",
            index_label="epoch")
        #print("------\n" * 3)
        print(name, cp)
        print("final metrics as following")
        model = load_model(cp)
        print("train data")
        mtrain = model.evaluate_generator(
            generator(x_train, y_train, PARA.batch_size), steps=20)
        print("keras metrics", model.metrics_names, mtrain)
        print("vali data")
        mvali = model.evaluate_generator(
            generator(x_vali, y_vali, PARA.batch_size), steps=20)
        print("keras metrics", model.metrics_names, mvali)
        print("test data")
        mtest = model.evaluate_generator(
            generator(x_test, y_test, PARA.batch_size), steps=20)
        print("keras metrics", model.metrics_names, mtest)
        stats[name] = {
            "train_loss": mtrain[0],
            "train_acc": mtrain[1],
            "vali_loss": mvali[0],
            "vali_acc": mvali[1],
            "test_loss": mtest[0],
            "test_acc": mtest[1],
        }
        print("------\n" * 3)
        K.clear_session()
    stats = pd.DataFrame(stats).T
    stats.to_csv(pre + "_model_loss_acc.txt", sep="\t")


def test(pref, model, sufix="", save=False):
    print(pref)
    x, y = [], []
    for line in open(pref):
        line = line.split("\n")[0].split("\t")
        x.append(line[0])
        y.append(int(line[1]))
    model = load_model(model)
    hist = {}
    for i, t in enumerate(tqdm(x)):
        mat = readImg(t)
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
    acc = mean_absolute_error(yts, yps)
    print("sklearn metrics accuracy MAE", acc)
    hist = pd.DataFrame(hist).T
    if save:
        hist.to_csv("%s_prob.txt" % sufix, sep="\t")


#train("label.txt")
train("label.txt", pre="base")
#test("label.txt", "base.h5", "test", True)
