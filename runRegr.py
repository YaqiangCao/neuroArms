# coding: utf-8
"""
2019-01-24: running time record added
2019-01-24: model selection added

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
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, Activation, MaxPooling2D
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# hyperparameters
class hyperparameters:
    num_classes = 1
    batch_size = 32
    learning_rate = 1e-4
    #last dim is channel
    #dims = (992, 1024, 1) #raw size, small image perfomrce better
    #dims = (496, 512, 1)
    dims = (224, 224, 1)
    weight_decay = 0.0005
    train_steps = 1560 / 2
    vali_steps = 156 / 2
    test_steps = 156 / 2
    epochs = 20
    gpus = 2


PARA = hyperparameters()

modelNames = [
    "base", "vgg16", "vgg19", "inception", "resnet18", "resnet34", "resnet50",
    "resnet101", "densenet40", "densenet121", "densenet161", "densenetI169",
    "densenet201", "densenet264"
]


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def BASE(input_shape, classes=None, blocks=7, include_top=False):
    model = Sequential()
    #bachnormalization
    model.add(BatchNormalization(name="input_norm", input_shape=input_shape))
    #block 1
    for i in range(blocks):
        #conv/fc -> batchnorm -> activation-> dropout
        model.add(
            Conv2D(
                2**(3 + i),
                (3, 3),
                padding='same',
                name="block%s_conv1" % i,
            ))
        model.add(BatchNormalization(name="block%s_norm1" % i))
        model.add(Activation("relu", name="block%s_active1" % i))
        model.add(
            Conv2D(
                2**(3 + i),
                (3, 3),
                padding='same',
                name="block%s_conv2" % i,
            ))
        model.add(BatchNormalization(name="block%s_norm2" % i))
        model.add(Activation("relu", name="block%s_active2" % i))
        model.add(
            Conv2D(
                2**(3 + i),
                (3, 3),
                padding='same',
                name="block%s_conv3" % i,
            ))
        model.add(BatchNormalization(name="block%s_norm3" % i))
        model.add(Activation("relu", name="block%s_active3" % i))
        model.add(
            MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2), name='block%s_pool' % i))
    if include_top and classes != None:
        #flatten and dropout
        model.add(Flatten(name="flatten"))
        #model.add(BatchNormalization(name="flatten_norm"))  #new
        #model.add(Dense(units=PARA.num_classes**3, activation='relu', name="fc1"))
        model.add(Dense(units=1024, activation='relu', name="fc1"))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization(name="fc1_norm"))  #new
        #model.add(Dense(units=PARA.num_classes**2, activation='relu', name="fc2"))
        model.add(Dense(units=1024, activation='relu', name="fc2"))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization(name="fc2_norm"))  #new
        model.add(Dense(units=classes, name="dense_final"))
        model.add(Activation("softmax", name="dense_activation"))
    return model


def buildModel(name):
    if name == "base":
        model = BASE(
            input_shape=PARA.dims, include_top=False, classes=PARA.num_classes)
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
    if name == "xception":
        model = Xception(
            include_top=False,
            weights=None,
            input_shape=PARA.dims,
            classes=PARA.num_classes)

    if name == "resnet34":
        model = ResNet34(
            PARA.dims,
            PARA.num_classes,
            include_top=False,
        )
    if name == "wideresnet":
        model = WideResidualNetwork(
            input_shape=PARA.dims, classes=PARA.num_classes, include_top=False)
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
    x = model.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="relu")(x)
    predictions = Dense(1, name="dense_final")(x)
    # creating the final model
    model = Model(input=model.input, output=predictions)
    # use multiple GPU
    #model = multi_gpu_model(model, gpus=PARA.gpus)
    model.compile(
        #loss=huber_loss,
        loss="mae",
        optimizer=Adam(lr=PARA.learning_rate),
        metrics=['mae', "mse", huber_loss],
    )
    return model


def getModel(name=None, checkpoint=None):
    if os.path.isfile(checkpoint):
        print("loading existing model")
        model = load_model(
            checkpoint, custom_objects={"huber_loss": huber_loss})
    else:
        model = buildModel(name)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=2)
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    callbacks = [reduce_lr, early_stop]
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
    for line in tqdm(list(open(f))):
        line = line.split("\n")[0].split("\t")
        if len(line) != 2:
            continue
        try:
            readImg(line[0])
            line[1] = int(line[1])
        except:
            continue
        if line[1] == -1:
            continue
        x.append(line[0])
        #y.append(line[1] + 1)
        y.append(line[1])
    x, y = np.array(x), np.array(y)
    return x, y


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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def getStat(model, x, y, pre):
    yps = []
    nxs = list(chunks(x, PARA.batch_size))
    for nx in tqdm(nxs):
        tmp = []
        for t in nx:
            mat = readImg(t)
            tmp.append(mat)
        tmp = np.array(tmp)
        yp = model.predict(tmp)
        yp = [nt[0] for nt in yp]
        yps.extend(list(yp))
    mae = mean_absolute_error(y, yps)
    #print("sklearn metrics accuracy MAE", mae)
    mse = mean_squared_error(y, yps)
    #print("sklearn metrics accuracy MSE", mse)
    nyps = [np.rint(yp) for yp in yps]
    acc = accuracy_score(y, nyps)
    #print("sklearn metrics accuracy acc", acc)
    rs = pd.Series(data=yps, index=x)
    rs.to_csv("%s_pred.txt" % pre, sep="\t")
    return mae, mse, acc


def train(pre="base"):
    x_train, y_train = getdata("trainF_overlay.txt")
    x_vali, y_vali = getdata("valiF_overlay.txt")
    x_test, y_test = getdata("testF_overlay.txt")
    PARA.train_steps = len(y_train) / PARA.batch_size
    PARA.vali_steps = len(y_vali) / PARA.batch_size
    PARA.test_steps = len(y_test) / PARA.batch_size
    stats = {}
    for name in [
            "base",
            "vgg16",
            "vgg19",
            "inception",
            #"xception",
            "resnet18",
            #"resnet34",
            "resnet50",
            "resnet101",
            "densenet40",
            #"densenet121",
            #"densenet161",
            #"densenetI169",
            #"densenet201",
            #"densenet264"
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
            #class_weight=class_weights,
            validation_data=generator(x_vali, y_vali, PARA.batch_size),
            validation_steps=PARA.vali_steps)
        K.clear_session()
        hist = pd.DataFrame(hist.history)
        hist.to_csv(
            "models/%s_%s_trainningHistroy.txt" % (pre, name),
            sep="\t",
            index_label="epoch")
        #print("------\n" * 3)
        usedTime = datetime.now() - s
        #print(name, cp)
        print("final metrics as following")
        model = load_model(cp, custom_objects={"huber_loss": huber_loss})
        print("train data")
        mtrain = getStat(model, x_train, y_train,
                         "predictions/train_%s" % name)
        print("keras metrics: [MAE,MSE,ACC]", mtrain)
        print("vali data")
        mvali = getStat(model, x_vali, y_vali, "predictions/vali_%s" % name)
        print("keras metrics: [MAE,MSE,ACC]", mvali)
        print("test data")
        mtest = getStat(model, x_test, y_test, "predictions/test_%s" % name)
        print("keras metrics:[MAE,MSE,ACC]", mtest)
        stats[name] = {
            "train_mae": mtrain[0],
            "train_mse": mtrain[1],
            "train_acc": mtrain[2],
            "vali_mae": mvali[0],
            "vali_mse": mvali[1],
            "vali_acc": mvali[2],
            "test_mae": mtest[0],
            "test_mse": mtest[1],
            "test_acc": mtest[2],
            "trainning_time": usedTime,
        }
        print("------\n" * 2)
        K.clear_session()
    stats = pd.DataFrame(stats).T
    stats.to_csv(pre + "_model_loss_acc.txt", sep="\t")


train(pre="MAE")
