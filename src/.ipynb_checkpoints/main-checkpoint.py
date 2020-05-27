import numpy as np
import pandas as pd
import os
import gc
#import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, AvgPool2D, Dropout, Input, BatchNormalization, AveragePooling2D, Add, MaxPooling2D, Multiply
from tensorflow.keras.layers import Cropping2D, UpSampling2D, Conv2DTranspose, concatenate, Concatenate, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from oct_conv2d import OctConv2D, OctConv2DTranspose

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def normalize(train, test):
    trn_mean = train.mean(axis=(0,1,2))
    trn_std = train.std(axis=(0,1,2))

    train = (train - trn_mean) / trn_std
    test = (test - trn_mean) / trn_std
    return train, test

def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''

    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]  
    
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]
    
    # 실제값이 0.1 이상인 픽셀의 위치 확인
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1
    
    # 실제 값에 결측값이 없는 픽셀의 위치 확인 
    IsNotMissing = y_true >= 0
    
    # mae 계산
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))
    
    # f1_score 계산 위해, 실제값에 결측값이 없는 픽셀에 대해 1과 0으로 값 변환
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)
    
    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)
    
    # f1_score 계산    
    f_score = f1_score(y_true, y_pred) 
    # f1_score가 0일 나올 경우를 대비하여 소량의 값 (1e-07) 추가 
    return mae / (f_score + 1e-07) 

def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    
    y_pred = y_pred.reshape(1, -1)[0]
    
    over_threshold = y_true >= 0.1
    
    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))

def fscore(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    y_true = y_true.reshape(1, -1)[0]
    
    y_pred = y_pred.reshape(1, -1)[0]
    
    remove_NAs = y_true >= 0
    
    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)
    
    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)
    
    return(f1_score(y_true, y_pred))

def maeOverFscore(y_true, y_pred):
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)

def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def score(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float16,  name='custom_mse')
    return score


def create_model():
    inputs=Input(x_train.shape[1:])
    
    bn=BatchNormalization()(inputs)
    conv0=Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(bn)
    
    bn=BatchNormalization()(conv0)
    conv=Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([conv0, conv], axis=3)

    bn=BatchNormalization()(concat)
    conv=Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
    concat=concatenate([concat, conv], axis=3)

    for i in range(5):
        bn=BatchNormalization()(concat)
        conv=Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
        concat=concatenate([concat, conv], axis=3)

    bn=BatchNormalization()(concat)
    o_conv=Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(bn)
    a_conv=Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid')(bn)
    outputs=Multiply()([o_conv, a_conv])
    model=Model(inputs=inputs, outputs=outputs)
    
    return model

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    
    return 1 - (numerator + 1) / (denominator + 1)

def js_divergence(target, pred):
    m = 0.5 * (pred + target)
    loss = loss = 0.5 * losses.kullback_leibler_divergence(pred, m) +  0.5 * losses.kullback_leibler_divergence(target, m)
    return loss

def custom_loss(y_true, y_pred):
    mae_loss = losses.mean_absolute_error(y_true, y_pred)
    y_true, y_pred = tf.math.sigmoid(y_true), tf.math.sigmoid(y_pred)
#     losses.kullback_leibler_divergence(y_true, y_pred)
    return losses.kullback_leibler_divergence(y_true, y_pred) + mae_loss # js_divergence(y_true, y_pred)


bs = 128

def train_model(x_data, y_data, k, s):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=7777)
    
    sp = [x for x in k_fold.split(x_data)]
    train_idx, val_idx = sp[s]
    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_val, y_val = x_data[val_idx], y_data[val_idx]
    model = create_model()
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(x_train, seed=42, batch_size=bs, shuffle=True)
    mask_generator = mask_datagen.flow(y_train, seed=42, batch_size=bs, shuffle=True)
    train_generator = zip(image_generator, mask_generator)
    opt = tf.keras.optimizers.Adam()
    model.compile(loss=custom_loss, optimizer=opt, metrics=[score])
    callbacks_list = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=3,
                factor=0.8
            ),
    tf.keras.callbacks.ModelCheckpoint(
                filepath = '../models/fi_model'+str(s)+'.h5',
                monitor='val_score',
                save_best_only=True
            ),
        ]
    model.fit(train_generator, steps_per_epoch=(len(y_train) // bs), epochs=50, validation_data=(x_val, y_val), callbacks=callbacks_list)
#    model.fit(x_train, y_train, batch_size=bs, epochs=50, validation_data=(x_val, y_val), callbacks=callbacks_list)


# def train_model(x_train, x_test, y_train, y_test):


#         # 데이터를 부풀릴시 많은 양의 메모리가 필요
# #             x_train, y_train = data_generator(x_train, y_train)

#     model = create_model()

#     data_gen_args = dict(horizontal_flip=True, vertical_flip=True)
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)
#     image_generator = image_datagen.flow(x_train, seed=42, batch_size=bs, shuffle=True)
#     mask_generator = mask_datagen.flow(y_train, seed=42, batch_size=bs, shuffle=True)
#     train_generator = zip(image_generator, mask_generator)


#     opt = tf.keras.optimizers.Adam()
# #    opt = tfa.optimizers.SWA(opt, start_averaging=30, average_period=2)
#     model.compile(loss=custom_loss, optimizer=opt, metrics=[score])
# #     model.compile(loss='mae', optimizer=opt, metrics=[score])

#     callbacks_list = [
#             tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_score',
#                 patience=3,
#                 factor=0.8
#             ),

#     tf.keras.callbacks.ModelCheckpoint(
#                 filepath = '../models/fi_model'+str(s)+'.h5',
#                 monitor='val_loss',
#                 save_best_only=True
#             )
#         ]
#     #model.load_weights('../models/dn_v2_model'+str(s)+'.h5')
#     model.fit(train_generator, steps_per_epoch=(len(y_train) // bs), epochs=130, validation_data=(x_val, y_val), callbacks=callbacks_list, verbose=2)
# #    model.fit(x_train, y_train, batch_size=bs, epochs=50, validation_data=(x_val, y_val), callbacks=callbacks_list)


submission = pd.read_csv('../data/sample_submission.csv')
train_files = os.listdir('../data/train')

train = []
for file in train_files:
    try:
        data = np.load('../data/train/'+file).astype('float32')
        train.append(data)
    except:
        continue

test = []
for sub_id in submission['id']:
    data = np.load('../data/test/'+'subset_'+sub_id+'.npy').astype('float32')
    test.append(data)

train = np.array(train)
test = np.array(test)

x_train = train[:,:,:,:10]
x_train = np.concatenate((x_train, np.expand_dims(train[:,:,:,10] - train[:,:,:,12], axis=-1)), axis=-1)
x_train = np.concatenate((x_train, np.expand_dims(train[:,:,:,11] - train[:,:,:,13], axis=-1)), axis=-1)
y_train = train[:,:,:,14]
x_test = test[:,:,:,:10]
x_test = np.concatenate((x_test, np.expand_dims(test[:,:,:,10] - test[:,:,:,12], axis=-1)), axis=-1)
x_test = np.concatenate((x_test, np.expand_dims(test[:,:,:,11] - test[:,:,:,13], axis=-1)), axis=-1)


# x_train = x_train[np.sum(y_train.reshape(-1, 1600)>=0, 1)==1600]
# y_train = y_train[np.sum(y_train.reshape(-1, 1600)>=0, 1)==1600]


# y_train_ = y_train.reshape(-1,y_train.shape[1]*y_train.shape[2])

# x_train = np.delete(x_train, np.where(y_train_<0)[0], axis=0)
# y_train = np.delete(y_train, np.where(y_train_<0)[0], axis=0)
# y_train = y_train.reshape(-1, x_train.shape[1], x_train.shape[2],1)

# y_train_ = np.delete(y_train_, np.where(y_train_<0)[0], axis=0)

x_train, x_test = normalize(x_train, x_test)

del train
del test
gc.collect()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.025, random_state=7777)


k = 5

train_model(x_train, y_train, k=k, s=0)
#train_model(x_train, y_train, k=k, s=1)
#train_model(x_train, y_train, k=k, s=2)
#train_model(x_train, y_train, k=k, s=3)
# train_model(x_train, y_train, k=k, s=4)

