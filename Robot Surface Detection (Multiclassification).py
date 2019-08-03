# -*- coding: utf-8 -*-
"""
Created on Sun Apr  21 18:21:35 2019

@author: aaronliu
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Activation, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import preprocessing
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import datetime
from time import time
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from matplotlib import rcParams
from keras import losses
from keras import layers
import re
from seaborn import countplot,lineplot, barplot
from numba import jit
from sklearn import preprocessing
from scipy.stats import randint as sp_randint
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import math
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import gc
gc.enable()
from sklearn.feature_selection import VarianceThreshold 
from sklearn import preprocessing

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data = pd.read_csv('C:\\Users\\user\\Desktop\\W&M BA Fall\\Kaggle\\career-con-2019\\X_train.csv')
target = pd.read_csv('C:\\Users\\user\\Desktop\\W&M BA Fall\\Kaggle\\career-con-2019\\y_train.csv')
data.describe()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Visualization & Exploration Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# graph 1 shows the dirtribution of dependent variable (target)
plt.figure(figsize=(15, 5))
sns.countplot(target['surface'])
plt.title('Target distribution', size=15)
plt.show()

# graph 2 shows the distribution among the given 10 original independent variables (attributes)
plt.figure(figsize=(26, 16))
for i,col in enumerate(aux.columns[3:13]):
    ax = plt.subplot(2,5,i+1)
    ax = plt.title(col)
    for surface in classes:
        surface_feature = aux[aux['surface'] == surface]
        sns.kdeplot(surface_feature[col], label = surface)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Preprocessing Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Feature Engineering 
# write function to generate euler angles from given quaternion (orientation x, y, z, w)
start_time_1 = datetime.datetime.now()
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z     

# normalize orientation x, y, z, w
def fe_step0 (actual):
    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html
    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html
    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html
    
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual

data = fe_step0(data)

# generate euler angles from given quaternion (orientation x, y, z, w)
def fe_step1 (actual):
    """Quaternions to Euler Angles"""
    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual

data = fe_step1(data)

# feature engineering with the attributs, including mean, maximum, minimum, standard deviation and so on
def feat_eng(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                             data['angular_velocity_Z'])** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z'])**0.5
    #data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +
                             #data['orientation_Z'])**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df

data = feat_eng(data)
stop_time_1 = datetime.datetime.now()
print ("Time required for feature engineering:",stop_time_1 - start_time_1)
print (data.shape)
data.head()

# deal with NaN
data.fillna(0,inplace=True)
data.replace(-np.inf,0,inplace=True)
data.replace(np.inf,0,inplace=True)

# save the data in csv form for future use after doing feature engineering
data.to_csv("afterFE_train.csv")


# load the data with feature engineering
dataFE = pd.read_csv('C:\\Users\\user\\Desktop\\W&M BA Fall\\Kaggle\\robot\\experiment\\afterFE_train.csv')
dataFE.head()
dataFE.shape

# scale the data for fitting neural network model
sc = StandardScaler()
dataFE = sc.fit_transform(dataFE)

# transform our target (surface) from string form to numeric form for prediction
target.head()
target['surface'] = le.fit_transform(target['surface'])
target['surface'].value_counts()
target.head()

# split the data into 80/20 proportion
X_train, X_test, y_train, y_test = train_test_split(dataFE, target['surface'], test_size=0.2, random_state=42)
X_train.shape
X_test.shape

# split training data to training set and validation set 
X_val = X_train[:1000]
X_train = X_train[1000:]
y_val = y_train[:1000]
y_train = y_train[1000:]
X_val.shape
X_train.shape
#
num_classes = 9
y_train = np_utils.to_categorical(y_train, num_classes)
y_val = np_utils.to_categorical(y_val, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

y_train.shape # (2048, 9)
y_val.shape # (1000, 9)
y_test.shape # (762, 9)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Loss Function Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# define loss function to deal with imbalanced data (much less data for some surface, such as hard tiles)
def focal_loss(gamma=2, alpha=0.25):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

def nbAaronModel_final():
    np.random.seed(7)  
    model = Sequential()
    model.add(Dense(1024, input_dim=210,  activation='relu'))
    model.add(Dense(512,activation = 'relu'))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(32,activation = 'relu'))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dense(9,activation = 'softmax'))
    model.compile(loss=focal_loss(alpha=6), optimizer= 'adam'
                  , metrics = ['accuracy'], )
    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fit Model Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model = nbAaronModel_final()
history_final = model.fit(X_train,y_train, epochs = 50
                          , batch_size = 40
                          , validation_data = (X_val,y_val))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show Output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
scores_final = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%"%(model.metrics_names[1],scores_final[1]*100))

# count the time of execution
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

#plot train and test loss, acc in related to train iterations(epochs) 
acc = history_final.history['acc']
val_acc = history_final.history['val_acc']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()