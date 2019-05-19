import numpy as np
import time
from helper.predictors import model_reg
from helper.makedata import get_target
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score

import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/EY-NEXT-WAVE-CHALLENGE-2019")

PATH_TO_DATA = './data/'

print('loading X_tr')
X_tr = np.load(PATH_TO_DATA+'twosteps/X_tr.npy')
print('loading X_te')
X_te = np.load(PATH_TO_DATA+'twosteps/X_te.npy')
print('loading X_ev')
X_ev = np.load(PATH_TO_DATA+'twosteps/X_ev.npy')
print('loading y_tr')
y_tr = np.load(PATH_TO_DATA+'twosteps/y_tr.npy')
print('loading y_ev')
y_ev = np.load(PATH_TO_DATA+'twosteps/y_ev.npy')

NORMALIZE = True

if NORMALIZE:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr[:,:-2] = scaler.fit_transform(X_tr[:,:-2])
    X_ev[:,:-2] = scaler.fit_transform(X_ev[:,:-2])
    X_te = scaler.transform(X_te)
    

cluster_centers_arr = np.load(PATH_TO_DATA + 'anchors_nlp_arr.npy')

full_model =  model_reg(120, cluster_centers_arr, lstm=True, time=5, feats=24)
root = PATH_TO_DATA + 'models/lstm_reg_rand'
ckpt = ModelCheckpoint('%s.h5'%(root), save_best_only=True, save_weights_only=True, 
                       verbose=1, monitor='val_loss', mode='min')

history  =  full_model.fit( X_tr[:,:-2]  , X_tr[:,-2:]  , epochs =  20 , batch_size = 500, verbose= 1,  
                           validation_data=[X_ev[:,:-2], X_ev[:,-2:]], callbacks=[ckpt])#


full_model =  model_reg(120, cluster_centers_arr, lstm=True, time=5, feats=24)
full_model.load_weights(PATH_TO_DATA + 'models/lstm_reg_rand.h5')

# Predictions
dt = time.time()

pos_ev = full_model.predict(X_ev[:,:-2], batch_size=50000)
pos_tr = full_model.predict(X_tr[:,:-2], batch_size=50000)
pos_te = full_model.predict(X_te)


dt = time.time() - dt
print('elapsed time: %.4f'%(dt/60))


# Scoring
print('eval: ',f1_score(y_ev, get_target(pos_ev)))
print('train: ', f1_score(y_tr, get_target(pos_tr)))










