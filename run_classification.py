import numpy as np
import pandas as pd
import time
from helper.getfeatures import get_names
from helper.makedata import order_feats
from helper.predictors import build_model, preproc, search_threshold
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score


import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/EY-NEXT-WAVE-CHALLENGE-2019")

PATH_TO_DATA = './data/'


features = get_names(pathfile=PATH_TO_DATA, name='features_norm.csv')
categorical_vars = order_feats(['in_center_entry_0', 'last_target_0', 'stuck_0', 'is_last_0',
                                'time_entry_h_0', 'time_exit_h_0'], max_lag = 4)
categorical_idxs = [features.index(categorical_var) for categorical_var in categorical_vars]
other_idxs = list(range(120))
unique_idxs = [16 if 'time' in categorical_var else 2 for categorical_var in categorical_vars]

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

df_tr = preproc(X_tr, categorical_idxs, other_idxs)
df_ev = preproc(X_ev, categorical_idxs, other_idxs)
df_te = preproc(X_te, categorical_idxs, other_idxs)

NORMALIZE = True

if NORMALIZE:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_tr[-1] = scaler.fit_transform(df_tr[-1])
    df_ev[-1] = scaler.transform(df_ev[-1])
    df_te[-1] = scaler.transform(df_te[-1])
    
full_model =  build_model(unique_idxs, 120, lstm=True)
root = PATH_TO_DATA + 'models/lstm_split'
ckpt = ModelCheckpoint('%s.h5'%(root), save_best_only=True, save_weights_only=True, verbose=1, 
                       monitor='val_f_score', mode='max')
history  =  full_model.fit( df_tr  , y_tr  , epochs =  15 , batch_size = 500, verbose= 1,  
                           validation_data=[df_ev, y_ev], callbacks=[ckpt])#

full_model = build_model(unique_idxs, 120, lstm=True)
full_model.load_weights(PATH_TO_DATA + 'models/lstm_split.h5') 

dt = time.time()

y_pred = full_model.predict(df_ev)[:,0]
y__ = (y_pred>0.5).astype(int)
print(f1_score(y_ev, y__))

dt = time.time() - dt
print('elapsed time: %.4f'%(dt/60))

best_score, best_threshold = search_threshold(y_ev, y_pred, f1_score, breaks=101)
print('eval: ', f1_score(y_ev, (y_pred>best_threshold).astype(int)))

_y_pred = full_model.predict(df_te)[:,0]
_predictions = (_y_pred>best_threshold).astype(int)

dF = pd.read_csv(PATH_TO_DATA + 'sample.csv')
dF['target'] = _predictions
dF.to_csv(PATH_TO_DATA + 'predictions/final_prediction.csv')









