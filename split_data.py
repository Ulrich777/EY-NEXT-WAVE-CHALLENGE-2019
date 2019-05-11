from helper.getfeatures import getdata
import time
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/EY-NEXT-WAVE-CHALLENGE-2019")

BUILD = True

PATH_TO_DATA = './data/'

if BUILD:

    print('loading features')

    MAX_LAG = 4

    dt = time.time()

    dG, features = getdata(pathfile=PATH_TO_DATA, 
                                      build=True, 
                                      augment=False, 
                                      max_lag=MAX_LAG, 
                                      start=True,
                                      extra=False)
    
    
    dt = time.time() - dt
    print('elapsed time: %.4f'%(dt/60))

    print('splitting....')
    Features = features + ['x_exit','y_exit']
    X_tr, X_ev, y_tr, y_ev = train_test_split(dG.loc[dG.to_predict==0, Features].values, 
                                          dG.loc[dG.to_predict==0, 'target'].values, test_size=0.1, random_state=42,
                                           stratify=dG.loc[dG.to_predict==0, 'target'].values)
    X_te = dG.loc[dG.to_predict==1, features].values

    print('saving X_tr')
    np.save(PATH_TO_DATA+'twosteps/X_tr.npy', X_tr)
    print('saving X_tv')
    np.save(PATH_TO_DATA+'twosteps/X_ev.npy', X_ev)
    print('saving X_te')
    np.save(PATH_TO_DATA+'twosteps/X_te.npy', X_te)
    print('saving y_tr')
    np.save(PATH_TO_DATA+'twosteps/y_tr.npy', y_tr)
    print('saving y_ev')
    np.save(PATH_TO_DATA+'twosteps/y_ev.npy', y_ev)
else:
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
    
    

