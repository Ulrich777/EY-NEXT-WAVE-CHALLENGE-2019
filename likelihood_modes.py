from helper.makedata import load_data
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import pandas as pd
import os
os.chdir("C:/Users/utilisateur/Desktop/LAST_YEAR/EY-NEXT-WAVE-CHALLENGE-2019")


PATH_TO_DATA = './data/'
dG = load_data(pathfile=PATH_TO_DATA, build = False, augment = False)


ANCHORIZE = False

if ANCHORIZE:
  WORDS = dG[['trajectory_id', 'hash','nb','x_entry','y_entry', 'x_exit','y_exit','to_predict','target']].copy()
  del dG
  
  arr = WORDS.loc[WORDS.to_predict==0, ['x_exit','y_exit']].values
  dep = WORDS.loc[:, ['x_entry','y_entry']].values
  
  bw_arr = estimate_bandwidth(arr, quantile=.1, n_samples=1000)
  print(bw_arr)
  bw_arr = 0.001 

  ms_arr = MeanShift(bandwidth=bw_arr, bin_seeding=True, min_bin_freq=5)
  ms_arr.fit(arr)
  cluster_centers_arr = ms_arr.cluster_centers_

  print("Clusters shape: ", cluster_centers_arr.shape)
  np.save(PATH_TO_DATA + 'anchors_nlp_arr.npy', cluster_centers_arr)
  
  WORDS.loc[WORDS.to_predict==0, 'arr'] = ms_arr.labels_
  WORDS.to_csv(PATH_TO_DATA+'nlp_words.csv', index=False)
  
else:
  cluster_centers_arr = np.load(PATH_TO_DATA + 'anchors_nlp_arr.npy')
  WORDS = pd.read_csv(PATH_TO_DATA+'nlp_words.csv')

  
  

