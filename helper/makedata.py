import pandas as pd
import time
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix#, accuracy_score


#=== CONSTANTS =============================#
SCALE = 10**6
X_UP = 3770901.5068 / SCALE
X_DOWN = 3750901.5068 / SCALE
Y_DOWN = -19268905.6133 / SCALE
Y_UP = -19208905.6133 / SCALE

MEAN_X = (X_DOWN + X_UP)/2
DEV_X = (X_UP - X_DOWN)/2

MEAN_Y = (Y_DOWN + Y_UP)/2
DEV_Y = (Y_UP - Y_DOWN)/2

#===== set of helper functions ========#
def time_convert(x):
    h,m,s = map(int,x.split(':'))
    return (h*60+m)*60+s

def HourDetail(hour, tag='h'):
    if tag=='h':
        return int(hour[:2])
    elif tag=='m':
        return int(hour[3:5])
    else:
        return int(hour[6:8])

def get_target(y):
    y_center = (y - np.array([MEAN_X, MEAN_Y]))/np.array([DEV_X, DEV_Y])
    y_center = (np.abs(y_center)<=1)
    y_ = y_center.sum(axis=1)
    y_ = (y_==2).astype(int)
    return y_
  
def detail(y_true, y_pred):
    print('F-score is ', f1_score(y_true, y_pred))
    print('\n====== Classification report============')
    print(classification_report(y_true, y_pred))
    print('\n======Confusion matrix=======')
    print(confusion_matrix(y_true, y_pred))
    
#def DistToAvg(x,y):
#  return np.sqrt((x - MEAN_X)**2 + (y-MEAN_Y)**2)

def DistToAvg(y):
  y_center = y - np.array([MEAN_X, MEAN_Y])
  return np.linalg.norm(y_center, axis=1)

def resize_xy(y):
  y_center = (y - np.array([MEAN_X, MEAN_Y]))/np.array([DEV_X, DEV_Y])
  return np.abs(y_center)


#============ load the data ========================
def load_data(pathfile='', build = True, augment = False):
    
    if build:
  
        dt = time.time()

        print('loading train')
        df = pd.read_csv(pathfile+'data_train.csv')
        df = df.drop('Unnamed: 0', axis=1)
        df['origin'] = 'train'

        print('loading test')
        dg = pd.read_csv(pathfile+'data_test.csv')
        dg = dg.drop('Unnamed: 0', axis=1)
        dg['origin'] = 'test'

        print('merging')
        df = df.append(dg)

        print('deleting speed measurements')
        df = df.drop(['vmin', 'vmax', 'vmean'], axis=1)

        print('making numbers')
        df['nb'] = 1
        df['nb'] = df.groupby('hash')['nb'].cumsum()
        
        if augment:
            df['nb'] = 2 * (df['nb']-1)

            print('imputing new departure')
            dD = df[['hash', 'trajectory_id', 'time_exit', 'x_exit', 'y_exit', 'origin', 'nb']].copy()
            dD.loc[:,'nb'] = dD.loc[:,'nb'] + 1

            print('imputing new arrival')
            dA = df[['hash', 'trajectory_id', 'time_entry', 'x_entry','y_entry', 'origin', 'nb']].copy()
            dA.loc[:,'nb'] = dA.loc[:,'nb'] - 1

            print('merging new lines')
            dG = dA.merge(dD, on=['hash', 'nb'], how='inner')
            del dA, dD
            dG.drop(['trajectory_id_x', 'origin_x'], axis=1, inplace=True)
            dG.columns = ['hash', 'time_exit', 'x_exit', 'y_exit','nb', 'trajectory_id', 'time_entry', 'x_entry', 'y_entry','origin']
            dG['trajectory_id'] += 'b'

            print('final dataset')
            df = df.append(dG)
            df.sort_values(['hash', 'nb'], inplace=True)
            del dG
    
        print('building time')
        df['time_entry_num'] = df.time_entry.apply(time_convert) / 3600
        df['time_exit_num'] = df.time_exit.apply(time_convert) / 3600
        df['duration'] = df['time_exit_num'] - df['time_entry_num']
    

        print('scaling coordinates')
        df[['x_entry', 'y_entry', 'x_exit', 'y_exit']] = df[['x_entry', 'y_entry', 'x_exit', 'y_exit']]/SCALE

        print('identifing lines to forecast')
        df['to_predict'] = 0
        df.loc[(df.origin=='test')&df.x_exit.isnull()& df.y_exit.isnull(), 'to_predict'] = 1
        df.loc[df.to_predict==0, 'target'] = get_target(df.loc[df.to_predict==0, ['x_exit','y_exit']].values)
    
        if augment:
            print('new lines identifier')
            df['recovered'] = ((df['nb']%2)==1).astype(int)
        print('no movement ')
        df['stuck'] = (df['time_entry']==df['time_exit']).astype(int)
        print('deterministic exit position predictions for test stuck lines')
        df.loc[(df.to_predict==1) & (df.stuck==1), ['x_exit', 'y_exit']] = df.loc[(df.to_predict==1) & (df.stuck==1), ['x_entry', 'y_entry']].values
        print('deterministic test predictions')
        df.loc[(df.to_predict==1) & (df.stuck==1), 'target'] = get_target(df.loc[(df.to_predict==1) & (df.stuck==1), ['x_exit', 'y_exit']].values )
    
        print('generating distance to average')
        #df['dist_to_avg'] = df.apply(lambda row : DistToAvg(row['x_entry'], row['y_entry']), axis=1)
        df['dist_to_avg'] = DistToAvg(df.loc[:,['x_entry', 'y_entry']].values)
        print('Identifying trajectory starting in the center')
        df['in_center_entry'] = get_target(df[['x_entry','y_entry']].values)
    
        #columns = list(df.columns)
        #if augment:
            #columns[columns.index('x_entry')] = 'x_it'
            #columns[columns.index('y_entry')] = 'y_it'
            #columns[columns.index('x_exit')] = 'Fx_it'
            #columns[columns.index('y_exit')] = 'Fy_it'
            #df.columns = columns   


        print('extracting hour')
        df['time_entry_h'] = df.time_entry.apply(lambda hour: HourDetail(hour, 'h'))
        df['time_exit_h'] = df.time_exit.apply(lambda hour: HourDetail(hour, 'h'))

        #print('extracting minute')
        #df['time_entry_m'] = df.time_entry.apply(lambda hour: HourDetail(hour, 'm'))
        #df['time_exit_m'] = df.time_exit.apply(lambda hour: HourDetail(hour, 'm'))

        #print('extracting seconds')
        #df['time_entry_s'] = df.time_entry.apply(lambda hour: HourDetail(hour, 's'))
        #df['time_exit_s'] = df.time_exit.apply(lambda hour: HourDetail(hour, 's'))
        print('help features')
        df["x_min_entry"] =  X_DOWN*(df.x_entry < X_DOWN) + \
                                df.x_entry*((X_DOWN <= df.x_entry) & (df.x_entry <= X_UP) )\
                                + X_UP*(df.x_entry > X_UP)
        df["y_min_entry"] = Y_DOWN*(df.y_entry < Y_DOWN) + \
                                df.y_entry*((Y_DOWN <= df.y_entry) & (df.y_entry <=  Y_UP) )\
                                +  Y_UP*(df.y_entry >  Y_UP)
        df["x_min_exit"] =  X_DOWN*(df.x_exit < X_DOWN) + \
                                df.x_exit*((X_DOWN <= df.x_exit) & (df.x_exit <= X_UP) )\
                                + X_UP*(df.x_exit > X_UP)
        df["y_min_exit"] = Y_DOWN*(df.y_exit < Y_DOWN) + \
                                df.y_exit*((Y_DOWN <= df.y_exit) & (df.y_exit <=  Y_UP) )\
                                +  Y_UP*(df.y_exit >  Y_UP)
         
        df["dist_min_entry"] = (df["x_entry"] - df["x_min_entry"]).abs() + (df["y_entry"] - df["y_min_entry"]).abs()
        df["dist_min_exit"] = (df["x_exit"] - df["x_min_exit"]).abs() + (df["y_exit"] - df["y_min_exit"]).abs()

                       
        df["dist_exit"] = (df["x_exit"] - df["x_entry"]).abs() + (df["y_exit"] - df["y_entry"]).abs()
        df["speed_exit"] = df["dist_exit"]/(df["duration"] + 0.1)
        df["speed_min_entry"] = df["dist_min_entry"]/(df["duration"] + 0.1)
        df["speed_min_exit"] = df["dist_min_exit"]/(df["duration"] + 0.1)
        
        if augment:
            df['nb'] = 1 + df['nb']/2
    
        print('saving')
        if augment:
            df.to_csv(pathfile+'data_aug.csv', index=False) 
        else:
            df.to_csv(pathfile+'data_help.csv', index=False) 
    
        dt = time.time() - dt
        print('elapsed time: %.4f'%(dt/60))
    
    else:
        dt = time.time()
        print('loading the augmented data')
        if augment:
            df = pd.read_csv(pathfile+'data_aug.csv')
        else:
            df = pd.read_csv(pathfile+'data_help.csv')
        dt = time.time() - dt
        print('elapsed time: %.4f'%(dt/60))
        
    return df

##==== adding lag for LSTM, CNN etc.. ============#
def add_lag(df, leave, idxs, lags=[1,2]):
  dG = df.copy()
  cols = [col for col in df.columns if col not in leave]
  for lag in lags:
    print(lag)
    dF = df[cols].copy()
    
    cols_1 = [col[:-2]+'_'+str(lag) if col not in ['hash','nb'] else col for col in cols]
    #print(cols_1)
    
    dF.columns = cols_1
    dF['nb'] += lag
    dG = dG.merge(dF, on = idxs, how='left')
    #"""
    for col in cols:
      if col not in idxs + ['duration_0'] and 'time_exit' not in col:
        dG.loc[dG[col[:-2]+'_'+str(lag)].isnull(), col[:-2]+'_'+str(lag)] = dG.loc[dG[col[:-2]+'_'+str(lag)].isnull(), col]
        
      if col=='duration_0':
        dG.loc[dG[col[:-2]+'_'+str(lag)].isnull(), col[:-2]+'_'+str(lag)] = 0.
      if 'time_exit' in col:
        print(col, col.replace('exit','entry'))
        dG.loc[dG[col[:-2]+'_'+str(lag)].isnull(), col[:-2]+'_'+str(lag)] = dG.loc[dG[col[:-2]+'_'+str(lag)].isnull(), col.replace('exit','entry')]
    #"""
  return dG

##====== getting columns ========##
def order_feats(feats, max_lag = 2, forward=False):
    features = feats.copy()
    
    if forward:
        for lag in range(0, max_lag):
            feats_lag = [feat[:-2]+'_'+str(lag) for feat in feats]
            features += feats_lag 
    else:
        for lag in range(1, max_lag+1):
            feats_lag = [feat[:-2]+'_'+str(lag) for feat in feats]
            features += feats_lag
        
    return features


def ForwardToLast(dG, idxs, forwards):
    dF = dG[idxs+forwards].copy()
    new_cols = ['last_'+col if col not in ['hash','nb'] else col for col in dF.columns]
    dF.columns = new_cols
    dF['nb'] += 1
    dG = dG.merge(dF, on = idxs, how='left')
    
    for col in forwards:
        if col not in ['target','dist_exit','speed_exit']:
            dG.loc[dG['last_'+col].isnull(),'last_'+col] = dG.loc[dG['last_'+col].isnull(),col.replace('exit','entry')]
        elif col=='target':
            dG.loc[dG['last_'+col].isnull(),'last_'+col] = dG.loc[dG['last_'+col].isnull(),col.replace('exit','in_center_entry')]
        else:
            dG.loc[dG['last_'+col].isnull(),'last_'+col] = 0.
        
    
    return dG
    
def GetLag(dG, idxs, forwards):
    dF = dG[idxs+forwards].copy()
    new_cols = ['last_'+col if col not in ['hash','nb'] else col for col in dF.columns]
    dF.columns = new_cols
    dF['nb'] += 1
    dG = dG.merge(dF, on = idxs, how='left')
    
    for col in forwards:
        dG.loc[dG['last_'+col].isnull(),'last_'+col] = 0.
        
    
    return dG 
  