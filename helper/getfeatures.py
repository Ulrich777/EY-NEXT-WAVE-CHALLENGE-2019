from makedata import load_data, ForwardToLast, add_lag, order_feats
import numpy as np
import pandas as pd

def get_day(x):
    x = x.split("_")[2]
    
    x = x if len(x) > 1 else "0" + x
    
    return "2018-10-%s"%x

def save_names(features, pathfile='', name='features'):
    df = pd.DataFrame({'names':features})
    df.to_csv(pathfile+name, index = False)
    
def get_names(pathfile='', name='features'):
    df = pd.read_csv(pathfile+name)
    return df['names'].values.tolist()

def getdata(pathfile='', build=True, augment=False, max_lag=4, start=False, extra=True):
    
    if augment:
        suff = 'aug'
    else:
        suff = 'norm'
    if start:
        dG = load_data(pathfile=pathfile, build = build, augment = augment)
        
        if extra:
            print('loading temperature data')
            temp = pd.read_csv(pathfile+'atlanta_meteo.csv')
        
            print('building day')
            dG['day'] = dG.trajectory_id.apply(get_day) 
            dG["date_time_entry"] =  dG.trajectory_id.apply(get_day) + ' ' + dG["time_entry"]
        
            print('convert to date format')
            dG['date_time_entry'] = pd.to_datetime(dG['date_time_entry'])
            temp['date_time'] = pd.to_datetime(temp['date_time'])
        
            print('aligning temperature dates to neighrest')
        
            L = []
            chunk = 50000
            dt = np.array([temp.date_time.values]*chunk).T

            for i in range(0, len(dG), chunk):
                l = np.argmin(np.abs(dt[:, :min(chunk, len(dG)-i)] - dG.date_time_entry.iloc[i:i+chunk].values[None,:]), axis = 0)
                L.append(l)
            del dt

            print('buidling dH')
            dG["meteo_index"] = np.concatenate(L)
            dH = dG.merge(temp, left_on = "meteo_index", right_index=True, copy = False)
            dH = dH.sort_values(['hash','nb']) 
        
        dG['nb_max'] = dG.groupby('hash')['nb'].transform('max')
        dG['is_last'] = (dG.nb == dG.nb_max).astype(int)
        
        print('laging the forward variables')
        idxs = ['hash','nb']
        forwards = ['x_exit', 'y_exit', 'x_min_exit', 'y_min_exit', 'dist_exit','dist_min_exit', 'speed_min_exit','speed_exit', 'target']
        
        dG = ForwardToLast(dG, idxs, forwards)
        dG.loc[dG.last_target.isnull(), 'last_target'] =  dG.loc[dG.last_target.isnull(), 'in_center_entry']
        
        if extra:
            dH = ForwardToLast(dH, idxs, forwards)
            dH.loc[dH.last_target.isnull(), 'last_target'] =  dH.loc[dH.last_target.isnull(), 'in_center_entry']
        
        print('preparation for the laging')
        leave =  ['x_exit', 'y_exit','trajectory_id','to_predict','target','origin', 'recovered',
                  'time_exit', 'time_entry', 'x_min_exit', 'y_min_exit','dist_exit', 
                  'dist_min_exit', 'speed_min_exit','speed_exit',
                  'nb_max', 'dist_cum_exit',
                  'day', 'date_time_entry','date_time', 'Condition', 'Wind','meteo_index']

        idxs = ['hash','nb']
        columns = [col+'_'+str(0) if col not in leave+idxs else col for col in dG.columns]
        dG.columns = columns
        
        print('lagging')
        dG = add_lag(dG, leave, idxs= idxs, lags=list(range(1,max_lag+1)))
        feats = [col for col in columns if col not in leave+idxs]
        features= order_feats(feats, max_lag = max_lag)
        save_names(features, pathfile=pathfile, name='features_%s.csv'%suff)
        
        if extra:
            _features = [col  for col in dH.columns if col not in idxs+leave]
            save_names(_features, pathfile=pathfile, name='_features_%s.csv'%suff)
        
        print('saving')
        dG.to_csv(pathfile+'dG_%s.csv'%suff, index=False) 
        if extra:
            dH.to_csv(pathfile+'dH_%s.csv'%suff, index=False) 

        
        
    else:
        dG = pd.read_csv(pathfile+'dG_%s.csv'%suff)
        features = get_names(pathfile=pathfile, name='features_%s.csv'%suff)
        if extra:
            dH = pd.read_csv(pathfile+'dH_%s.csv'%suff)
            _features = get_names(pathfile=pathfile, name='_features_%s.csv'%suff)
    
    if extra:
        return dG, dH, features, _features
    else:
        return dG, features
        
        

