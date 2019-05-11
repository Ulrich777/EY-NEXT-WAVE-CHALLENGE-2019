# univariate lstm example
import numpy as np
from keras.layers import Reshape, Input, CuDNNLSTM, Bidirectional, Dense, Dropout, LSTM, Embedding, concatenate
from keras.models import Model
import keras.backend as K


def f_score(y_true, y_pred):
    """
    f1 score

    :param y_true:
    :param y_pred:
    :return:
    """
    tp_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    fp_3d = K.concatenate(
        [
            K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
            K.cast(K.round(y_pred), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )
    fn_3d = K.concatenate(
        [
            K.cast(y_true, 'bool'),
            K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
            K.cast(K.ones_like(y_pred), 'bool')
        ], axis=1
    )

    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * ((precision * recall) / (precision + recall))


#=============================================================#

def model_reg(dim, cluster_centers_arr, lstm=False, time=5, feats=24):
  """ 
  defining regression model
  
  """
  
  inp = Input(shape=(dim,))
  
  if lstm:
      x = Reshape(target_shape=(time, feats,))(inp)
      x = Bidirectional(CuDNNLSTM(100 , return_sequences=True))(x)
      x = CuDNNLSTM(100, return_sequences=False)(x)
      x = Dense(500, activation='relu')(x)
      x = Dropout(0.5)(x)
      
  else:
    x = Dense(500, activation='relu')(inp)
    x = Dropout(0.5)(x)
  
  x = Dense(cluster_centers_arr.shape[0], activation="softmax")(x)
  x = Dense(2, activation='linear', weights=[cluster_centers_arr, np.array([0.,0.])], name='pos')(x)
  
  model = Model(inputs=inp, outputs=x)
  
  model.compile(loss='mse', optimizer='adam')
  
  
  return model

#======================================================================#



def build_model(unique_idxs, dim, lstm=False):
    models = []
    inputs = []

    for unique_idx in unique_idxs :
    
        inp = Input(shape=(1,))
        inputs.append(inp)
        #no_of_unique_cat  = dG[categoical_var].nunique()
        no_of_unique_cat  = unique_idx
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 50 )
        embedding_size = max(3, embedding_size)
        embedding_size = int(embedding_size)
        x = Embedding( no_of_unique_cat+1, embedding_size)(inp)
        x = Reshape(target_shape=(embedding_size,))(x)
        models.append( x )
    
    inp = Input(shape=(dim,))
    inputs.append(inp)
    
    if lstm:
        x = Reshape(target_shape=(5,24,))(inp)
        x = Bidirectional(LSTM(100 , activation='relu', return_sequences=True))(x)
        x = LSTM(100, activation='relu', return_sequences=False)(x)
        x = Dense(50, activation='relu')(x)
    
    else:
        x = Dense(64, activation='relu')(inp)
    models.append(x)
 
  
    model_concat =  concatenate(models)
    model_concat = Dense(32, activation='relu')(model_concat)
    model_concat = Dense(1, activation='sigmoid')(model_concat)

    full_model = Model(inputs=inputs,outputs=model_concat)
    full_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f_score])
    return full_model

#========================================================================#
def preproc(X_train, categorical_idxs, other_idxs) : 

    input_list_train = []
    
    
    for idx in categorical_idxs :

        
        input_list_train.append( X_train[:,idx].astype(int)  )
        

    #the rest of the columns
    input_list_train.append(X_train[:,other_idxs])
    return input_list_train

#=========================================================================#
def search_threshold(y, oof, metrics, breaks=101):
    best_threshold = -1
    best_score = -1
        
    thresholds = np.linspace(0,1, breaks)
        
    for threshold in thresholds:
        score = metrics(y, (oof>threshold).astype(int))
        if score>best_score:
            best_score = score
            best_threshold = threshold
                
    print('best threshold: %.3f  best score: %.4f'%(best_threshold, best_score))
    return best_score, best_threshold

