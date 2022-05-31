import pandas as pd
import numpy as np

def create_dataset(X, y, n_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        v = X.iloc[i:(i + n_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + n_steps])
    return np.array(Xs), np.array(ys)

def ts_train_test_split(ts,train_pct):
    train_size = int(len(ts) * train_pct)
    test_size = len(ts) - train_size
    train, test = ts.iloc[0:train_size], ts.iloc[train_size:len(ts)]
    return train,test
    
    