import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller

import FinanceDataReader as fdr

def load_data(args):  
    if args.data == 'BTCdaily210101':
        data = fdr.DataReader('KS11', '2021-01-01')
        series = data[['Close']]
    elif args.data == 'ETTh1':
        data =  pd.read_csv('data/ETTh1.csv')
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        data.set_index('date', inplace = True)
        series = data[['OT']]
    elif args.data == 'WTH':
        data =  pd.read_csv('data/WTH.csv')
        data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M', errors='raise')
        data.set_index('date', inplace = True)
        series = data[['WetBulbCelsius']]
    elif args.data == 'ECL':
        data =  pd.read_csv('data/ECL.csv')
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
        data.set_index('date', inplace = True)
        series = data[['MT_320']]
    return series
    
    