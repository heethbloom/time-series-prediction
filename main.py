#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, dtw_metric, mse
from darts.models import NBEATSModel
import dilateloss 
from tslearn.metrics import dtw, dtw_path

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pickle
import argparse
from build_model import build_model
from load_model import load_model
from load_data import load_data
from csv import writer


############################
#data input & preprocessing#
############################
#%%
# import FinanceDataReader as fdr

# # Read data:
# data = fdr.DataReader('BTC/USD', '2021-01-01')

# with open( "BTCdaily210101", "wb" ) as file:
#     pickle.dump(data, file)

def train(args):
    series = load_data(args)
    
    # # 6:2:2 dataset split
    val_test_size = int(len(series)*0.4)  # arg - portion
    test_size = int(len(series)*0.2)
    train_due_date = series.index[-(val_test_size)]
    val_due_date = series.index[-(test_size)]
    test_due_date = series.index[-1]

    series = TimeSeries.from_dataframe(series, time_col=None, value_cols=None, fill_missing_dates=True, freq= 'H', fillna_value=None) # arg - freq
    train, val_test = series.split_before(train_due_date)
    val, test = val_test.split_before(val_due_date)

    scaler = Scaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    actual = scaler.transform(series)
    
    #############
    #model train#
    #############
    model = build_model(args)
    model.fit(train, val_series=val, verbose=True)

# # pred value print for n window with m stride
# def inference_all(args):
#     series = load_data(args)
#     test_size = int(len(series) * args.testratio)  # arg - portion
#     train_due_date, test_due_date= series.index[-(test_size)], series.index[-1]
#     series = TimeSeries.from_dataframe(series, time_col=None, value_cols=None, fill_missing_dates=True, freq= args.freq, fillna_value=None) # arg - freq
#     scaler = Scaler()
#     model = load_model(args)
#     og_test_len = len(series[train_due_date:test_due_date])-args.predlen+1
#     for i in np.arange(og_test_len // args.infstride):
#         new_valid_due_date = series.index[-test_size + i*args.infstride] 
#         print(new_valid_due_date)
#         next_input, next_output = series.split_before(new_valid_due_date)
#         next_input = scaler.transform(next_input)
#         next_output = scaler.transform(next_output)
#         new_test_len = len(series[new_valid_due_date:test_due_date])
#         print(new_test_len, i, og_test_len)
#         pred = model.predict(n = args.predlen, series=next_input)
#         real_pred = scaler.inverse_transform(pred)
#         print(scaler.inverse_transform(next_input[:args.predlen]))
#         print(real_pred)
#         # real_pred.to_csv('pred_real')

# def inference_n(args):
#     series = load_data(args)
#     test_size = int(len(series) * args.testratio)  # arg - portion
#     train_due_date, test_due_date= series.index[-(test_size)], series.index[-1]
#     series = TimeSeries.from_dataframe(series, time_col=None, value_cols=None, fill_missing_dates=True, freq= args.freq, fillna_value=None) # arg - freq
#     scaler = Scaler()
#     model = load_model(args)
    
#     for i in np.arange(args.infnum // args.infstride):
#         new_valid_due_date = series.index[-test_size + i * args.infstride] 
#         print(new_valid_due_date)
#         next_input, next_output = series.split_before(new_valid_due_date)
#         next_input = scaler.transform(next_input)
#         next_output = scaler.transform(next_output)
#         new_test_len = len(series[new_valid_due_date:test_due_date])
#         print(new_test_len, i, args.infnum)
#         pred = model.predict(n = args.predlen, series=next_input)
#         real_pred = scaler.inverse_transform(pred)
#         print(scaler.inverse_transform(next_output[:args.predlen]))
#         print(real_pred)
#         # real_pred.to_csv('pred_real')

# %% 
# 이슈1 : 평가지표를 역스케일 후의 값으로 계산해야 하는가? (o)
def eval_all(args):
    data = load_data(args)
    val_test_size, test_size = int(len(data)*0.4), int(len(data)*0.2)  
    train_due_date, val_due_date, test_due_date = data.index[-(val_test_size)], data.index[-(test_size)], data.index[-1]

    series = TimeSeries.from_dataframe(data, time_col=None, value_cols=None, fill_missing_dates=True, freq= 'H', fillna_value=None) # arg - freq
    train, val_test = series.split_before(train_due_date)

    scaler = Scaler()
    train = scaler.fit_transform(train)
    model = load_model(args)
    model_name = '{}_{}_{}_{}_{}'.format(args.model, args.criterion, args.dalpha, args.inputlen, args.predlen)
    
    input_batch, target_batch = [], []
    test_len = len(series[val_due_date:test_due_date])-args.predlen+1
    # 버전 1 : 원래 값으로 평가
    # print('start batch making...')
    # for i in np.arange(test_len//args.infstride):
    #     new_valid_due_date = data.index[-test_size+i*args.infstride] 
    #     next_input, next_target = series.split_before(new_valid_due_date)
    #     next_input = scaler.transform(next_input)
    #     input_batch.append(next_input[-args.inputlen:])
    #     target_batch.append(next_target[:args.predlen])
    # print('batch done!!')    
    
    # pred_batch = model.predict(n = args.predlen, series= input_batch)
    # pred_batch = [scaler.inverse_transform(x) for x in pred_batch]
    
    # 버전 2 : 정규화된 값으로 평가 
    print('start batch making...')
    for i in np.arange(test_len//args.infstride):
        new_valid_due_date = data.index[-test_size+i*args.infstride] 
        next_input, next_target = series.split_before(new_valid_due_date)
        next_input = scaler.transform(next_input)
        next_target = scaler.transform(next_target)
        input_batch.append(next_input[-args.inputlen:])
        target_batch.append(next_target[:args.predlen])
    print('batch done!!')    
    pred_batch = model.predict(n = args.predlen, series= input_batch)
    
    
    
    def calcdtw(pred, target):
        _, loss_dtw = dtw_path(pred.pd_series(), target.pd_series())   
        return loss_dtw   

    def calctdi(pred, target):
        N_output =  len(target)
        path, _ = dtw_path(pred.pd_series(), target.pd_series())   
        loss_tdi = 0
        for i,j in path:
            loss_tdi += (i-j)*(i-j)
        loss_tdi /= (N_output*N_output)
        return  loss_tdi
    
    print('start calc...')   
    losses_mse = [mse(x, y) for x, y in zip(pred_batch, target_batch)]
    losses_mape = [mape(x, y) for x, y in zip(pred_batch, target_batch)]
    losses_dtw = [calcdtw(x, y) for x, y in zip(pred_batch, target_batch)]      
    losses_tdi = [calctdi(x, y) for x, y in zip(pred_batch, target_batch)]   
    print('calc done!!')  

    result_dir = "/home/rj0517/darts/results"
    
    def createFolder(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
                
    createFolder(result_dir + f'/{args.data}/')
    
    def createTable(filename):
        if not os.path.exists(filename):
            title = "model mse mape dtw tdi".split(" ")
            f = open(filename, "w")
            w = writer(f)
            w.writerow(title)
            f.close()
         
    createTable(result_dir + f'/{args.data}/' + "result_table.csv")
    
    lst = [model_name, np.array(losses_mse).mean(), np.array(losses_mape).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean()]
    f = open(result_dir + f'/{args.data}/' + "result_table.csv", 'a', newline='')
    w = writer(f)
    w.writerow(lst)
    f.close


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 데이터 
    parser.add_argument('--data', type=str, default='WTH', help = 'BTCdaily210101, ETTh1, WTH, ECL') 
    # parser.add_argument('--testratio', type=float, default= 0.3)
    parser.add_argument('--freq', type=str, default= "H", help = "D, QM")
    
    # 모델 세팅
    parser.add_argument('--mode', type=str, default='train', help='train / inference_n / inference_all / eval_all / plot')
    parser.add_argument('--model', type=str, default='nbeats', help='model name- / tcn / dlinear')  
    parser.add_argument('--epoch', type=int, default=500, help='num_epoch') 
    parser.add_argument('--criterion', type=str, default="shapedilate",
                        help='MSE, dilate, dvd, wd, shapedilate')
    parser.add_argument('--dalpha', type=float, default=0.5,help='[0.1 ~ 1.0]')
    
    parser.add_argument('--inputlen', type=int, help='GPU input length', default=24)
    parser.add_argument('--predlen', type=int, help='GPU output length', default=24)
    
    
                    
    # 추론 및 평가 관련
    parser.add_argument('--infnum', type=int, help='inference iteration number', default=1)
    parser.add_argument('--infstride', type=int, help='inference iteration stride', default=1)                
                    
    parser.add_argument('--gpu_index', '-g', type=int, help='GPU index', default=1)
    parser.add_argument('--version', default='directionalquantile_v1', help='experiment version for logger') # default=None
    
    args = parser.parse_args()

    if 'train' in args.mode:
        train(args)
    
    elif 'test' in args.mode:
        eval_all(args)
    # elif 'inference' in args.mode:
    #     inference_all(args)