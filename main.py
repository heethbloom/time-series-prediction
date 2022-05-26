import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, dtw_metric, mse
from darts.models import NBEATSModel 

from tslearn.metrics import dtw, dtw_path
import pickle

############################
#data input & preprocessing#
############################

# import FinanceDataReader as fdr

# # Read data:
# data = fdr.DataReader('BTC/USD', '2021-01-01')

# with open( "BTCdaily210101", "wb" ) as file:
#     pickle.dump(data, file)

with open( "BTCdaily210101", "rb" ) as file:     # arg - data
    data = pickle.load(file)

test_size = int(len(data)*0.1)  # arg - portion
train_due_date = data.index[-(test_size)]
test_due_date = data.index[-1]

series = data[['Close']]  # arg - columns
series = TimeSeries.from_dataframe(series, time_col=None, value_cols=None, fill_missing_dates=True, freq= 'D', fillna_value=None) # arg - freq
train, test = series.split_before(train_due_date)

scaler = Scaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
actual = scaler.transform(series)


##############################
#model load#
##############################
model_dvd = NBEATSModel.load_from_checkpoint('nbeats_derivative_dilate', best=False)
pred = model_dvd.predict(n = 10, series=train)



##############################
#predicted results fot export#
##############################

pred.pd_dataframe().to_csv()







######################
#model eval for whole#
######################
def eval(model, pred_length):
    losses_mape = []
    losses_dtw = []
    losses_tdi = []
    test_len = len(series[train_due_date:test_due_date])-pred_length+1

    for i in np.arange(test_len):
        new_valid_due_date = data.index[-test_size+i] 
        next_input, next_target = series.split_before(new_valid_due_date)
        next_input = scaler.transform(next_input)
        next_target = scaler.transform(next_target)
        pred_series = model.predict(n = pred_length, series= next_input)
        
        loss_mape = mape(pred_series, next_target[:pred_length])
         # DTW and TDI
        N_output =  len(next_target[:pred_length])
        loss_dtw, loss_tdi = 0,0
        path, sim = dtw_path(pred_series.pd_series(), next_target[:pred_length].pd_series())   
        loss_dtw += sim
                    
        Dist = 0
        for i,j in path:
            Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)            
                        
        losses_mape.append( loss_mape.item())
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    print("MAPE:{:.2f}%, DTW:{:.2f}, TDI:{:.2f}".format(np.array(losses_mape).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean())) 

eval(model_nbeats, 10)


#################
#result plotting#
#################

def plot_model(model_1, pred_length, train, val, val_len):
    pred_series_1 = model_1.predict(n = pred_length, series=train)
    plt.figure(figsize=(8, 5))
    scaler.inverse_transform(actual[-val_len-50:-val_len]).plot(label="actual")
    scaler.inverse_transform(actual[-val_len-1:-(val_len-pred_length)]).plot(label="actual_val")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_1, ignore_time_axes =True)).plot(label="forecast_dvdilate")       
    plt.legend()
    plt.show()

og_test_len = len(series[train_due_date:test_due_date])-10
for i in np.arange(og_test_len):
    new_valid_due_date = data.index[-test_size+i] 
    print(new_valid_due_date)
    next_input, next_output = series.split_before(new_valid_due_date)
    next_input = scaler.transform(next_input)
    next_output = scaler.transform(next_output)
    new_test_len = len(series[new_valid_due_date:test_due_date])
    print(new_test_len, i, og_test_len)
    plot_model(model_nbeats, 10, next_input, next_output, new_test_len)