
### input data preparing
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, dtw_metric, mse
from darts.models import NBEATSModel, TCNModel
from dilateloss import DILATEloss, DILATEDIVloss, DILATEMSEloss, TDIMSEloss, WeightedDILATEloss, EXPWeightedDILATEloss, DerivativeDILATEloss, WDerivativeDILATEloss

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tslearn.metrics import dtw, dtw_path
import pickle

#%%
# import FinanceDataReader as fdr

# Read data:
# data = fdr.DataReader('BTC/USD', '2021-01-01', '2022-04-25')

# with open( "BTCdaily210101", "wb" ) as file:
#     pickle.dump(data, file)
#%%

# with open( "BTCdaily210101", "rb" ) as file:
#     data = pickle.load(file)

# data.tail()

#read data
#%%
data =  pd.read_csv('data/WTH.csv')
# data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y %H:%M', errors='raise')
data.set_index('date', inplace = True)

# data =  pd.read_csv('data/ETTh1.csv')
# data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
# data.set_index('date', inplace = True)
# data = data.iloc[-3000:]


# data =  pd.read_csv('data/ECL.csv')
# data.head()
# data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
# data.set_index('date', inplace = True)
#%%

# data =  pd.read_csv('data/ETTh1.csv')
# data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
# data = data.set_index('date', drop=True)[::-1]
# data.iloc[-3000:]
#%%
int(len(data))
#%%
data.head


#%%
# pick data
# 6:2:2 dataset split
val_test_size = int(len(data)*0.4)  # arg - portion
test_size = int(len(data)*0.2)
train_due_date = data.index[-(val_test_size)]
val_due_date = data.index[-(test_size)]
test_due_date = data.index[-1]

series = data[['WetBulbCelsius']]  # arg - columns
# series = data[['MT_320']]  # arg - columns
# series = data[['OT']
series = TimeSeries.from_dataframe(series, time_col=None, value_cols=None, fill_missing_dates=True, freq= 'H', fillna_value=None) # arg - freq
train, val_test = series.split_before(train_due_date)
val, test = val_test.split_before(val_due_date)

scaler = Scaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)
actual = scaler.transform(series)

#%%
train.plot(label = "train")
val.plot(label = 'val')
test.plot(label = "test")
plt.title("WBC")
plt.show()


### 
#model load
#%%
# model_dm = NBEATSModel.load_from_checkpoint('nbeats_dilate_mse', best=False)
model_d = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_dilate', best=False)
model_m = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_MSE', best=False)
# model_dm = NBEATSModel.load_from_checkpoint('nbeats_dilate_mse', best=False)
# model_dd = NBEATSModel.load_from_checkpoint('nbeats_dilate_div', best=False)
# model_wd = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_weighted_dilate/wd_alpha=0.7000000000000001', best=False)
# model_ewd = NBEATSModel.load_from_checkpoint('nbeats_exp_weighted_dilate', best=False)
model_dvd = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_dvd', best=False)
model_sdd = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_shapedilate', best=False)
model_sid = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/nbeats_shapeindepdilate', best=False)

#%%
model_etth1 = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/WTH/nbeats/MSE_0.5_24_24', best=True)
model_etth2 = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/WTH/nbeats/dilate_0.5_24_24', best=True)
model_etth3 = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/WTH/nbeats/wd_0.5_24_24', best=True)
# model_etth2 = TCNModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/ETTh1/tcn/MSE_0.5_48_24', best=True)
# model_etth3 = TCNModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/ETTh1/tcn/MSE_0.5_72_24', best=True)

# model_etth2 = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/ETTh1/nbeats/dilate_0.5_24_24', best=True)
# model_etth3 = NBEATSModel.load_from_checkpoint('/home/rj0517/darts/darts_logs/ETTh1/nbeats/wd_0.7_24_24', best=True)
#%%
# model_etth2.predict(n = 24, series=train[-96:])
### plotting

#%%
def plot_3_model(model_1,model_2,model_3, pred_length, train, val, val_len):
    pred_series_1 = model_1.predict(n = pred_length, series=train[-96:])
    pred_series_2 = model_2.predict(n = pred_length, series=train[-96:])
    pred_series_3 = model_3.predict(n = pred_length, series=train[-96:])
    plt.figure(figsize=(8, 5))
    scaler.inverse_transform(actual[-val_len-50:-val_len]).plot(label="actual")
    scaler.inverse_transform(actual[-val_len-1:-(val_len-pred_length)]).plot(label="actual_val")
    # scaler.inverse_transform( val[:pred_length]).plot(label="check_val")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_1, ignore_time_axes =True)).plot(label="forecast_mse")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_2, ignore_time_axes =True)).plot(label="forecast_dilate")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_3, ignore_time_axes =True)).plot(label="forecast_wd")
    # plt.text(3,3,"MAPE_1:{:.2f}%, MAPE_2:{:.2f}% ".format(mape(scaler.inverse_transform(pred_series_1), scaler.inverse_transform(val[:pred_length])), 
    #                                                    mape(scaler.inverse_transform(pred_series_2), scaler.inverse_transform(val[:pred_length]))))
    plt.title("MAPE_1:{:.2f}%, MAPE_2:{:.2f}%, MAPE_3:{:.2f}% ".format(
                                                       mape(pred_series_1, val[:pred_length]), 
                                                       mape(pred_series_2, val[:pred_length]),
                                                       mape(pred_series_3, val[:pred_length])))
    # plt.title("DTW_1:{:.2f}, DTW_2:{:.2f}, DTW_3:{:.2f} ".format(dtw_metric(pred_series_1, val[:pred_length]), 
    #                                                    dtw_metric(pred_series_2, val[:pred_length])
    #                                                    ,dtw_metric(pred_series_3, val[:pred_length])))
    # plt.title("r^2: {:.2f}% ".format(r2_score(scaler.inverse_transform(pred_series), scaler.inverse_transform(val[:pred_length]))))        
    plt.legend()
    plt.show()
#%%

def plot_2_model(model_1,model_2, pred_length, train, val, val_len):
    pred_series_1 = model_1.predict(n = pred_length, series=train)
    # pred_series_1 = model_1.predict(n = pred_length, series=train[-inputlen:]) 가 맞나?
    pred_series_2 = model_2.predict(n = pred_length, series=train)

    plt.figure(figsize=(8, 5))
    scaler.inverse_transform(actual[-val_len-50:-val_len]).plot(label="actual")
    scaler.inverse_transform(actual[-val_len-1:-(val_len-pred_length)]).plot(label="actual_val")
    # scaler.inverse_transform( val[:pred_length]).plot(label="check_val")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_1, ignore_time_axes =True)).plot(label="mse")
    scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series_2, ignore_time_axes =True)).plot(label="dilate")
    # plt.text(3,3,"MAPE_1:{:.2f}%, MAPE_2:{:.2f}% ".format(mape(scaler.inverse_transform(pred_series_1), scaler.inverse_transform(val[:pred_length])), 
    #                                                    mape(scaler.inverse_transform(pred_series_2), scaler.inverse_transform(val[:pred_length]))))
    plt.title("MAPE_1:{:.2f}%, MAPE_2:{:.2f}% ".format(
                                                       mape(pred_series_1, val[:pred_length]), 
                                                       mape(pred_series_2, val[:pred_length]),
                                                       ))
    # plt.title("DTW_1:{:.2f}, DTW_2:{:.2f}, DTW_3:{:.2f} ".format(dtw_metric(pred_series_1, val[:pred_length]), 
    #                                                    dtw_metric(pred_series_2, val[:pred_length])
    #                                                    ,dtw_metric(pred_series_3, val[:pred_length])))
    # plt.title("r^2: {:.2f}% ".format(r2_score(scaler.inverse_transform(pred_series), scaler.inverse_transform(val[:pred_length]))))        
    plt.legend()
    plt.show()
    
#%%
  
#%%
## test viz 전구간 보기
og_test_len = len(series[val_due_date:test_due_date])-10
for i in np.arange(og_test_len):
    new_valid_due_date = data.index[-test_size+i] 
    print(new_valid_due_date)
    next_input, next_output = series.split_before(new_valid_due_date)
    next_input = scaler.transform(next_input)
    next_output = scaler.transform(next_output)
    new_test_len = len(series[new_valid_due_date:test_due_date])
    print(new_test_len, i, og_test_len)
    plot_2_model(model_etth1, model_etth1, 24, next_input, next_output, new_test_len)

#%%

#%%
## viz ex 앞에서 부터 i 번째 구간 보기 (testset)
i = 200
new_valid_due_date = data.index[-val_test_size+i] # val set
# new_valid_due_date = data.index[-test_size+i]     # test set
print(new_valid_due_date)

next_input, next_output = series.split_before(new_valid_due_date)
next_input = scaler.transform(next_input)
next_output = scaler.transform(next_output)
test_len = len(series[new_valid_due_date:test_due_date])
print(test_len)
plot_3_model( model_etth1, model_etth2, model_etth3, 24, next_input, next_output, test_len)


##eval
#%%

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
    return np.array(losses_mape).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean()
    # print("MAPE:{:.2f}%, DTW:{:.2f}, TDI:{:.2f}".format(np.array(losses_mape).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean())) 
        


#%%

a, b, c = eval(model_etthm, 10)
aa, bb, cc = eval(model_etthd, 10)
aaa, bbb, ccc = eval(model_etthsd, 10)

#%%
print(a, aa, aaa)
print(b, bb, bbb)
print(c, cc, ccc)

#%%
#back test
backtest_series = model_nbeats_base.historical_forecasts(
    val,
    start=pd.Timestamp("20210322"),
    forecast_horizon=10,
    # stride=1,
    retrain=False,
    verbose=True,
)

#%%
plt.figure(figsize=(8, 5))
scaler.inverse_transform(val).plot(label="train")
scaler.inverse_transform(backtest_series).plot(label="backtest")
plt.legend()
plt.show

#%%
print("MAPE = {:.2f}%".format(mape(scaler.inverse_transform(val), scaler.inverse_transform(backtest_series))))
print("MAPE = {:.2f}%".format(dtw_metric(val, backtest_series)))


#%%
#back test
mse_backtest_series = model_nbeats_mse.historical_forecasts(
    val,
    start=pd.Timestamp("20220222"),
    forecast_horizon=10,
    # stride=1,
    retrain=False,
    verbose=True,
    )

#%%
plt.figure(figsize=(8, 5))
scaler.inverse_transform(val).plot(label="train")
scaler.inverse_transform(mse_backtest_series).plot(label="backtest")
plt.legend()
plt.show



### dataframe save


# %%

df_actual_past = scaler.inverse_transform(actual[:-val_len]).pd_dataframe()
df_actual_future = scaler.inverse_transform(actual[-val_len-1:-(val_len-pred_length)]).pd_dataframe()
df_pred = scaler.inverse_transform(actual[-val_len-1:-val_len].concatenate(pred_series, ignore_time_axes =True)).pd_datafra

#%%
mape(scaler.inverse_transform(pred_series), scaler.inverse_transform(val)[:7])
# %%
val[:7]
# %%
#, R^2: {:.2f}%, format(r2_score(scaler.inverse_transform(pred_series), scaler.inverse_transform(val)[:n]))

# %%
