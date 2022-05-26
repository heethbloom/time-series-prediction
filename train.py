#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, dtw_metric, mse
from darts.models import NBEATSModel
import dilateloss 

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
#%%
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
#%%
#############
#model train#
#############

model_nbeats = NBEATSModel(
    input_chunk_length=30,
    output_chunk_length=10,
    generic_architecture=True,
    num_stacks=4,
    num_blocks=4,
    num_layers=4,
    layer_widths=128,
    n_epochs=150,
    loss_fn = dilateloss.DerivativeDILATEloss(alpha=0.7, device= 'cuda:1'),
    nr_epochs_val_period=10,
    # optimizer_cls = torch.optim.Adam,
    optimizer_kwargs = {'lr': 1e-4},
    # lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR,
    batch_size=1024,
    pl_trainer_kwargs = {"accelerator": "gpu", "gpus": [1]},
    force_reset=True,
    save_checkpoints = True,
    log_tensorboard =True,
    # work_dir = "/home/rj0517/darts/darts_logs/nbeats_derivative_dilate",
    model_name="nbeats_dvd",
)

model_nbeats.fit(series=train)


#%%
##############################
#model load#
##############################
model_dvd = NBEATSModel.load_from_checkpoint('nbeats_dvd', best=False)
pred = model_dvd.predict(n = 10, series=train)
# print(pred)
real_pred = scaler.inverse_transform(pred)
# print(real_pred)


#%%
##############################
#predicted results fot export#
##############################

real_pred.to_csv('pred_real')

# %%
