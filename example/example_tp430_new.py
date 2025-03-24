import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import torch

import deepwater


BACKEND = "lstm" # old "cudnn" vs new "lstm", "seq2seq"

# ====================================== Main =====================================
# set GPU
if torch.cuda.is_available():
    DEVICE = 0
    torch.cuda.set_device(DEVICE)
else:
    print('Warning: torch device not found')

# set seeds
deepwater.utils.set_seeds(7)

# set hyper-parameters
params = {
    "epoch_run": 100,
    "epoch_save": 5,
    "batch_size": 512, # 430
    "num_workers": 2,
    "window": 100,     # rho
    "n_sequences": 512,
    "num_layers": 1,
    "hidden_size": 512,
    "drop_rate": 0.1,
    "loss_function": "rmse" # vs nse
}

# set loss weights for multitask training (up to three tasks)
#  - assign a value between 0 and 1 to each training task
#  - set it to 0 if the task is not used
w1 = 1
w2 = 0
w3 = 1

# set input and output folders
dir_proj = "tp430"
dir_input = os.path.join("chem_input", dir_proj)
print(f'input dir: {dir_input}')

dir_model = "hidden%d_dr%.2f_batch%d_window%d_epoch%d_weights_%d_%d_%d" % (
    params['hidden_size'],
    params['drop_rate'],
    params['batch_size'],
    params['window'],
    params['epoch_run'],
    w1, w2, w3
)

dir_output = os.path.join("output", dir_proj, dir_model)
print(f'output dir: {dir_output}')


# ----------------------------------- load data -----------------------------------
# variables:
#   y: target               format: y[ns, nt, ny]
#   x: time-dependent       format: x[ns, nt, nx]
#   c: constant             format: c[ns, nc]

# dimensions:
#   ns: number of sites
#   nt: number of time steps
#   ny: number of target variables (up to three y1, y2, and y3)
#   nx: number of time-dependent variables
#   nc: number of constant variables

n_sites = 430

# time length
time_range = pd.date_range('1980-01-01', '2019-12-31')
time_range_length = len(time_range)

# load train/test time split
print('  loading train/test time split ...')
date_split = pd.read_excel(os.path.join(dir_input, 'TP_splitting.xlsx'))

# load target variables
dir_y = {
    "y1": os.path.join(dir_input, 'input_yobs_TP_zscore.csv'),
    "y2": os.path.join(dir_input, 'input_yobs_Qnorm_zscore.csv'),
    "y3": os.path.join(dir_input, 'input_yobs_TPflux_zscore.csv')
}
print('  loading targets: y1, y2, y3 ...')
y = deepwater.utils.load_variables_timeseries(dir_y, n_sites, time_range_length)

# load constant variables
dir_c = {
    "c1_topo": os.path.join(dir_input, 'input_c_caravan_topo_zscore.csv'),
    "c1_clim": os.path.join(dir_input, 'input_c_caravan_clim_zscore.csv'),
    "c1_hydro": os.path.join(dir_input, 'input_c_caravan_hydro_zscore.csv'),
    "c1_land": os.path.join(dir_input, 'input_c_caravan_land_zscore.csv'),
    "c1_soil": os.path.join(dir_input, 'input_c_caravan_soil_zscore.csv'),
    "c1_geol": os.path.join(dir_input, 'input_c_caravan_geol_zscore.csv'),
    # "c2_TP": os.path.join(dir_input, 'input_c_TP_zscore.csv'),
    "c2_Qnorm": os.path.join(dir_input, 'input_c_Qnorm_zscore.csv'),
    # "c2_TPflux": os.path.join(dir_input, 'input_c_TPflux_zscore.csv')
}
print('  loading variables: c ...')
c = deepwater.utils.load_variables_constant(dir_c)
c[np.where(np.isnan(c))] = 0   # replace NaN with 0 (mean)

# load time-dependent variables
dir_x = {
    "x_time": os.path.join(dir_input, 'input_xplus_timestamp_zscore.csv'),
    
    "y2": os.path.join(dir_input, 'input_yobs_Qnorm_zscore.csv'),

    "x_prcp": os.path.join(dir_input, 'input_xforce_prcp_zscore.csv'),
    "x_dayl": os.path.join(dir_input, 'input_xforce_dayl_zscore.csv'),
    "x_srad": os.path.join(dir_input, 'input_xforce_srad_zscore.csv'),
    "x_swe": os.path.join(dir_input, 'input_xforce_swe_zscore.csv'),
    "x_tmax": os.path.join(dir_input, 'input_xforce_tmax_zscore.csv'),
    "x_tmin": os.path.join(dir_input, 'input_xforce_tmin_zscore.csv'),
    "x_vp": os.path.join(dir_input, 'input_xforce_vp_zscore.csv')
}
print('  loading variables: x ...')
x = deepwater.utils.load_variables_timeseries(dir_x, n_sites, time_range_length)
x[np.where(np.isnan(x))] = 0  # replace NaN with 0 (mean)


# ---------------------------------- train model ----------------------------------
d_input = x.shape[-1] + c.shape[-1]
d_output = y.shape[-1]

if BACKEND == "cudnn": # old backend
    # load model and loss function for training
    if torch.cuda.is_available():
        model = deepwater.model.rnn.CudnnLstmModel(
            nx=d_input, ny=d_output, 
            hiddenSize=params['hidden_size'], dr=params['drop_rate']
        )
    else:
        model = deepwater.model.rnn.CpuLstmModel(
            nx=d_input, ny=d_output, 
            hiddenSize=params['hidden_size'], dr=params['drop_rate']
        )

    if params['loss_function'] == 'rmse':
        loss_function = deepwater.model.crit.RmseLoss()
    elif params['loss_function'] == 'nse':
        loss_function = deepwater.model.crit.NSELosstest()

    # train model
    model = deepwater.model.train.trainModel_printLoss(
        model,
        x, y, c, w1, w2, w3, date_split,
        lossFun=loss_function,
        nEpoch=params['epoch_run'],
        miniBatch=[params['batch_size'], params['window']],
        saveEpoch=params['epoch_save'],
        saveFolder=dir_output
    )

    # output prediction for all-time period
    model_predictions = deepwater.model.train.loadModel(dir_output, epoch=params['epoch_run'])
    deepwater.model.train.testModel(model_predictions, x, c, outdir=dir_output, outname='_all', batchSize=params['batch_size'])


else: # new backend
    loader_train, loader_test = deepwater.utils.get_dataloaders(
        x, c, y, date_split, 
        window=params['window'], 
        n_sequences=params['n_sequences'],
        # dataloader args
        batch_size=params['batch_size'],
        num_workers=params['num_workers']
    )

    if BACKEND == "lstm":
        model = deepwater.utils.LSTM(
            input_size=d_input, 
            output_size=d_output,
            hidden_size=params['hidden_size'], 
            num_layers=params['num_layers'], 
            dropout=params['drop_rate']
        )
    elif BACKEND == "seq2seq": 
        #:# the code runs, but the model doesn't work
        model = deepwater.utils.Seq2Seq(
            input_size=d_input, 
            output_size=d_output,
            hidden_size=params['hidden_size'], 
            num_layers=params['num_layers'], 
            dropout=params['drop_rate']
        )
    else:
        print("Warning: wrong backend")

    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())

    if params['loss_function'] == 'rmse':
        loss_function = deepwater.model.crit.RmseLoss()
    elif params['loss_function'] == 'nse':
        loss_function = deepwater.model.crit.NSELosstest()

    deepwater.utils.train(
        model=model, 
        optimizer=optimizer,
        loader_train=loader_train, 
        loader_test=loader_test, 
        w1=w1, w2=w2, w3=w3, 
        loss_function=loss_function, 
        epoch_run=params['epoch_run'], 
        epoch_save=params['epoch_save'], 
        dir_output=dir_output,
        device=DEVICE
    )


    #:# run for full reproduciblity; it can be GBs of data
    # torch.save(loader_train,  os.path.join(dir_output, 'loader_train.pt'))
    # torch.save(loader_test, os.path.join(dir_output, 'loader_test.pt'))