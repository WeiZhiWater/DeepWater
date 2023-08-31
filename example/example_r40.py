"""
This sample file is part of the DeepWater package.
Repository: https://github.com/WeiZhiWater/DeepWater

This script is designed to test the execution and training of the deep learning model for
riverine water quality.

For demonstration purposes, it includes data of dissolved oxygen, water temperature, and discharge,
from only 40 US rivers, allowing for CPU-based training.

DeepWater is open-source software, licensed under the GNU Lesser General Public License as published
by the Free Software Foundation.

For contact: weizhi7367@gmail.com
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch

sys.path.append('../')
from deepwater.model import rnn, crit, train

# define functions
def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)

def load_timeseries(dict_data, chem_site, chem_length):
    """Load data from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (chem_site, chem_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)


# ======================================== Main ========================================
# set seeds
random_seed = 7
set_seeds(random_seed)

# set hyper-parameters
hyper_params = {
    "epoch_run": 100,
    "epoch_save": 5,
    "batch_size": 40,
    "rho": 360,
    "hidden_size": 40,
    "drop_rate": 0.2
}

# set loss weights for multitask training (up to three tasks)
#  - assign a value between 0 and 1 to each training task
#  - set it to 0 if the task is not used
w1 = 1
w2 = 1
w3 = 1

# set GPU
if torch.cuda.is_available():
    GPUid = 0
    torch.cuda.set_device(GPUid)

# ----------------------------------- load data -----------------------------------
# variables:
#   x: forcing              format: x[nb, nt, nx]
#   y: observations         format: y[nb, nt, ny]
#   c: constant attributes  format: c[nb, nc]

# dimensions:
#   nb: number of basins
#   nt: number of time steps
#   nx: number of time-dependent forcing variables
#   ny: number of target variables (up to three y1, y2, and y3)
#   nc: number of constant attributes

# initialize chemical data
chem_site = 40
chem_date = pd.date_range('1981-01-01', '2019-12-31')
chem_date_train = pd.date_range('1981-01-01', '2009-12-31')
chem_date_test = pd.date_range('2010-01-01', '2019-12-31')
chem_length = len(chem_date)

# set input and output folders
dir_proj = "example_r40"
dir_input = os.path.join("chem_input", dir_proj)

dir_model = "hidden%d_dr%.2f_batch%d_rho%d_epoch%d_weights_%d_%d_%d" % (
    hyper_params['hidden_size'],
    hyper_params['drop_rate'],
    hyper_params['batch_size'],
    hyper_params['rho'],
    hyper_params['epoch_run'],
    w1, w2, w3
)
dir_output = os.path.join("output", dir_proj, dir_model)

# load constant attributes
dir_c = {
    "c_topo": os.path.join(dir_input, 'input_c_topo_zscore.csv'),
    "c_clim": os.path.join(dir_input, 'input_c_clim_zscore.csv'),
    "c_hydro": os.path.join(dir_input, 'input_c_hydro_zscore.csv'),
    "c_land": os.path.join(dir_input, 'input_c_land_zscore.csv'),
    "c_soil": os.path.join(dir_input, 'input_c_soil_zscore.csv')
}
print('  loading attribute: c ...')
c = load_attribute(dir_c)
c[np.where(np.isnan(c))] = 0   # replace NaN with 0

# load time-series obs
dir_y = {
    "y1": os.path.join(dir_input, 'input_yobs_DO_zscore.csv'),
    "y2": os.path.join(dir_input, 'input_yobs_Qnorm_zscore.csv'),
    "y3": os.path.join(dir_input, 'input_yobs_WT_zscore.csv')
}
print('  loading obs: y1, y2, y3 ...')
y = load_timeseries(dir_y, chem_site, chem_length)

# load time-series forcing
dir_x = {
    "x_prcp": os.path.join(dir_input, 'input_xforce_prcp_zscore.csv'),
    "x_swe": os.path.join(dir_input, 'input_xforce_swe_zscore.csv'),
    "x_tavg": os.path.join(dir_input, 'input_xforce_tavg_zscore.csv'),
    "x_tmax": os.path.join(dir_input, 'input_xforce_tmax_zscore.csv'),
    "x_tmin": os.path.join(dir_input, 'input_xforce_tmin_zscore.csv'),
    "x_vp": os.path.join(dir_input, 'input_xforce_surfp_zscore.csv'),
    "x_windu": os.path.join(dir_input, 'input_xforce_windu_zscore.csv'),
    "x_windv": os.path.join(dir_input, 'input_xforce_windv_zscore.csv')
}
print('  loading forcing: x ...')
x = load_timeseries(dir_x, chem_site, chem_length)
x[np.where(np.isnan(x))] = 0  # replace NaN with 0

# load date split: flexible training and testing splitting
print('  loading date split ...\n')
date_split = pd.read_excel(os.path.join(dir_input, 'global_DO_splitting.xlsx'))

print('output location:', dir_output, '\n')

# ----------------------------------- model training -----------------------------------
# nx: number of time-dependent forcing variables (including constant attributes)
# nc: number of constant attributes
# ny: number of target variables
nx = x.shape[-1] + c.shape[-1]
ny = y.shape[-1]

# set LSTM model for training
if torch.cuda.is_available():
    model = rnn.CudnnLstmModel(
        nx=nx, ny=ny,
        hiddenSize=hyper_params["hidden_size"],
        dr=hyper_params["drop_rate"]
    )
else:
    model = rnn.CpuLstmModel(
        nx=nx, ny=ny,
        hiddenSize=hyper_params["hidden_size"],
        dr=hyper_params["drop_rate"]
    )

# set loss function
lossFun = crit.RmseLoss()

# model training
model = train.trainModel_printLoss(
    model,
    x, y, c,
    w1, w2, w3,
    date_split,
    lossFun,
    nEpoch=hyper_params["epoch_run"],
    miniBatch=[hyper_params["batch_size"], hyper_params["rho"]],
    saveEpoch=hyper_params["epoch_save"],
    saveFolder=dir_output
)

# model prediction for all-time period
epoch_test = hyper_params["epoch_run"]
model_test = train.loadModel(dir_output, epoch=epoch_test)

train.testModel(
    model_test,
    x, c,
    outdir=dir_output,
    outname=str(epoch_test),
    batchSize=chem_site
)