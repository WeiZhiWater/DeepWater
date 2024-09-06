"""
This sample file is part of the DeepWater package.
Repository: https://github.com/WeiZhiWater/DeepWater

This script is designed to test the explanation of the deep learning model for
riverine water quality.

For demonstration purposes, it includes an LSTM trained on data of dissolved oxygen, water temperature, 
and discharge, from only 40 US rivers, allowing for CPU-based training.

DeepWater is open-source software, licensed under the GNU Lesser General Public License as published
by the Free Software Foundation.

For contact: weizhi7367@gmail.com
"""
#%%
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('../')
import deepwater

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

# set GPU
if torch.cuda.is_available():
    GPUid = 0


# ----------------------------------- load data -----------------------------------
# variables:
#   x: forcing              format: x[nb, nt, nx]
#   c: constant attributes  format: c[nb, nc]

# dimensions:
#   nb: number of basins
#   nt: number of time steps
#   nx: number of time-dependent forcing variables
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

dir_model = "hidden40_dr0.20_batch40_rho360_epoch100_weights_1_1_1"
dir_output = os.path.join("output", dir_proj, dir_model)

# load constant attributes
dir_c = {
    "c_topo": os.path.join(dir_input, 'input_c_topo_zscore.csv'),
    "c_clim": os.path.join(dir_input, 'input_c_clim_zscore.csv'),
    "c_hydro": os.path.join(dir_input, 'input_c_hydro_zscore.csv'),
    "c_land": os.path.join(dir_input, 'input_c_land_zscore.csv'),
    "c_soil": os.path.join(dir_input, 'input_c_soil_zscore.csv')
}
c = load_attribute(dir_c)
c[np.where(np.isnan(c))] = 0   # replace NaN with 0
c = torch.from_numpy(c)
c_long = torch.reshape(c, (c.shape[0], 1, c.shape[1])).repeat((1, 14244, 1))

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
x = load_timeseries(dir_x, chem_site, chem_length)
x[np.where(np.isnan(x))] = 0  # replace NaN with 0
x = torch.from_numpy(x)

# create the dataset
X = torch.cat((x, c_long), 2)
X = X.to(GPUid).double()

print(f'Loaded data of shape: {X.shape}')


# ----------------------------------- load model ----------------------------------

model = deepwater.model.train.loadModel(dir_output, epoch="100").to(GPUid).double()

print(f'Loaded model of class: {type(model)}')

# test prediction on a GPU
preds = model(X[0:10])
preds = model(X[[0]])

print(f'Prediction output for a single observation is of shape: {preds.shape}')


# ---------------------------------- explain model --------------------------------
#%%
last_year = (13879, 13900) # 14244
y_index = 0 # 0 = DO,  1 = Qnorm  , 2 = WT

# local explanation for a single river
river = X[[0]]
local_explanation = deepwater.explain.LocalExplanation(model, data=X, method="shap_deeplift")
local_explanation.explain(river, target=y_index, time=last_year)

# feature attributions in an array
print(local_explanation.result.shape)

# plot feature attributions in time (for a single river)
local_explanation.plot_line(max_features=5, rolling=14)
plt.show()

# plot local feature importance (aggregated over time)
local_explanation.plot_bar(max_features=15)
plt.show()

#%%
# global explanation for a set of rivers
rivers = X[0:10]
global_explanation = deepwater.explain.GlobalExplanation(model, data=X, method="integrated_gradients")
global_explanation.explain(rivers, target=y_index, time=last_year)

# feature attributions in an array
print(global_explanation.result.shape)

# plot global feature importance in time (aggregated over rivers)
global_explanation.plot_line()

# plot global feature importance (aggregated over rivers and time)
global_explanation.plot_bar()