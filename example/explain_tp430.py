import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import deepwater

MODEL_NAME = "hidden512_dr0.10_batch512_window100_epoch100_weights_1_0_1"

# ====================================== Main =====================================
# set GPU
if torch.cuda.is_available():
    DEVICE = 0
    torch.cuda.set_device(DEVICE)
else:
    print('Warning: torch device not found')

# set seeds
deepwater.utils.set_seeds(7)

# set input and output folders
dir_proj = "tp430"
dir_input = os.path.join("chem_input", dir_proj)
print(f'input dir: {dir_input}')

dir_output = os.path.join("output", dir_proj, MODEL_NAME)
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
time_range = pd.date_range('1980-01-01', '2019-12-31')
time_range_length = len(time_range)

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
c = torch.from_numpy(c)
c_long = torch.reshape(c, (c.shape[0], 1, c.shape[1])).repeat((1, time_range_length, 1))

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
x = torch.from_numpy(x)

# create the dataset
X = torch.cat((x, c_long), 2)
X = X.to(DEVICE).double()

print(f'Loaded data of shape: {X.shape}')


# ----------------------------------- load model ----------------------------------
# output from the `example_tp430.py` file
model = torch.load(os.path.join(dir_output, 'model_ep100.pt'), weights_only=False).to(DEVICE).double()

print(f'Loaded model of class: {type(model)}')

# test prediction on a GPU
preds = model(X[0:10])
preds = model(X[[0]])

print(f'Prediction output for a single observation is of shape: {preds.shape}')


# ---------------------------------- explain model --------------------------------

last_year = (time_range_length - 365, time_range_length)
y_index = 0 # 0 = TP,  1 = Qnorm, 2 = TPFlux


# -- local explanation for a single river
river = X[[5]]
local_explanation = deepwater.explain.LocalExplanation(model, data=X, method="shap_deeplift")
deepwater.utils.set_seeds(7)
local_explanation.explain(river, time=last_year, target=y_index)

# feature attributions in an array
print(f'Local explanation for a single observation is of shape: {local_explanation.result.shape}')

# plot feature attributions in time (for a single river)
local_explanation.plot_line(max_features=5, rolling=14)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'local_explanation_plot_line.png'))
plt.clf()

# plot local feature importance (aggregated over time)
local_explanation.plot_bar(max_features=15)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'local_explanation_plot_bar.png'))
plt.clf()


# -- global explanation for a set of rivers
rivers = X[0:10]
global_explanation = deepwater.explain.GlobalExplanation(model, data=X, method="integrated_gradients")
deepwater.utils.set_seeds(7)
global_explanation.explain(rivers, time=last_year, target=y_index, batch_size=2)

# feature attributions in an array
print(f'Global explanation for a set of observations is of shape: {global_explanation.result.shape}')

# plot global feature importance in time (aggregated over rivers)
global_explanation.plot_line(max_features=5, rolling=7)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'global_explanation_plot_line.png'))
plt.clf()

# plot global feature importance (aggregated over rivers and time)
global_explanation.plot_bar(max_features=15)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'global_explanation_plot_bar.png'))
plt.clf()


# -- estimate explanations with another algorithm
local_explanation_saliency = deepwater.explain.LocalExplanation(model, data=X, method="saliency")
deepwater.utils.set_seeds(7)
local_explanation_saliency.explain(river, time=last_year, target=y_index)
local_explanation_saliency.plot_bar(max_features=15)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'local_explanation_saliency_plot_bar.png'))
plt.clf()

global_explanation_saliency = deepwater.explain.GlobalExplanation(model, data=X, method="saliency")
deepwater.utils.set_seeds(7)
global_explanation_saliency.explain(rivers, time=last_year, target=y_index, batch_size=2)
global_explanation_saliency.plot_bar(max_features=15)
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'global_explanation_saliency_plot_bar.png'))
plt.clf()