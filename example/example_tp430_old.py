import sys
sys.path.append('../')
from deepwater.model import rnn, crit, train
import numpy as np
import pandas as pd
import torch
import random

# Options for training and testing
# 0: train base model
# 1: test trained model
Action = [0, 1]

# Set hyper-parameters
EPOCH = 100
saveEPOCH = 5
BATCH_SIZE = 430
RHO = 100
HIDDENSIZE = 300
DROPRATE = 0.1

# weights for the loss function
w1 = 1
w2 = 0
w3 = 1

# specify GPU
if torch.cuda.is_available():
    GPUid = 0
    torch.cuda.set_device(GPUid)

# set seeds
randomseed = 7
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loss option: NSE = 0 vs. RMSE = 1
lossoption = 1

# output folder
path_output_temp = ['output/tp430/hidden' + str(HIDDENSIZE) + '_dr' + str(round(DROPRATE, 2)) + '_batch' + str(BATCH_SIZE) + 
                  '_rho' + str(RHO) + '_epoch' + str(EPOCH) + '_weights_' + str(w1) + '_' + str(w2) + '_' + str(w3)]
path_output = ''.join(path_output_temp)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # GAGES2 dataset, TP sites (>= 100 records, > TP 0.01 mg/L)
chem_site = 430

# date length
chem_date = pd.date_range('1980-01-01', '2019-12-31')
chem_date_train = pd.date_range('1980-01-01', '2009-12-31')
chem_date_test = pd.date_range('2010-01-01', '2019-12-31')
chem_length = len(chem_date)
chem_length_train = len(chem_date_train)
chem_length_test = len(chem_date_test)

# gages2 attributes # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print('  loading c attributes ...')
c1_gages2_topo = np.loadtxt('chem_input/tp430/input_c_caravan_topo_zscore.csv', delimiter=",", skiprows=1)
c1_gages2_clim = np.loadtxt('chem_input/tp430/input_c_caravan_clim_zscore.csv', delimiter=",", skiprows=1)
c1_gages2_hydro = np.loadtxt('chem_input/tp430/input_c_caravan_hydro_zscore.csv', delimiter=",", skiprows=1)
c1_gages2_land = np.loadtxt('chem_input/tp430/input_c_caravan_land_zscore.csv', delimiter=",", skiprows=1)
c1_gages2_soil = np.loadtxt('chem_input/tp430/input_c_caravan_soil_zscore.csv', delimiter=",", skiprows=1)
c1_gages2_geol = np.loadtxt('chem_input/tp430/input_c_caravan_geol_zscore.csv', delimiter=",", skiprows=1)

#c1_caravan = np.loadtxt('chem_input/tp430/input_c_caravan_tp430_zscore.csv', delimiter=",", skiprows=1)

# Qnorm attributes
c2_TP = np.loadtxt('chem_input/tp430/input_c_TP_zscore.csv', delimiter=",", skiprows=1)
c2_Qnorm = np.loadtxt('chem_input/tp430/input_c_Qnorm_zscore.csv', delimiter=",", skiprows=1)
c2_TPflux = np.loadtxt('chem_input/tp430/input_c_TPflux_zscore.csv', delimiter=",", skiprows=1)


c = np.concatenate((c1_gages2_topo, c1_gages2_clim, c1_gages2_hydro, c1_gages2_land, c1_gages2_soil, c1_gages2_geol, c2_Qnorm), axis=1)
c[np.where(np.isnan(c))] = 0   # replace NaN with 0 (mean)


# temporal y obs # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print('  loading y1, TP ...')
chem_inputy1 = np.loadtxt('chem_input/tp430/input_yobs_TP_zscore.csv', delimiter=",", skiprows=1)
y0 = np.reshape(np.ravel(chem_inputy1.T), (chem_site, chem_length, 1))

print('  loading y2, Q ...')
chem_inputy2 = np.loadtxt('chem_input/tp430/input_yobs_Qnorm_zscore.csv', delimiter=",", skiprows=1)
y1 = np.reshape(np.ravel(chem_inputy2.T), (chem_site, chem_length, 1))

print('  loading y3, TP flux ...')
chem_inputy3 = np.loadtxt('chem_input/tp430/input_yobs_TPflux_zscore.csv', delimiter=",", skiprows=1)
y2 = np.reshape(np.ravel(chem_inputy3.T), (chem_site, chem_length, 1))

y = np.concatenate((y0, y1, y2), axis=2)


# timestamp
print('  loading timestamp ...')
time_inputx = np.loadtxt('chem_input/tp430/input_xplus_timestamp_zscore.csv', delimiter=",", skiprows=1)
x_time = np.reshape(np.ravel(time_inputx.T), (chem_site, chem_length, 1))


# temporal x forcings # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print('  loading x forcings, prcp ...')
chem_inputx_prcp = np.loadtxt('chem_input/tp430/input_xforce_prcp_zscore.csv', delimiter=",", skiprows=1)
x_prcp = np.reshape(np.ravel(chem_inputx_prcp.T), (chem_site, chem_length, 1))

print('  loading x forcings, dayl ...')
chem_inputx_dayl = np.loadtxt('chem_input/tp430/input_xforce_dayl_zscore.csv', delimiter=",", skiprows=1)
x_dayl = np.reshape(np.ravel(chem_inputx_dayl.T), (chem_site, chem_length, 1))

print('  loading x forcings, srad ...')
chem_inputx_srad = np.loadtxt('chem_input/tp430/input_xforce_srad_zscore.csv', delimiter=",", skiprows=1)
x_srad = np.reshape(np.ravel(chem_inputx_srad.T), (chem_site, chem_length, 1))

print('  loading x forcings, swe ...')
chem_inputx_swe = np.loadtxt('chem_input/tp430/input_xforce_swe_zscore.csv', delimiter=",", skiprows=1)
x_swe = np.reshape(np.ravel(chem_inputx_swe.T), (chem_site, chem_length, 1))

print('  loading x forcings, tmax ...')
chem_inputx_tmax = np.loadtxt('chem_input/tp430/input_xforce_tmax_zscore.csv', delimiter=",", skiprows=1)
x_tmax = np.reshape(np.ravel(chem_inputx_tmax.T), (chem_site, chem_length, 1))

print('  loading x forcings, tmin ...')
chem_inputx_tmin = np.loadtxt('chem_input/tp430/input_xforce_tmin_zscore.csv', delimiter=",", skiprows=1)
x_tmin = np.reshape(np.ravel(chem_inputx_tmin.T), (chem_site, chem_length, 1))

print('  loading x forcings, vp ...\n')
chem_inputx_vp = np.loadtxt('chem_input/tp430/input_xforce_vp_zscore.csv', delimiter=",", skiprows=1)
x_vp = np.reshape(np.ravel(chem_inputx_vp.T), (chem_site, chem_length, 1))



# assemble temporal forcings
x = np.concatenate((x_time, y1, x_prcp, x_dayl, x_srad, x_swe, x_tmax, x_tmin, x_vp), axis=2)
x[np.where(np.isnan(x))] = 0  # replace NaN with 0 (mean)


# training and testing splitting
print('  loading training and testing splitting ...')
date_split = pd.read_excel('chem_input/tp430/TP_splitting.xlsx')

# train and test splitting
#xtrain = x[:, 0:chem_length_train, :]
#xtest = x[:, chem_length_train:chem_length, :]
#ytrain = y[:, 0:chem_length_train, :]
#ytest = y[:, chem_length_train:chem_length, :]

print(path_output)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Train the base model
if 0 in Action:
    # load data
    # df, x, y, c = master.loadData(optData) # df: CAMELS dataframe; x: forcings; y: obs; c:attributes
    # main outputs of this step are numpy ndArrays: x[nb,nt,nx], y[nb,nt, ny], c[nb,nc]
    # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
    # ny: number of target variables, nc: number of constant attributes
    nx = x.shape[-1] + c.shape[-1]  # update nx, nx = nx + nc
    ny = y.shape[-1]

    # load model for training
    if torch.cuda.is_available():
        model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE, dr=DROPRATE)
    else:
        model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE, dr=DROPRATE)

    if lossoption == 1:     # training by RMSE
        lossFun = crit.RmseLoss()
    elif lossoption == 0:   # training by NSE
        lossFun = crit.NSELosstest()
    # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose

    # # train model # #   
    model = train.trainModel_printLoss(
        model,
        x, y, c, w1, w2, w3, date_split,
        lossFun=lossFun,
        nEpoch=EPOCH,
        miniBatch=[BATCH_SIZE, RHO],
        saveEpoch=saveEPOCH,
        saveFolder=path_output
    )

    # output prediction for all-time period
    TestEPOCH = EPOCH
    modelpred = train.loadModel(path_output, epoch=TestEPOCH)
    train.testModel(modelpred, x, c, outdir=path_output, outname='all_100', batchSize=100)
