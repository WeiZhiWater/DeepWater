import sys
sys.path.append('../')
from hydroDL.master import default
from hydroDL.master import master
from hydroDL.model import rnn, crit, train
import numpy as np
import pandas as pd
import torch
import random

# Options for training and testing
# 0: train base model
# 1: test trained model
Action = [0, 1]

# Set hyper-parameters
EPOCH = 200
saveEPOCH = 5
BATCH_SIZE = 40
RHO = 360
HIDDENSIZE = 40
DROPRATE = 0.2

# weights for the loss function
w0 = 1
w1 = 0
w2 = 1

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
out_model_temp = ['output/example_r40/train30test10_hidden' + str(HIDDENSIZE) + '_dr' + str(round(DROPRATE, 2))
                  + '_batch' + str(BATCH_SIZE) + '_rho' + str(RHO) + '_epoch' + str(EPOCH) + '_weights_' + str(w0) + '_' + str(w1) + '_' + str(w2)]
out_model = ''.join(out_model_temp)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # GAGES2 dataset, US 580 + LamaH 216
chem_site = 40

# date length
chem_date = pd.date_range('1981-01-01', '2019-12-31')
chem_date_train = pd.date_range('1981-01-01', '2009-12-31')
chem_date_test = pd.date_range('2010-01-01', '2019-12-31')
chem_length = len(chem_date)
chem_length_train = len(chem_date_train)
chem_length_test = len(chem_date_test)

print('  loading c attributes ...')
# gages2 attributes
c1_topo = np.loadtxt('chem_input/example_r40/input_c_topo_zscore.csv', delimiter=",", skiprows=1)
c1_clim = np.loadtxt('chem_input/example_r40/input_c_clim_zscore.csv', delimiter=",", skiprows=1)
c1_hydro = np.loadtxt('chem_input/example_r40/input_c_hydro_zscore.csv', delimiter=",", skiprows=1)
c1_land = np.loadtxt('chem_input/example_r40/input_c_land_zscore.csv', delimiter=",", skiprows=1)
c1_soil = np.loadtxt('chem_input/example_r40/input_c_soil_zscore.csv', delimiter=",", skiprows=1)

c = np.concatenate((c1_topo, c1_clim, c1_hydro, c1_land, c1_soil), axis=1)
c[np.where(np.isnan(c))] = 0   # replace NaN with 0 (mean)

# first y obs, DO
print('  loading y0, DO ...')
chem_inputy = np.loadtxt('chem_input/example_r40/input_yobs_DO_zscore.csv', delimiter=",", skiprows=1)
y0 = np.reshape(np.ravel(chem_inputy.T), (chem_site, chem_length, 1))

# second y, Q
print('  loading y1, Qnorm ...')
chem_inputx_Qnorm = np.loadtxt('chem_input/example_r40/input_xplus_Qnorm_zscore.csv', delimiter=",", skiprows=1)
y1 = np.reshape(np.ravel(chem_inputx_Qnorm.T), (chem_site, chem_length, 1))

# third y, WT
print('  loading y2, WT ...\n')
chem_inputx_waterT = np.loadtxt('chem_input/example_r40/input_yobs_WT_zscore.csv', delimiter=",", skiprows=1)
y2 = np.reshape(np.ravel(chem_inputx_waterT.T), (chem_site, chem_length, 1))

y = np.concatenate((y0, y1, y2), axis=2)


# temporal x forcings
print('  loading x forcings, prcp ...')
chem_inputx_prcp = np.loadtxt('chem_input/example_r40/input_xforce_prcp_zscore.csv', delimiter=",", skiprows=1)
x_prcp = np.reshape(np.ravel(chem_inputx_prcp.T), (chem_site, chem_length, 1))

print('  loading x forcings, swe ...')
chem_inputx_swe = np.loadtxt('chem_input/example_r40/input_xforce_swe_zscore.csv', delimiter=",", skiprows=1)
x_swe = np.reshape(np.ravel(chem_inputx_swe.T), (chem_site, chem_length, 1))

print('  loading x forcings, tavg ...')
chem_inputx_tavg = np.loadtxt('chem_input/example_r40/input_xforce_tavg_zscore.csv', delimiter=",", skiprows=1)
x_tavg = np.reshape(np.ravel(chem_inputx_tavg.T), (chem_site, chem_length, 1))

print('  loading x forcings, tmax ...')
chem_inputx_tmax = np.loadtxt('chem_input/example_r40/input_xforce_tmax_zscore.csv', delimiter=",", skiprows=1)
x_tmax = np.reshape(np.ravel(chem_inputx_tmax.T), (chem_site, chem_length, 1))

print('  loading x forcings, tmin ...')
chem_inputx_tmin = np.loadtxt('chem_input/example_r40/input_xforce_tmin_zscore.csv', delimiter=",", skiprows=1)
x_tmin = np.reshape(np.ravel(chem_inputx_tmin.T), (chem_site, chem_length, 1))

print('  loading x forcings, vp ...')
chem_inputx_vp = np.loadtxt('chem_input/example_r40/input_xforce_surfp_zscore.csv', delimiter=",", skiprows=1)
x_vp = np.reshape(np.ravel(chem_inputx_vp.T), (chem_site, chem_length, 1))

print('  loading x forcings, windu ...')
chem_inputx_windu = np.loadtxt('chem_input/example_r40/input_xforce_windu_zscore.csv', delimiter=",", skiprows=1)
x_windu = np.reshape(np.ravel(chem_inputx_windu.T), (chem_site, chem_length, 1))

print('  loading x forcings, windv ...\n')
chem_inputx_windv = np.loadtxt('chem_input/example_r40/input_xforce_windv_zscore.csv', delimiter=",", skiprows=1)
x_windv = np.reshape(np.ravel(chem_inputx_windv.T), (chem_site, chem_length, 1))

# assemble temporal forcings
x = np.concatenate((x_prcp, x_swe, x_tavg, x_tmax, x_tmin, x_vp, x_windu, x_windv), axis=2)
x[np.where(np.isnan(x))] = 0  # replace NaN with 0 (mean)

# training and testing splitting
print('  loading training and testing splitting ...\n')
date_split = pd.read_excel('chem_input/example_r40/global_DO_splitting.xlsx', delimiter=",")

print(out_model)

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
        lossFun1 = crit.NSELosstest()
    elif lossoption == 0:   # training by NSE
        lossFun = crit.NSELosstest()
        lossFun1 = crit.RmseLoss()
    # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose

    # # train model # #
    model = train.trainModel_printLoss(
        model,
        x, x,
        y, y,
        c, w0, w1, w2, date_split,
        lossFun, lossFun1,
        lossTrain=lossoption,
        nEpoch=EPOCH,
        miniBatch=[BATCH_SIZE, RHO],
        saveEpoch=saveEPOCH,
        saveFolder=out_model)

    # output prediction for all-time period
    TestEPOCH = EPOCH
    modelpred = master.loadModel(out_model, epoch=TestEPOCH)
    train.testModel(modelpred, x, c, outdir=out_model, outname='all_50', batchSize=40)
