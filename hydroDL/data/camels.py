# read camels dataset
import os
import pandas as pd
import numpy as np
import datetime as dt
from hydroDL import utils, pathCamels
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json
from . import Dataframe

# module variable
tRange = [19800101, 20150101]
tRangeobs = [19790101, 20150101]  # streamflow observations
tLst = utils.time.tRange2Array(tRange)
tLstobs = utils.time.tRange2Array(tRangeobs)
nt = len(tLst)
ntobs = len(tLstobs)
# forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
forcingLst = ['dayl', 'prcp', 'srad', 'tmax', 'tmin', 'vp']
attrLstSel = [
    'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
    'lai_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
    'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
    'max_water_content', 'geol_1st_class', 'geol_2nd_class', 'geol_porostiy',
    'geol_permeability'
]

def readGageInfo(dirDB):
    gageFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'basin_metadata',
                            'gauge_information.txt')

    data = pd.read_csv(gageFile, sep='\t', header=None, skiprows=1)
    # header gives some troubles. Skip and hardcode
    fieldLst = ['huc', 'id', 'name', 'lat', 'lon', 'area']
    out = dict()
    for s in fieldLst:
        if s is 'name':
            out[s] = data[fieldLst.index(s)].values.tolist()
        else:
            out[s] = data[fieldLst.index(s)].values
    return out

def readUsgsGage(usgsId, *, readQc=False):
    ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    huc = gageDict['huc'][ind]
    usgsFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
                            'basin_dataset_public_v1p2', 'usgs_streamflow',
                            str(huc).zfill(2),
                            '%08d_streamflow_qc.txt' % (usgsId))
    dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None)
    obs = dataTemp[4].values
    obs[obs < 0] = np.nan
    if readQc is True:
        qcDict = {'A': 1, 'A:e': 2, 'M': 3}
        qc = np.array([qcDict[x] for x in dataTemp[5]])
    if len(obs) != ntobs:
        out = np.full([ntobs], np.nan)
        dfDate = dataTemp[[1, 2, 3]]
        dfDate.columns = ['year', 'month', 'day']
        date = pd.to_datetime(dfDate).values.astype('datetime64[D]')
        [C, ind1, ind2] = np.intersect1d(date, tLstobs, return_indices=True)
        out[ind2] = obs
        if readQc is True:
            outQc = np.full([ntobs], np.nan)
            outQc[ind2] = qc
    else:
        out = obs
        if readQc is True:
            outQc = qc

    if readQc is True:
        return out, outQc
    else:
        return out


def readUsgs(usgsIdLst):
    t0 = time.time()
    y = np.empty([len(usgsIdLst), ntobs])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(usgsIdLst[k])
        y[k, :] = dataObs
    print("read usgs streamflow", time.time() - t0)
    return y


def readForcingGage(usgsId, varLst=forcingLst, *, dataset='daymet'):
    # dataset = daymet or maurer or nldas
    forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    huc = gageDict['huc'][ind]

    dataFolder = os.path.join(
        dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
        'basin_dataset_public_v1p2', 'basin_mean_forcing')
    if dataset is 'daymet':
        tempS = 'cida'
    else:
        tempS = dataset
    dataFile = os.path.join(dataFolder, dataset,
                            str(huc).zfill(2),
                            '%08d_lump_%s_forcing_leap.txt' % (usgsId, tempS))
    dataTemp = pd.read_csv(dataFile, sep=r'\s+', header=None, skiprows=4)
    nf = len(varLst)
    out = np.empty([nt, nf])
    for k in range(nf):
        # assume all files are of same columns. May check later.
        ind = forcingLst.index(varLst[k])
        out[:, k] = dataTemp[ind + 4].values
    return out


def readForcing(usgsIdLst, varLst):
    t0 = time.time()
    x = np.empty([len(usgsIdLst), nt, len(varLst)])
    for k in range(len(usgsIdLst)):
        data = readForcingGage(usgsIdLst[k], varLst)
        x[k, :, :] = data
    print("read forcing", time.time() - t0)
    return x


def readAttrAll(*, saveDict=False):
    dataFolder = os.path.join(dirDB, 'camels_attributes_v2.0',
                              'camels_attributes_v2.0')
    fDict = dict()  # factorize dict
    varDict = dict()
    varLst = list()
    outLst = list()
    keyLst = ['topo', 'clim', 'hydro', 'vege', 'soil', 'geol']

    for key in keyLst:
        dataFile = os.path.join(dataFolder, 'camels_' + key + '.txt')
        dataTemp = pd.read_csv(dataFile, sep=';')
        varLstTemp = list(dataTemp.columns[1:])
        varDict[key] = varLstTemp
        varLst.extend(varLstTemp)
        k = 0
        nGage = len(gageDict['id'])
        outTemp = np.full([nGage, len(varLstTemp)], np.nan)
        for field in varLstTemp:
            if is_string_dtype(dataTemp[field]):
                value, ref = pd.factorize(dataTemp[field], sort=True)
                outTemp[:, k] = value
                fDict[field] = ref.tolist()
            elif is_numeric_dtype(dataTemp[field]):
                outTemp[:, k] = dataTemp[field].values
            k = k + 1
        outLst.append(outTemp)
    out = np.concatenate(outLst, 1)
    if saveDict is True:
        fileName = os.path.join(dataFolder, 'dictFactorize.json')
        with open(fileName, 'w') as fp:
            json.dump(fDict, fp, indent=4)
        fileName = os.path.join(dataFolder, 'dictAttribute.json')
        with open(fileName, 'w') as fp:
            json.dump(varDict, fp, indent=4)
    return out, varLst


def readAttr(usgsIdLst, varLst):
    attrAll, varLstAll = readAttrAll()
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = gageDict['id']
    C, indGrid, ind2 = np.intersect1d(idLstAll, usgsIdLst, return_indices=True)
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out


def calStat(x):
    a = x.flatten()
    b = a[~np.isnan(a)] # kick out Nan
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatgamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    b = a[~np.isnan(a)] # kick out Nan
    b = np.log10(np.sqrt(b)+0.1) # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatbasinnorm(x):  # for daily streamflow normalized by basin area and precipitation
    basinarea = readAttr(gageDict['id'], ['area_gages2'])
    meanprep = readAttr(gageDict['id'], ['p_mean'])
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    flowua = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3))) # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)] # kick out Nan
    b = np.log10(np.sqrt(b)+0.1) # do some tranformation to change gamma characteristics plus 0.1 for 0 values
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatAll():
    statDict = dict()
    idLst = gageDict['id']
    # usgs streamflow
    y = readUsgs(idLst)
    # statDict['usgsFlow'] = calStatgamma(y)
    statDict['usgsFlow'] = calStatbasinnorm(y)
    # forcing
    x = readForcing(idLst, forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        if var=='prcp':
            statDict[var] = calStatgamma(x[:, :, k])
        else:
            statDict[var] = calStat(x[:, :, k])
    # const attribute
    attrData, attrLst = readAttrAll()
    for k in range(len(attrLst)):
        var = attrLst[k]
        statDict[var] = calStat(attrData[:, k])
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    with open(statFile, 'w') as fp:
        json.dump(statDict, fp, indent=4)


def transNorm(x, varLst, *, toNorm):
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    for k in range(len(varLst)):
        var = varLst[k]
        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k])+0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var == 'prcp' or var == 'usgsFlow':
                    x[:, k] = np.log10(np.sqrt(x[:, k])+0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, :, k] = (np.power(10,out[:, :, k])-0.1)**2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var == 'prcp' or var == 'usgsFlow':
                    out[:, k] = (np.power(10,out[:, k])-0.1)**2
    return out

def basinNorm(x, gageid, toNorm):
    # for regional training, gageid should be numpyarray
    if type(gageid) is str:
        if gageid == 'All':
            gageid = gageDict['id']
    nd = len(x.shape)
    basinarea = readAttr(gageid, ['area_gages2'])
    meanprep = readAttr(gageid, ['p_mean'])
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:,:,0] # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basinarea, (1, x.shape[1]))
    tempprep = np.tile(meanprep, (1, x.shape[1]))
    if toNorm is True:
        flow = (x * 0.0283168 * 3600 * 24) / ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3))) # (m^3/day)/(m^3/day)
    else:

        flow = x * ((temparea * (10 ** 6)) * (tempprep * 10 ** (-3)))/(0.0283168 * 3600 * 24)
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow

def createSubsetAll(opt, **kw):
    if opt is 'all':
        idLst = gageDict['id']
        subsetFile = os.path.join(dirDB, 'Subset', 'all.csv')
        np.savetxt(subsetFile, idLst, delimiter=',', fmt='%d')

# Define and initialize module variables
if os.path.isdir(pathCamels['DB']):
    dirDB = pathCamels['DB']
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)
else:
    dirDB = None
    gageDict = None
    statDict = None

def initcamels(rootDB = pathCamels['DB']):
    # reinitialize module variable
    global dirDB, gageDict, statDict
    dirDB = rootDB
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, 'Statistics_basinnorm.json')
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, 'r') as fp:
        statDict = json.load(fp)


class DataframeCamels(Dataframe):
    def __init__(self, *, subset='All', tRange):
        self.subset = subset        
        if subset == 'All':  # change to read subset later
            self.usgsId = gageDict['id']
            crd = np.zeros([len(self.usgsId), 2])
            crd[:, 0] = gageDict['lat']
            crd[:, 1] = gageDict['lon']
            self.crd = crd
        elif type(subset) is list:
            self.usgsId = np.array(subset)
            crd = np.zeros([len(self.usgsId), 2])
            C, ind1, ind2 = np.intersect1d(self.usgsId, gageDict['id'], return_indices=True)
            crd[:, 0] = gageDict['lat'][ind2]
            crd[:, 1] = gageDict['lon'][ind2]
            self.crd = crd
        else:
            raise Exception('The format of subset is not correct!')
        self.time = utils.time.tRange2Array(tRange)        

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def getDataObs(self, *, doNorm=True, rmNan=True, basinnorm = True):
        data = readUsgs(self.usgsId)
        if basinnorm is True:
            data = basinNorm(data, gageid=self.usgsId, toNorm=True)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]
        if doNorm is True:
            data = transNorm(data, 'usgsFlow', toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
            # data[np.where(np.isnan(data))] = -99
        return data

    def getDataTs(self, *, varLst=forcingLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        # read ts forcing
        data = readForcing(self.usgsId, varLst) # data:[gage*day*variable]
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataConst(self, *, varLst=attrLstSel, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        data = readAttr(self.usgsId, varLst)
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0









