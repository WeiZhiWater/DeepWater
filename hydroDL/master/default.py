import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv, camels
# SMAP default options
optDataSMAP = OrderedDict(
    name='hydroDL.data.dbCsv.DataframeCsv',
    rootDB=hydroDL.pathSMAP['DB_L3_Global'],
    subset='CONUSv4f1',
    varT=dbCsv.varForcing,
    varC=dbCsv.varConst,
    target=['SMAP_AM'],
    tRange=[20150401, 20160401],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0)
optTrainSMAP = OrderedDict(miniBatch=[100, 30], nEpoch=500, saveEpoch=100, seed=None)

# Streamflow default options
optDataCamels = OrderedDict(
    name='hydroDL.data.camels.DataframeCamels',
    subset='All',
    varT=camels.forcingLst,    # Wei note: function to load CAMELS data, variable name
    varC=camels.attrLstSel,    # Wei note: function to load CAMELS data, variable name
    target=['Streamflow'],
    tRange=[19900101, 19950101],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0,
    damean=False,
    davar='streamflow',
    dameanopt=0,
    lckernel=None,
    fdcopt=False)

optTrainCamels = OrderedDict(miniBatch=[100, 200], nEpoch=100, saveEpoch=50, seed=None)

""" model options """
optLstm = OrderedDict(
    name='hydroDL.model.rnn.CudnnLstmModel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),        # Wei note: bug? should be optDataCamels['varT']
    ny=1,
    hiddenSize=256,
    doReLU=True)

optLstmClose = OrderedDict(
    name='hydroDL.model.rnn.LstmCloseModel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),
    ny=1,
    hiddenSize=256,
    doReLU=True)

optCnn1dLstm = OrderedDict(
    name='hydroDL.model.rnn.CNN1dLSTMInmodel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),
    ny=1,
    nobs=7,
    hiddenSize=256,
    # CNN kernel parameters
    # Nkernel, Kernel Size, Stride
    convNKS=[(10, 5, 1), (3, 3, 3), (2, 2, 1)],
    doReLU=True,
    poolOpt=None)

optLossRMSE = OrderedDict(name='hydroDL.model.crit.RmseLoss', prior='gauss')
optLossSigma = OrderedDict(name='hydroDL.model.crit.SigmaLoss', prior='gauss')
optLossNSE = OrderedDict(name='hydroDL.model.crit.NSELosstest', prior='gauss')
optLossMSE = OrderedDict(name='hydroDL.model.crit.MSELoss', prior='gauss')


def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                if key in ['subset', 'daObs', 'poolOpt','seed', 'lckernel']:
                    opt[key] = kw[key]
                else:
                    opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt


def forceUpdate(opt, **kw):
    for key in kw:
        opt[key] = kw[key]
    return opt
