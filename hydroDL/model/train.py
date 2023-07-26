import numpy as np
import torch
import time
import os
import hydroDL
from hydroDL.model import rnn
from hydroDL.master import master
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

def percentage_day(date_split):
    sites = date_split['siteID'].unique()
    tempData = []
    for ind, s in enumerate(sites):
        S_training = date_split.loc[date_split['siteID'] == s, 'S_Training']
        E_training = date_split.loc[date_split['siteID'] == s, 'E_Training']
        d1 = date(S_training[ind].year, S_training[ind].month, S_training[ind].day)
        d2 = date(E_training[ind].year, E_training[ind].month, E_training[ind].day)
        delta = d2 - d1
        tempData.append(delta.days)
    temp = pd.Series(tempData)
    date_split['days_num'] = temp
    sumdays = np.sum(temp)
    tempPercent = []
    for s in sites:
        days = date_split.loc[date_split['siteID'] == s, 'days_num'].values[0]
        tempPercent.append(days/sumdays)
    temp1 = pd.Series(tempPercent)
    date_split['day_percent'] = temp1
    return date_split


# Wei ZHi: print loss for testing data, 2020-09-15
# Wei ZHi: add ability for flexible training and testing splitting, 2021-11-05
# Wei ZHi: add three weights for y0, y1, and y2, 2021-11-06

def trainModel_printLoss(model,
               x, x_test,
               y, y_test,
               c, w0, w1, w2, date_split,
               lossFun, lossFun1,
               *,
               lossTrain=1,
               nEpoch=300,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq'):

    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]

    # new train and test splitting
    date_split_new = percentage_day(date_split)
    nt_new = date_split_new['days_num'].sum() / ngrid

    #nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))
    nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt_new)))

    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * (rho - model.ct) / ngrid / nt)))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        lossFun1 = lossFun1.cuda()  # Wenyu Ouyang
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()

    if saveFolder is not None:
        if not os.path.isdir(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, 'run_printLoss.csv')
        rf = open(runFile, 'w+')

    # track loss
    pltRMSE_train = np.zeros([nEpoch, 2])
    pltRMSE_test = np.zeros([nEpoch, 2])
    pltNSE_train = np.zeros([nEpoch, 2])
    pltNSE_test = np.zeros([nEpoch, 2])

    for iEpoch in range(1, nEpoch + 1):
        lossEp_RMSE = 0
        lossEp_RMSE_test = 0
        lossEp_NSE = 0
        lossEp_NSE_test = 0
        t0 = time.time()
        i_grids = []  # Wenyu Ouyang
        # restart training mode after testing prediction
        # model.train(mode=True)

        model.train()  # prep model for training
        for iIter in range(0, nIterEp):
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:   # Wei: this is the model case
                #iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                iGrid, iT = randomIndex_percentage(ngrid, [batchSize, rho], date_split_new)
                i_grids.append(iGrid)  # Wenyu Ouyang
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP = model(xTrain)   # Wei: this is the model call

            if type(model) in [rnn.CudnnLstmModel_R2P]:
                # yP = rho/time * Batchsize * Ntraget_var
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP, Param_R2P = model(xTrain)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None)
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)
                yP = model(xTrain, zTrain)
            else:
                Exception('unknown model')

            # calculate loss
            loss = lossFun(yP, yTrain, w0, w1, w2)   # Wei ZHi, add three weights, 2021-11-06
            loss.backward()
            optim.step()
            model.zero_grad()

            # track loss
            if lossTrain == 1:    # training by RMSE
                lossEp_RMSE = lossEp_RMSE + loss.item()
                loss1 = lossFun1(yP, yTrain)  # NSE
                lossEp_NSE = lossEp_NSE + loss1.item()
            elif lossTrain == 0:  # training by NSE
                lossEp_NSE = lossEp_NSE + loss.item()
                loss1 = lossFun1(yP, yTrain)  # RMSE
                lossEp_RMSE = lossEp_RMSE + loss1.item()
        lossEp_RMSE = lossEp_RMSE / nIterEp
        lossEp_NSE = lossEp_NSE / nIterEp

        # save training model
        if saveFolder is not None:
            if iEpoch % saveEpoch == 0:
                modelFile = os.path.join(saveFolder, 'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # testing part, using x_test
        ngrid_test, nt_test, nx_test = x_test.shape
        #nIterEp_test = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid_test / nt_test)))
        #nIterEp_test = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid_test / (14610 - nt_new))))   # Wei ZHi, 2021-09-15
        nIterEp_test = int(np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid_test / (nt - nt_new))))       # Wei ZHi, 2021-11-05
            
        model.eval()  # prep model for evaluation
        # for larger training data than testing data, 2020-11-08
        if nIterEp >= nIterEp_test:
            for iIter in range(0, nIterEp_test):
                iGridTest = i_grids[iIter]   # Wenyu Ouyang
                #iGridTemp, iTTest = randomIndex(ngrid_test, nt_test, [batchSize, rho])  # Wenyu Ouyang

                # Wei, 2021-09-15
                # based on iGRidTest to randomly pick starting date in the testing period
                iTTest = randomIndex_percentage_test(nt, iGridTest, [batchSize, rho], date_split_new)   # Wei ZHi, 2021-11-05
                xTest = selectSubset(x_test, iGridTest, iTTest, rho, c=c)  # Wenyu Ouyang
                yTest = selectSubset(y_test, iGridTest, iTTest, rho)  # Wenyu Ouyang
                yP_test = model(xTest)  # Wei: this is the model call
                loss_test = lossFun(yP_test, yTest, w0, w1, w2)    # Wei ZHi, add three weights, 2021-11-06
                loss1_test = lossFun1(yP_test, yTest)
                if lossTrain == 1:
                    lossEp_RMSE_test = lossEp_RMSE_test + loss_test.item()
                    lossEp_NSE_test = lossEp_NSE_test + loss1_test.item()
                elif lossTrain == 0:
                    lossEp_RMSE_test = lossEp_RMSE_test + loss1_test.item()
                    lossEp_NSE_test = lossEp_NSE_test + loss_test.item()

            lossEp_RMSE_test = lossEp_RMSE_test / nIterEp_test
            lossEp_NSE_test = lossEp_NSE_test / nIterEp_test
        else:  # smaller training data than testing data
            lossEp_RMSE_test = np.nan
            lossEp_NSE_test = np.nan

        # printing loss
        logStr = 'Epoch {}, time {:.2f}, RMSE {:.3f}, RMSE_test {:.3f}, ' \
                 'NSE {:.3f}, NSE_test {:.3f}'.format(iEpoch, time.time()-t0,
                                                      lossEp_RMSE, lossEp_RMSE_test,
                                                      lossEp_NSE, lossEp_NSE_test)
        logStr_screen = 'Epoch {}, time {:.2f}, RMSE {:.3f}, RMSE_test {:.3f}, ' \
                        'NSE {:.3f}, NSE_test {:.3f}'.format(iEpoch, time.time()-t0,
                                                             lossEp_RMSE, lossEp_RMSE_test,
                                                             lossEp_NSE, lossEp_NSE_test)
        print(logStr_screen)

        # tracking loss
        pltRMSE_train[iEpoch-1, 0] = iEpoch
        pltRMSE_train[iEpoch-1, 1] = lossEp_RMSE
        pltRMSE_test[iEpoch-1, 0] = iEpoch
        pltRMSE_test[iEpoch-1, 1] = lossEp_RMSE_test

        pltNSE_train[iEpoch-1, 0] = iEpoch
        pltNSE_train[iEpoch-1, 1] = lossEp_NSE
        pltNSE_test[iEpoch-1, 0] = iEpoch
        pltNSE_test[iEpoch-1, 1] = lossEp_NSE_test

        # save loss
        if saveFolder is not None:
            rf.write(logStr + '\n')

    if saveFolder is not None:
        rf.close()
    return model


def trainModel(model,
               x,
               y,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq'):
    batchSize, rho = miniBatch
    # x- input; z - additional input; y - target; c - constant input
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / nt)))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')
    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                # xTrain = rho/time * Batchsize * Ninput_var
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yP = model(xTrain)
            if type(model) in [rnn.CudnnLstmModel_R2P]:
                # yP = rho/time * Batchsize * Ntraget_var
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c, tupleOut=True)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP, Param_R2P = model(xTrain)
            if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                               rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                    zTrain = selectSubset(z, iGrid, iT=None, rho=None)
                else:
                    zTrain = selectSubset(z, iGrid, iT, rho)
                yP = model(xTrain, zTrain)

            # if type(model) in [hydroDL.model.rnn.LstmCnnCond]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT, rho)
            #     zTrain = selectSubset(z, iGrid, None, None)
            #     yP = model(xTrain, zTrain)
            # if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            #     iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
            #     xTrain = selectSubset(x, iGrid, iT, rho)
            #     yTrain = selectSubset(y, iGrid, iT + model.ct, rho - model.ct)
            #     zTrain = selectSubset(z, iGrid, iT, rho)
            #     yP = model(xTrain, zTrain)
            else:
                Exception('unknown model')
            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        # save model and loss
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(model, modelFile)
    if saveFolder is not None:
        rf.close()
    return model


def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model


def testModel(model, x, c, *, outdir='output', outname='train', batchSize=None, filePathLst=None, doMC=False, outModel=None, savePath=None):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    if type(x) is tuple or type(x) is list:
        x, z = x
        if type(model) is rnn.CudnnLstmModel:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if torch.cuda.is_available():
        model = model.cuda()

    model.train(mode=False)
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)

    # deal with file name to save
    if filePathLst is None:
        # filePathLst = ['out' + str(x) for x in range(ny)]
        # filePathLst = [outdir + '/out_' + outname]   # Wei, 0603
        filePathLst = [outdir + '/out_' + outname + '_y' + str(x) for x in range(ny)]    # Wei ZHi, 2021-09-06
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, 'a')
        fLst.append(f)

    # forward for each batch
    for i in range(0, len(iS)):
        print('batch {}'.format(i))
        xTemp = x[iS[i]:iE[i], :, :]
        if c is not None:
            cTemp = np.repeat(
                np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), nt, axis=1)
            xTest = torch.from_numpy(
                np.swapaxes(np.concatenate([xTemp, cTemp], 2), 1, 0)).float()
        else:
            xTest = torch.from_numpy(
                np.swapaxes(xTemp, 1, 0)).float()
        if torch.cuda.is_available():
            xTest = xTest.cuda()
        if z is not None:
            if type(model) in [rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
                zTest = torch.from_numpy(z[iS[i]:iE[i], :]).float()
            else:
                zTemp = z[iS[i]:iE[i], :, :]
                zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
            if torch.cuda.is_available():
                zTest = zTest.cuda()

        if type(model) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:   # Wei: this is the model case
            # if z is not None:
            #     xTest = torch.cat((xTest, zTest), dim=2)
            yP = model(xTest)

            if doMC is not False:
                ySS = np.zeros(yP.shape)
                yPnp=yP.detach().cpu().numpy()
                for k in range(doMC):
                    # print(k)
                    yMC = model(xTest, doDropMC=True).detach().cpu().numpy()
                    ySS = ySS+np.square(yMC-yPnp)
                ySS = np.sqrt(ySS)/doMC
        if type(model) in [rnn.LstmCloseModel, rnn.AnnCloseModel, rnn.CNN1dLSTMmodel, rnn.CNN1dLSTMInmodel,
                           rnn.CNN1dLCmodel, rnn.CNN1dLCInmodel]:
            yP = model(xTest, zTest)
        if type(model) in [hydroDL.model.rnn.LstmCnnForcast]:
            yP = model(xTest, zTest)
        if type(model) in [rnn.CudnnLstmModel_R2P]:
            xTemp = torch.from_numpy(np.swapaxes(xTemp,1,0)).float()
            cTemp = torch.from_numpy(np.swapaxes(cTemp,1,0)).float()
            xTemp = xTemp.cuda()
            cTemp = cTemp.cuda()
            xTest_tuple = (xTemp, cTemp)
            if outModel is None:
                yP, Param_R2P = model(xTest_tuple, outModel = outModel)
                Parameters_R2P = Param_R2P.detach().cpu().numpy().swapaxes(0, 1)
            else:
                Param_R2P = model(xTest_tuple, outModel = outModel)
                Parameters_R2P = Param_R2P.detach().cpu().numpy()
                hymod_forcing = xTemp.detach().cpu().numpy().swapaxes(0, 1)

                runFile = os.path.join(savePath, 'hymod_run.csv')
                rf = open(runFile, 'a+')

                q = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
                evap = torch.zeros(hymod_forcing.shape[0], hymod_forcing.shape[1])
                for pix in range(hymod_forcing.shape[0]):
                    # model_hymod = rnn.hymod(a=Parameters_R2P[pix,0,0], b=Parameters_R2P[pix,0,1],\
                    #     cmax=Parameters_R2P[pix,0,2], rq=Parameters_R2P[pix,0,3],\
                    #         rs=Parameters_R2P[pix,0,4], s=Parameters_R2P[pix,0,5],\
                    #             slow=Parameters_R2P[pix,0,6],\
                    #                 fast=[Parameters_R2P[pix,0,7], Parameters_R2P[pix,0,8], Parameters_R2P[pix,0,9]])
                    model_hymod = rnn.hymod(a=Parameters_R2P[pix,0], b=Parameters_R2P[pix,1],\
                        cmax=Parameters_R2P[pix,2], rq=Parameters_R2P[pix,3],\
                            rs=Parameters_R2P[pix,4], s=Parameters_R2P[pix,5],\
                                slow=Parameters_R2P[pix,6],\
                                    fast=[Parameters_R2P[pix,7], Parameters_R2P[pix,8], Parameters_R2P[pix,9]])
                    for hymod_t in range(hymod_forcing.shape[1]):
                        q[pix, hymod_t], evap[pix, hymod_t] = model_hymod.advance(hymod_forcing[pix,hymod_t,0],hymod_forcing[pix,hymod_t,1])
                        nstepsLst = '{:.5f} {:.5f} {:.5f} {:.5f}'.format(hymod_forcing[pix,hymod_t,0], hymod_forcing[pix,hymod_t,1], q[pix,hymod_t], evap[pix,hymod_t])
                        print(nstepsLst)
                        rf.write(nstepsLst + '\n')

        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if doMC is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        for k in range(ny):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        if doMC is not False:
            for k in range(ny):
                f = fLst[ny+k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(
                    f, header=False, index=False)

        model.zero_grad()
        torch.cuda.empty_cache()

    for f in fLst:
        f.close()

    if batchSize == ngrid:
        # For Wenping's work to calculate loss of testing data
        # Only valid for testing without using minibatches
        yOut = torch.from_numpy(yOut)
        if type(model) in [rnn.CudnnLstmModel_R2P]:
            Parameters_R2P = torch.from_numpy(Parameters_R2P)
            if outModel is None:
                return yOut, Parameters_R2P
            else:
                return q, evap, Parameters_R2P
        else:
            return yOut

def testModelCnnCond(model, x, y, *, batchSize=None):
    ngrid, nt, nx = x.shape
    ct = model.ct
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    # cTest = torch.from_numpy(np.swapaxes(y[:, 0:ct, :], 1, 0)).float()
    cTest = torch.zeros([ct, ngrid, y.shape[-1]], requires_grad=False)
    for k in range(ngrid):
        ctemp = y[k, 0:ct, 0]
        i0 = np.where(np.isnan(ctemp))[0]
        i1 = np.where(~np.isnan(ctemp))[0]
        if len(i1) > 0:
            ctemp[i0] = np.interp(i0, i1, ctemp[i1])
            cTest[:, k, 0] = torch.from_numpy(ctemp)

    if torch.cuda.is_available():
        xTest = xTest.cuda()
        cTest = cTest.cuda()
        model = model.cuda()

    model.train(mode=False)

    yP = torch.zeros([nt - ct, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    for i in range(0, len(iS)):
        xTemp = xTest[:, iS[i]:iE[i], :]
        cTemp = cTest[:, iS[i]:iE[i], :]
        yP[:, iS[i]:iE[i], :] = model(xTemp, cTemp)
    yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
    return yOut


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor

def randomIndex_percentage(ngrid, dimSubset, date_split_new):
    batchSize, rho = dimSubset
    iGrid = np.random.choice(list(range(0, ngrid)), size=batchSize, p=date_split_new['day_percent'].tolist())
    iT = []
    for i in iGrid:
        nt = date_split_new.iloc[i]['days_num']
        T = np.random.randint(0, nt-rho, [1])[0]
        iT.append(T)
    return iGrid, iT

def randomIndex_percentage_test(nt_total, iGridTest, dimSubset, date_split_new):
    batchSize, rho = dimSubset
    #iGrid = np.random.choice(list(range(0, ngrid)), size=batchSize, p=date_split_new['day_percent'].tolist())
    iT = []
    for i in iGridTest:
        nt = date_split_new.iloc[i]['days_num']
        #T = np.random.randint(nt, 14610-rho, [1])[0]      # Wei ZHi, 2021-09-15
        T = np.random.randint(nt, nt_total - rho, [1])[0]  # Wei ZHi, 2021-11-05
        iT.append(T)
    return iT

def randomIndex(ngrid, nt, dimSubset):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    return iGrid, iT


def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
        if nt <= rho:
            iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out
