#%%
import utils

import os
import numpy as np
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torcheval.metrics as metrics


#%%
df          = pd.read_csv('data/processed/monthly.csv')
df_metadata = pd.read_csv('data/processed/monthly_metadata.csv')


#%%
target_feature   = "o"
threshold_ncount = 10
threshold_time   = "2009-01"
n_steps_in = 12
n_steps_out = 1


#%%
# train and valid are disjoint subsets of gauges
gauge_ids_train = df_metadata.gauge_id[(df_metadata.time_start < threshold_time) & 
                                       (df_metadata.ncount >= threshold_ncount)].values
gauge_ids_valid = df_metadata.gauge_id[(df_metadata.time_start >= threshold_time) & 
                                       (df_metadata.ncount >= threshold_ncount)].values

df_train = df[(df.gauge_id.isin(gauge_ids_train)) & (df.year_month < threshold_time)]
df_test  = df[(df.gauge_id.isin(gauge_ids_train)) & (df.year_month >= threshold_time)]
df_valid = df[(df.gauge_id.isin(gauge_ids_valid))]


#%% 
X_train, y_train = utils.split_gauges_sequences(
    df_train, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)
X_test, y_test = utils.split_gauges_sequences(
    df_test, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)
X_valid, y_valid = utils.split_gauges_sequences(
    df_valid, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)


#%% 
loader_train = data.DataLoader(data.TensorDataset(
    utils.preprocess_data(X_train), y_train
    ), shuffle=True, batch_size=32)
loader_test = data.DataLoader(data.TensorDataset(
    utils.preprocess_data(X_test), y_test
    ), shuffle=True, batch_size=32)
loader_valid = data.DataLoader(data.TensorDataset(
    utils.preprocess_data(X_valid), y_valid
    ), shuffle=True, batch_size=32)


#%%
model = utils.LSTM(
    input_size=20, 
    hidden_size=128, 
    num_layers=1, 
    dropout=0.4, 
    output_size=n_steps_out
)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
metric_train, metric_test, metric_valid = metrics.R2Score(), metrics.R2Score(), metrics.R2Score()


#%%
n_epochs = 300
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader_train:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == n_epochs - 1:
        model.eval()    
        with torch.no_grad():
            loss_train, loss_test, loss_valid = 0, 0, 0
            for X_batch, y_batch in loader_train:
                y_pred = model(X_batch.to(device))
                loss_train += loss_fn(y_pred, y_batch.to(device)).item()
                metric_train.update(y_pred.to('cpu'), y_batch.to('cpu'))
            for X_batch, y_batch in loader_test:
                y_pred = model(X_batch.to(device))
                loss_test += loss_fn(y_pred, y_batch.to(device)).item()
                metric_test.update(y_pred.to('cpu'), y_batch.to('cpu'))
            for X_batch, y_batch in loader_valid:
                y_pred = model(X_batch.to(device))
                loss_valid += loss_fn(y_pred, y_batch.to(device)).item()
                metric_valid.update(y_pred.to('cpu'), y_batch.to('cpu'))
        
        print("Epoch %d | Train RMSE: %.2f, R2: %.3f | Out-of-time RMSE: %.2f, R2: %.3f | Out-of-distribution RMSE: %.2f, R2: %.3f" % 
              (epoch+1, 
               np.sqrt(loss_train / len(loader_train)), 
               metric_train.compute().item(),
               np.sqrt(loss_test / len(loader_test)), 
               metric_test.compute().item(),
               np.sqrt(loss_valid / len(loader_valid)), 
               metric_valid.compute().item()))
    

#%%
newpath = f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}'
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(model.state_dict(), f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/model.pt')
torch.save(loader_train, f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_train.pt')
torch.save(loader_test, f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_test.pt')