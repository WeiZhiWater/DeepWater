#%%
import utils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.utils.data as data


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


# %% analysis
model =utils.LSTM(
    input_size=20, 
    hidden_size=128, 
    num_layers=1, 
    dropout=0.4, 
    output_size=n_steps_out
)
model.load_state_dict(torch.load(f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/model.pt'))
model.to(device)
model.eval()  


# %%
df_single = df[df.gauge_id == df_metadata.sort_values("ncount", ascending=False).gauge_id.values[5]]

X_single, y_single = utils.split_gauges_sequences(
    df_single, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out,
    filter_na=False
)
loader_train_mini = data.DataLoader(data.TensorDataset(
    utils.preprocess_data(X_single), y_single
    ), shuffle=True, batch_size=1)

#%%
temp = pd.DataFrame({'y_real': [], 'y_pred': []})
with torch.no_grad():
    for X_batch, y_batch in loader_train_mini:
        y_pred = model(X_batch.to(device))
        temp = pd.concat([temp, pd.DataFrame({'y_real': [y_batch.item()], 'y_pred': [y_pred.item()]})])
temp.set_index(df_single.year_month[11:], inplace=True)

# %%
plt.figure(figsize=(12,6))
sns.set(font_scale=1.3)
sns.set_theme(style="white")
sns.lineplot(temp[224:250], linewidth=2.5, palette="tab10", marker="o", markersize=10)
plt.xticks(rotation=66)
plt.xlabel(None)
plt.title("LSTM prediction of dissolved oxygen 'o'")
plt.show()