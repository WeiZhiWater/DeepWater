#%%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torcheval.metrics as metrics

#%% parameters
target_feature   = "o"
threshold_ncount = 10
threshold_time   = "2009-01"
n_steps_in = 12
n_steps_out = 1

#%% load objects from train_model.py

class LSTM(nn.Module):
    def __init__(
            self, 
            input_size,
            hidden_size=128, 
            num_layers=1, 
            dropout=0, 
            output_size=1
        ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.output_size = output_size
    def forward(self, x):
        x, _ = self.lstm(x)
        # extract only the last time step
        x = x[:, (x.shape[1]-self.output_size):x.shape[1], :]
        x = self.linear(x).flatten(1)
        return x
 
model = LSTM(
    input_size=20, 
    hidden_size=128, 
    num_layers=1, 
    dropout=0.3, 
    output_size=n_steps_out
)

model.load_state_dict(torch.load(f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/model.pt'))
model.to(device)
model.eval()
loader_train = torch.load(f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_train.pt')
loader_test = torch.load(f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_test.pt')
loader_train_mini = torch.load(f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_train_mini.pt')

#%%
#%% https://github.com/trais-lab/dattri

# import dattri
# from dattri.algorithm.influence_function import IFAttributorLiSSA
# from dattri.task import AttributionTask

# def loss_func(params, data_target_pair):
#     x, y = data_target_pair
#     loss = nn.MSELoss()
#     yhat = torch.func.functional_call(model, params, x)
#     return loss(yhat, y)

# task = AttributionTask(
#     loss_func=loss_func,
#     model=model,
#     checkpoints=model.state_dict()
# )

# attributor = IFAttributorLiSSA(
#     task=task,
#     # regularization=1e-2,
#     device=device
# )

# attributor.cache(loader_train)
# with torch.no_grad():
#     score = attributor.attribute(loader_train, loader_test)

# %%
from pydvl.influence import SequentialInfluenceCalculator
from pydvl.influence.torch import DirectInfluence, EkfacInfluence
from pydvl.influence.torch.util import (
   NestedTorchCatAggregator,
   TorchNumpyConverter,
)

loss = nn.MSELoss()

with torch.no_grad():
    infl_model = EkfacInfluence(model, loss, hessian_regularization=0.01)
    infl_model = infl_model.fit(loader_train_mini)
