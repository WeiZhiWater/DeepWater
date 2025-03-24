import torch
import numpy as np


def preprocess_data(X):
    return torch.log10(torch.nan_to_num(X)+1)


# split a multivariate sequence into samples
def split_gauge_sequence(sequence, target, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(sequence.shape[0]):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix].drop(target, axis=1)
        seq_y = sequence[end_ix-1:out_end_ix][target]
        X.append(seq_x.values)
        y.append(seq_y.values)
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()


# split sequences for all gauges and combine into one dataset
def split_gauges_sequences(df, target, n_steps_in, n_steps_out, filter_na=True):
    X_list, y_list = [], []
    for gauge_id in df.gauge_id.unique():
        X_gauge, y_gauge = split_gauge_sequence(
            df[df.gauge_id == gauge_id].drop(["gauge_id", "year_month"], axis=1), 
            target=target, 
            n_steps_in=n_steps_in, 
            n_steps_out=n_steps_out
        )
        if filter_na:
            not_nan = ~y_gauge.isnan().flatten()
            X_list.append(X_gauge[not_nan])
            y_list.append(y_gauge[not_nan])
        else:
            X_list.append(X_gauge)
            y_list.append(y_gauge)
    return torch.vstack(X_list), torch.vstack(y_list)


class LSTM(torch.nn.Module):
    def __init__(
            self, 
            input_size,
            hidden_size=128, 
            num_layers=1, 
            dropout=0, 
            output_size=1
        ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.output_size = output_size
    def forward(self, x):
        x, _ = self.lstm(x)
        # extract only the last time step
        x = x[:, (x.shape[1]-self.output_size):x.shape[1], :]
        x = self.linear(x).flatten(1)
        return x