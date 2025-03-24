import os
import time
import random
import torch
import torch.nn as nn
import numpy as np


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_variables_timeseries(dict_data, chem_site, chem_length):
    """Load data from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = np.loadtxt(path, delimiter=",", skiprows=1)
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (chem_site, chem_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)


def load_variables_constant(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)


#:# split a multivariate sequence into samples
# deterministic for validation
def split_sequence_deterministic(x, y, window):
    sequences_x, sequences_y = list(), list()
    split_points = np.arange(0, x.shape[0] - window, window)
    for i in split_points:
        sequence_x = x[i:i+window, :]
        sequence_y = y[i:i+window, :]
        sequences_x.append(sequence_x)
        sequences_y.append(sequence_y)
    return sequences_x, sequences_y

# random for subsampling in training
def split_sequence_random(x, y, window, n_sequences):
    sequences_x, sequences_y = list(), list()
    split_points = np.random.randint(0, x.shape[0] - window, n_sequences)
    for i in split_points:
        sequence_x = x[i:i+window, :]
        sequence_y = y[i:i+window, :]
        sequences_x.append(sequence_x)
        sequences_y.append(sequence_y)
    return sequences_x, sequences_y


def get_dataloaders(x, c, y, date_split, window, n_sequences, **kwargs):
    # combine x and c
    c = np.moveaxis(np.tile(c, (x.shape[1], 1, 1)), 0, 1)
    xc = np.concat((x, c), axis=2) # [ns, nt, nx + nc]

    # save results from date splits
    seq_xc_train = list()
    seq_y_train = list()
    seq_xc_test = list()
    seq_y_test = list()

    # split data into train and test sets
    for site, row in date_split.iterrows():
        split_point = (row['E_Training'] - row['S_Training']).days + 1
        
        xc_site_train = xc[site, :split_point, :]
        y_site_train = y[site, :split_point, :]
        seq_xc_site_train, seq_y_site_train = split_sequence_random(xc_site_train, y_site_train, window, n_sequences)
        seq_xc_train.append(seq_xc_site_train)
        seq_y_train.append(seq_y_site_train)

        xc_site_test = xc[site, split_point:, :]
        y_site_test = y[site, split_point:, :]
        seq_xc_site_test, seq_y_site_test = split_sequence_deterministic(xc_site_test, y_site_test, window)
        seq_xc_test.append(seq_xc_site_test)
        seq_y_test.append(seq_y_site_test)

    # combine all sequences
    xc_train = torch.tensor(np.concat(seq_xc_train)).float()
    y_train = torch.tensor(np.concat(seq_y_train)).float()
    xc_test = torch.tensor(np.concat(seq_xc_test)).float()
    y_test = torch.tensor(np.concat(seq_y_test)).float()
    
    # create data loaders
    loader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xc_train, y_train), 
        shuffle=True, drop_last=True,
        **kwargs
    )
    loader_test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xc_test, y_test), 
        shuffle=False, drop_last=False,
        **kwargs
    )
    return loader_train, loader_test


def train(
        model, optimizer,
        loader_train, loader_test, 
        w1, w2, w3, loss_function, 
        epoch_run, epoch_save, 
        dir_output,
        device
    ):

    if dir_output is not None:
        if not os.path.isdir(dir_output):
            os.makedirs(dir_output)
        path_log = os.path.join(dir_output, 'run_printLoss.csv')
        file_log = open(path_log, 'w+')

    t0 = time.time()
    for epoch in range(epoch_run):
        model.train()
        for X_batch, y_batch in loader_train:
            y_pred = model(X_batch.to(device))
            loss = loss_function(y_pred, y_batch.to(device), w1, w2, w3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % epoch_save == 0 or epoch == epoch_run - 1:
            model.eval()    
            with torch.no_grad():
                loss_train, loss_test = 0, 0
                for X_batch, y_batch in loader_train:
                    y_pred = model(X_batch.to(device))
                    loss_train += loss_function(y_pred, y_batch.to(device), w1, w2, w3).item()
                for X_batch, y_batch in loader_test:
                    y_pred = model(X_batch.to(device))
                    loss_test += loss_function(y_pred, y_batch.to(device), w1, w2, w3).item()

            log_text = 'epoch: {} | time: {:.2f} | Train RMSE: {:.3f} | Test RMSE: {:.3f}'.format(
                epoch + 1, time.time() - t0, loss_train / len(loader_train), loss_test / len(loader_test))

            if dir_output is not None:
                file_log.write(log_text + '\n')
                path_model = os.path.join(dir_output, 'model_ep' + str(epoch + 1) + '.pt')
                torch.save(model, path_model)

            print(log_text)

    if dir_output is not None:
        file_log.close()

    return model


class LSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Use dropout in the LSTM if there are multiple layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        # Apply an extra dropout layer to LSTM outputs
        self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        lstm_out, _ = self.lstm(x)
        # Apply dropout to the LSTM output
        dropout_out = self.dropout_layer(lstm_out)
        # Linear layer maps the hidden state to the output dimension
        output = self.linear(dropout_out)
        return output
    

#:# the code runs, but the model doesn't work
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.decoder_lstm = nn.LSTM(self.output_size, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        _, hidden = self.encoder_lstm(input)
        input_t = torch.zeros(input.size(0), 1, self.output_size, dtype=torch.float, device=input.device)
        output_tensor = torch.zeros(input.size(0), input.size(1), self.output_size, device=input.device)
        
        for t in range(input.size(1)):
            output_t, hidden = self.decoder_lstm(input_t, hidden)
            output_t = self.linear(self.dropout(output_t[:, -1, :]))  # Apply dropout before linear layer
            input_t = output_t.unsqueeze(1)  # Prepare output as next decoder input
            output_tensor[:, t, :] = output_t
        
        return output_tensor
#:#