## Example

We show how to train, evaluate, and explain an LSTM model trained on chemical data from 40 rivers (r40). 


### 0. Install requirements

The code can be run, for example, using a conda environment with Python v3.9 and PyTorch v2.0.1

To set up an environment, use the terminal with the following steps:

```bash
conda create -n deepwater python=3.9

conda activate deepwater

pip install -r requirements.txt

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```


### 1. Train and evaluate the model

Run the following command:

```bash
python example_r40.py
```

It will use data from `chem_input/example_r40` to train an LSTM model for 100 epochs. 


### 2. Explain the model to interpret feature importance

To explain the model, install the additional `captum` package.

```bash
conda install captum==0.7.0
```

```bash
python explain_r40.py
```