## Example

We show how to train, evaluate, and explain an LSTM model trained on:
- chemical data from 40 rivers (r40)
- phosphorus data from 430 sites (tp430)


### 0. Install requirements

The code can be run, for example, using a conda environment with Python v3.10 and PyTorch v2.4.1

To set up an environment, use the terminal with the following steps:

```bash
conda env create -f env.yml
```


### 1. Train and evaluate the model

Run the following command:

```bash
python example_r40.py
```

It will use data from `chem_input/example_r40` to train an LSTM model for 100 epochs. 

```bash
python example_tp430_new.py
```

It will use data from `chem_input/tp430` to train an LSTM model (see parameterys in the file). 


### 2. Explain the model to interpret feature importance

To explain the model, run the following command:

```bash
python explain_r40.py
```

It will use data from `chem_input/example_r40` to explain the model from `output/example_r40`. 

```bash
python explain_tp430.py
```

It will use data from `chem_input/tp430` to explain the model from `output/tp430`. 