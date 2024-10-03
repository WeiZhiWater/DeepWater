import torch
import gc
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from captum.attr import IntegratedGradients, DeepLiftShap, Saliency



class LocalExplanation:
    """ Explanation of a single prediction, e.g. for one river.

    Parameters
    ----------
    model : object
        Model to be explained.
    data : np.ndarray (3d)
        Dataset modeling data distribution used to compute the explanation.
    method : str
        One of {'integrated_gradients', 'shap_deeplift', 'saliency'}.
    **kwargs
            Hyperparameters passed to the explanation class. 

    Attributes
    ----------
    explanation : object
        Attribution object from `captum`.
    result : np.array
        Feature attributions computed with the `explain()` method; to be visualized using `plot_*()`.
    
    """

    def __init__(self, model, data, method="integrated_gradients", **kwargs):
        self._model = model
        self._data = data
        self._method = method
        if method == "integrated_gradients":
            self.explanation = IntegratedGradients(model, **kwargs)
        elif method == "shap_deeplift":
            self.explanation = DeepLiftShap(model, **kwargs)
        elif method == "saliency":
            self.explanation = Saliency(model, **kwargs)
        else:
            raise TypeError("`method` has to be one of\
                            {'integrated_gradients', 'shap_deeplift', 'saliency'}")
        self.result = None


    def explain(self, x, time, target=0, **kwargs):
        """ Compute feature attributions.

        Parameters
        ----------
        x : np.ndarray (3d)
            An instance for which the prediction is to be explained.
        time : (int, int)
            Index of start and end times in which model output is to be explained.
        target : int
            Index of the target feature / model output to be explained. Defaults to 0.
        **kwargs
            Hyperparameters passed to the `attribute()` method. 

        """
        _attr_t = torch.Tensor()
        for _idt in tqdm(range(time[0], time[1])):
            with torch.no_grad():
                if self._method == "shap_deeplift" and "baselines" not in kwargs:
                    _attr = self.explanation.attribute(
                        x, 
                        target=(_idt, target), 
                        baselines=self._data,
                        **kwargs
                    )
                else:
                    _attr = self.explanation.attribute(
                        x, 
                        target=(_idt, target), 
                        **kwargs
                    )
                _attr_t = torch.cat((_attr_t, _attr[:, _idt].unsqueeze(1).to('cpu')), 1)
            del _attr
            gc.collect()
            torch.cuda.empty_cache()
        self.result = _attr_t.numpy()


    def plot_line(self, feature_names=None, max_features=5, rolling=1, **kwargs):
        """ Visualize the computed feature attributions in time.

        Parameters
        ----------
        feature_names : array-like (1d)
            Names of features to be used in the plot's legend.
        max_features : int
            Number of 'most important' features to be included in the plot. Defaults to 10.
        rolling : int
            Size of the moving window in `pandas.DataFrame.rolling()`. Defaults to 1, which means 
            no averaging over the rolling window.
        **kwargs
            Hyperparameters passed to `sns.lineplot()`. 

        """
        _df = pd.DataFrame(self.result[0])
        _importance = _df.abs().mean(axis=0)
        if feature_names:
            _importance.index = feature_names
        _top_features = _importance.sort_values(ascending=False).head(max_features).index
        _df_plot = _df.loc[:, _top_features].rolling(rolling).mean()
        ax = sns.lineplot(_df_plot, **kwargs)
        ax.axhline(y=0, linewidth=2, color="black", ls=':')
        ax.set(title="Local explanation", xlabel="Time", ylabel="Attribution")
        sns.move_legend(ax, title="Feature", loc="best")
        return ax


    def plot_bar(self, feature_names=None, max_features=10, **kwargs):
        """ Visualize the computed feature attributions aggregated over time.

        Parameters
        ----------
        feature_names : array-like (1d)
            Names of features to be used in the plot's axis.
        max_features : int
            Number of 'most important' features to be included in the plot. Defaults to 10.
        **kwargs
            Hyperparameters passed to `sns.barplot()`. 

        """
        _df = pd.DataFrame(self.result[0])
        _importance = _df.abs().mean(axis=0)
        if feature_names:
            _importance.index = feature_names
        _df_plot = _importance.sort_values(ascending=False).head(max_features).reset_index()
        ax = sns.barplot(_df_plot, x=0, y="index", orient="h", order=_df_plot['index'], **kwargs)
        ax.set(title="Local explanation", xlabel="Importance", ylabel="Feature")
        return ax



class GlobalExplanation:
    """ Aggregated explanation of a model, e.g. for a set of rivers.

    Parameters
    ----------
    model : object
        Model to be explained.
    data : np.ndarray (3d)
        Dataset modeling data distribution used to compute the explanation.
    method : str
        One of {'integrated_gradients', 'shap_deeplift', 'saliency'}.
    **kwargs
            Hyperparameters passed to the explanation class. 

    Attributes
    ----------
    explanation : object
        Attribution object from `captum`.
    result : np.array
        Feature attributions computed with the `explain()` method; to be visualized using `plot_*()`.
    
    """

    def __init__(self, model, data, method="integrated_gradients", **kwargs):
        self._model = model
        self._data = data
        self._method = method
        if method == "integrated_gradients":
            self.explanation = IntegratedGradients(model, **kwargs)
        elif method == "shap_deeplift":
            self.explanation = DeepLiftShap(model, **kwargs)
        elif method == "saliency":
            self.explanation = Saliency(model, **kwargs)
        else:
            raise TypeError("`method` has to be one of\
                            {'integrated_gradients', 'shap_deeplift', 'saliency'}")
        self.result = None


    def explain(self, X,  target=0, time=(0, 1), batch_size=1, **kwargs):
        """ Compute feature importances.

        Parameters
        ----------
        X : np.ndarray (3d)
            A dataset for which the predictions are to be explained.
        target : int
            Index of the target feature / model output to be explained.
        time : (int, int)
            Index of start and end times in which model output is to be explained.
        batch_size : int
            Batch size for iterating over the dataset `X`. Defaults to 1. Increase to speed up 
            computation if there is enough memory.
        **kwargs
            Hyperparameters passed to the `attribute()` method. 

        """
        _n_observations = X.shape[0]
        _n_batches = int(_n_observations / batch_size)
        _attr_x_t = torch.Tensor()
        for _idx in tqdm(range(_n_batches)):
            _attr_t = torch.Tensor()
            for _idt in range(time[0], time[1]):
                with torch.no_grad():
                    if self._method == "shap_deeplift" and "baselines" not in kwargs:
                        _attr = self.explanation.attribute(
                            X[(_idx * batch_size):((_idx+1) * batch_size)], 
                            target=(_idt, target), 
                            baselines=self._data,
                            **kwargs
                        )
                    else:
                        _attr = self.explanation.attribute(
                            X[(_idx * batch_size):((_idx+1) * batch_size)], 
                            target=(_idt, target), 
                            **kwargs
                        )  
                    _attr_t = torch.cat((_attr_t, _attr[:, _idt].unsqueeze(1).to('cpu')), 1)
                del _attr
                gc.collect()
                torch.cuda.empty_cache()
            _attr_x_t = torch.cat((_attr_x_t, _attr_t), 0)
        self.result = _attr_x_t.numpy()


    def plot_line(self, feature_names=None, max_features=5, rolling=1, **kwargs):
        """ Visualize the computed feature importance in time aggregated over the dataset.

        Parameters
        ----------
        feature_names : array-like (1d)
            Names of features to be used in the plot's legend.
        max_features : int
            Number of 'most important' features to be included in the plot. Defaults to 10.
        rolling : int
            Size of the moving window in `pandas.DataFrame.rolling()`. Defaults to 1, which means 
            no averaging over the rolling window.
        **kwargs
            Hyperparameters passed to `sns.lineplot()`. 

        """
        _df = pd.DataFrame(np.abs(self.result).mean(axis=0))
        _importance = _df.mean(axis=0)
        if feature_names:
            _importance.index = feature_names
        _top_features = _importance.sort_values(ascending=False).head(max_features).index
        _df_plot = _df.loc[:, _top_features].rolling(rolling).mean()
        ax = sns.lineplot(_df_plot, **kwargs)
        ax.axhline(y=0, linewidth=2, color="black", ls=':')
        ax.set(title="Global explanation", xlabel="Time", ylabel="Importance")
        sns.move_legend(ax, title="Feature", loc="best")
        return ax


    def plot_bar(self, feature_names=None, max_features=10, **kwargs):
        """ Visualize the computed feature importance aggregated over the time and dataset.

        Parameters
        ----------
        feature_names : array-like (1d)
            Names of features to be used in the plot's axis.
        max_features : int
            Number of 'most important' features to be included in the plot. Defaults to 10.
        **kwargs
            Hyperparameters passed to `sns.barplot()`. 

        """
        _df = pd.DataFrame(np.abs(self.result).mean(axis=0))
        _importance = _df.mean(axis=0)
        if feature_names:
            _importance.index = feature_names
        _df_plot = _importance.sort_values(ascending=False).head(max_features).reset_index()
        ax = sns.barplot(_df_plot, x=0, y="index", orient="h", order=_df_plot['index'], **kwargs)
        ax.set(title="Global explanation", xlabel="Importance", ylabel="Feature")
        return ax