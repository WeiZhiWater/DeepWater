import torch
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from captum.attr import IntegratedGradients, DeepLiftShap



class LocalExplanation:
    """ Explanation of a single prediction, e.g. for one river.

    Parameters
    ----------
    model : object
        Model to be explained.
    data : np.ndarray (3d)
        Data used to compute the explanation.
    method : str
        Either 'integrated_gradients' or 'shap_deeplift'

    Attributes
    ----------
    explanation : object
        Attribution object from `captum`.
    result : np.array
        Feature attributions computed with the `explain()` method; to be visualized using `plot()`.
    
    """

    def __init__(self, model, data, method="integrated_gradients"):
        self._model = model
        self._data = data
        self._method = method

        if method == "integrated_gradients":
            self.explanation = IntegratedGradients(model)
            self._baselines = 0
        elif method == "shap_deeplift":
            self.explanation = DeepLiftShap(model)
            self._baselines = self._data
        else:
            raise TypeError("`method` has to be one of {'integrated_gradients', 'shap_deeplift'}")

        self.result = None


    def explain(self, x, target=0, time=(0, 1), **kwargs):
        """ Compute feature attributions.

        Parameters
        ----------
        x : np.ndarray (3d)
            An instance for which the prediction is to be explained.
        target : int
            Index of the target feature / model output to be explained.
        time : (int, int)
            Index of start and end times in which model output is to be explained.
        **kwargs
            Hyperparameters passed to the `attribute()` method. 

        """
        _attr_t = torch.Tensor()
        for _idt in tqdm(range(time[0], time[1])):
            with torch.no_grad():
                _attr = self.explanation.attribute(
                    x, 
                    target=(_idt, target), 
                    baselines=self._baselines,
                    **kwargs
                )
                _attr_t = torch.cat((_attr_t, _attr[:, _idt].unsqueeze(1).to('cpu')), 1)
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
        _importance = _df.abs().sum(axis=0)
        if feature_names:
            _importance.index = feature_names
        _top_features = _importance.sort_values(ascending=False).head(max_features).index
        _df_plot = _df.loc[:, _top_features].rolling(rolling).mean()
        ax = sns.lineplot(_df_plot, **kwargs)
        ax.axhline(y=0, linewidth=2, color="black", ls=':')
        ax.set(xlabel="time", ylabel="attribution")
        sns.move_legend(ax, title="feature", loc="best")
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
            Hyperparameters passed to `sns.lineplot()`. 

        """
        _df = pd.DataFrame(self.result[0])
        _importance = _df.abs().sum(axis=0)
        if feature_names:
            _importance.index = feature_names
        _df_plot = _importance.sort_values(ascending=False).head(max_features).reset_index()
        ax = sns.barplot(_df_plot, x=0, y="index", orient="h", order=_df_plot['index'])
        ax.set(xlabel="importance", ylabel="feature")
        return ax



class GlobalExplanation:

    def __init__(self):
        pass


    def explain(self, X, target, **kwargs):
        """ Compute feature attributions.

        Parameters
        ----------
        X : np.ndarray (3d)
            A dataset for which the predictions are to be explained.
        target : int
            Index of the target feature / model output to be explained.
        **kwargs
            Hyperparameters passed to the `attribute()` method. 

        """
        
        pass


    def plot_line(self):
        """ Visualize the computed feature importance in time aggregated over the dataset.
        """

        pass


    def plot_bar(self):
        """ Visualize the computed feature importance aggregated over the time and dataset.
        """

        pass