import torch
from captum.attr import IntegratedGradients, DeepLiftShap, GradientShap


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
        x : np.ndarray (2d)
            An instance for which the prediction is to be explained.
        target : int
            Index of the target feature / model output to be explained.
        time : (int, int)
            Index of start and end times in which model output is to be explained.
        **kwargs
            Hyperparameters passed to the `attribute()` method. 

        """
        attr_t = torch.Tensor()
        for idt in range(time[0], time[1]):
            attr = self.explanation.attribute(
                x, 
                target=(idt, target), 
                baselines=self._baselines,
                **kwargs
            )
            attr_t = torch.cat((attr_t, attr[:, idt].unsqueeze(1).to('cpu')), 1)
            del attr
            torch.cuda.empty_cache()

        self.result = attr_t.numpy()


    def plot_line(self):
        """ Visualize the computed feature attributions in time.
        """

        pass


    def plot_bar(self):
        """ Visualize the computed feature attributions aggregated over time.
        """

        pass



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