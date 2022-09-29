import shap
import matplotlib.pyplot as plt
import numpy as np


def plot_shap_values(model, X, save_fig=False, name_file=None, path=None):
    """Plot shap values for a given scikit-learn model.

    Parameters
    ----------
    model : instance of BaseEstimator from scikit-learn.
            Model for which shap values have to be computed.
    X : array, shape (n_samples, n_features).
        Training data.
    save_fig : bool
               Whether or not to save the figures.
    name_file : str
                Name of the file if saved.
    path : str
           Path for the saved figure.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    if np.ndim(shap_values) == 3:
        shap_values = shap_values[:, :, 1]
    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    ax.set_yticklabels(labels, rotation=45, fontsize=10)
    fig.set_size_inches(8.5, 5.5)
    fig.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
    fig.show()
