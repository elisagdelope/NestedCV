import shap
import matplotlib.pyplot as plt
import numpy as np


def plot_shap_values(model, X, save_fig=False, name_file=None, path=None, names_list=None, plot_title=None):
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
    names_list : list
                 Names of the features (length # features) to appear in the plot.
    plot_title : str
                 Title of the plot.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    if np.ndim(shap_values) == 3:
        shap_values = shap_values[:, :, 1]
    if not names_list:
        names_list = list(X.columns)

    shap_fig = shap.summary_plot(shap_values=shap_values.values, features=X, feature_names=names_list, show=False)
#    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    fig.axes[-1].set_aspect(150)  # set the color bar
    fig.axes[-1].set_box_aspect(150)  # set the color bar
    fig.set_size_inches(15, 10)  # set figure size
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), rotation=15, fontsize=8)
    plt.title(plot_title)
    fig.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
 #   fig.show()
    plt.close(fig)
