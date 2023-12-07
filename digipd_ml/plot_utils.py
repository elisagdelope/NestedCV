import shap
import matplotlib.pyplot as plt
import numpy as np
import textwrap

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
    fig.axes[-1].set_aspect(120)  # set the color bar
    fig.axes[-1].set_box_aspect(120)  # set the color bar
    fig.axes[-1].set_ylabel('Feature value', fontsize=13, labelpad=-25)
    fig.set_size_inches(11, 6.5)  # set figure size
    wrapped_labels = [textwrap.fill(str(label.get_text()), 50) for label in labels]
    ax.set_yticklabels(wrapped_labels, fontsize=13)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), rotation=15, fontsize=11)
    plt.title(plot_title)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    if save_fig:
        fig.savefig(path + name_file)
        plt.close(fig)
    else:
       fig.show()




def plot_from_shap_values(shap_values, X, save_fig=False, name_file=None, path=None, names_list=None, plot_title=None):
    """Plot shap values for a given scikit-learn model.

    Parameters
    ----------
    shap_values : array, shape (n_samples, n_features)
                  Shap values matrix
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

    if not names_list:
        names_list = list(X.columns)

    shap_fig = shap.summary_plot(shap_values=shap_values, features=X, feature_names=names_list, show=False)
    #    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    fig.axes[-1].set_aspect(120)  # set the color bar
    fig.axes[-1].set_box_aspect(120)  # set the color bar
    fig.axes[-1].set_ylabel('Feature value', fontsize=13, labelpad=-25)
    fig.set_size_inches(11, 6.5)  # set figure size
    wrapped_labels = [textwrap.fill(str(label.get_text()), 50) for label in labels]
    ax.set_yticklabels(wrapped_labels, fontsize=13)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), rotation=15, fontsize=11)
    plt.title(plot_title)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    if save_fig:
        fig.savefig(path + name_file)
        plt.close(fig)
    else:
       fig.show()
