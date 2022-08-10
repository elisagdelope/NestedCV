import shap
import matplotlib.pyplot as plt


def plot_shap_values(model, X, save_fig=False, name_file=None, path=None):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_values.values = shap_values.values[:, :, 1]
    shap_values.base_values = shap_values.base_values[:, 1]
    shap.plots.beeswarm(shap_values, show=False)
    fig, ax = plt.gcf(), plt.gca()
    labels = ax.get_yticklabels()
    ax.set_yticklabels(labels, rotation=45, fontsize=8)
    fig.tight_layout()
    if save_fig:
        fig.savefig(path + name_file)
    fig.show()
