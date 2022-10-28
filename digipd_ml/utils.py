import pandas as pd
import numpy as np
import shap


def match_patients(data, clinical, on='Patient_ID'):
    '''Match samples between the two dataframes data and clinical.
    By default, function matches patients based on Patient_ID column.
    Patient_ID will be the indices (rownames) of the returned dataframes.

    Parameters
    ----------
    data : DataFrame

    clinical : Clinical DataFrame

    Returns
    -------
    joined_data : DataFrame
        Joined dataframe contaning samples matching beetween the two dataframes.

    '''
    # match gait features and diagnosis
    is_in = list()
    for i in range(clinical.shape[0]):
        is_in.append(clinical[on].iloc[i] in data.index)

    clinical = clinical.loc[is_in]
    clinical = clinical.sort_values(by=on)
    clinical.index = clinical[on]
    clinical = clinical.drop([on], axis=1)

    data = data.join(clinical)

    return data


def feature_importances_shap_values(model, X, names_list=None, n=20):
    """
    Extracts the top n relevant features based on SHAP values in an ordered way

    Parameters
    ----------
    model : instance of BaseEstimator from scikit-learn.
            Model for which shap values have to be computed.
    X : array, shape (n_samples, n_features).
        Training data.
    names_list : list
                 Names of the features (length # features) to appear in the plot.
    n : number of features to retrieve
    """

    # generate shap values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    if np.ndim(shap_values) == 3:
        shap_values = shap_values[:, :, 1]
    if not names_list:
        names_list = list(X.columns)
    shap_df = pd.DataFrame(shap_values.values, columns = names_list)
    vals = np.abs(shap_df).mean(0)
    shap_importance = pd.DataFrame(list(zip(names_list, vals)),
                                      columns=['feature','shap_value'])
    shap_importance.sort_values(by=['shap_value'],
                                ascending=False,
                                inplace=True)
    shap_importance = shap_importance.iloc[0:n,]
    
    return shap_importance
