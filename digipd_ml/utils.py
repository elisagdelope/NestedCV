import pandas as pd
import numpy as np


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
