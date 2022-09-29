import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor, \
     GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def normalized_mse(y, x):
    mse_ref = np.var(y)

    return mean_squared_error(y, x) / mse_ref


# Define a class that performs comparison of various ML algorithms based
#  on nested CV
class NestedCV:
    def __init__(self, n_split_outer=10, n_split_inner=5):
        self.n_split_outer = n_split_outer
        self.n_split_inner = n_split_inner
        self.cv_outer = KFold(n_splits=n_split_outer)
        self.cv_inner = KFold(n_splits=n_split_inner)
        self.names_models = [
            "linearSVR", "RBFSVR", "polySVR", "RandomForest", "GradientBoosting",
            "Neural_network", "Adaboost", "ridge", "LinearRegression"]

        self.dict_models = {}
        self.tol = 1e-3
        self.max_iter = 10_000
        self.dict_models["linearSVR"] = NuSVR(
            kernel='linear', tol=self.tol, max_iter=self.max_iter)
        self.dict_models["RBFSVR"] = NuSVR(
            kernel='rbf', tol=self.tol, max_iter=self.max_iter)
        self.dict_models["polySVR"] = NuSVR(
            kernel='poly', tol=self.tol, max_iter=self.max_iter)
        self.dict_models["RandomForest"] = RandomForestRegressor(
            n_estimators=500)
        self.dict_models["ridge"] = Ridge(
            tol=self.tol, max_iter=self.max_iter)
        self.dict_models["Neural_network"] = MLPRegressor()
        self.dict_models["KNN"] = KNeighborsRegressor()
        self.dict_models["Adaboost"] = AdaBoostRegressor()
        self.dict_models["GradientBoosting"] = GradientBoostingRegressor()
        self.dict_models["LinearRegression"] = LinearRegression()

        self.dict_parameters = {}
        self.dict_parameters["linearSVR"] = {
            'model__C': np.logspace(-4, 4, 10),
            'model__nu': np.linspace(0.1, 1, 5)}
        self.dict_parameters["RBFSVR"] = {
            'model__C': np.logspace(-4, 4, 10),
            'model__nu': np.linspace(0.1, 1, 5)}
        self.dict_parameters["polySVR"] = {
            'model__C': np.logspace(-4, 4, 10),
            'model__nu': np.linspace(0.1, 1, 5)}
        self.dict_parameters["RandomForest"] = {
            'model__n_estimators': [100, 250, 500, 1000]}
        self.dict_parameters["Adaboost"] = {
            'model__n_estimators': [100, 250, 500, 1000],
            'model__learning_rate': [0.001, 0.01, 0.1, 1.0]}
        self.dict_parameters["Neural_network"] = {'model__alpha': np.logspace(
            -4, 4, 25)}
        self.dict_parameters["KNN"] = {'model__n_neighbors': np.arange(1, 11)}
        self.dict_parameters["ridge"] = {'model__alpha': np.logspace(-4, 4, 10)}
        self.dict_parameters["GradientBoosting"] = {
            'model__n_estimators': [100, 250, 500, 1000],
            'model__learning_rate': [0.001, 0.01, 0.1, 1.0]}
        self.dict_parameters["LinearRegression"] = {}

        self.list_metrics = ['mse', 'norm_mse', 'r2_score']

    def set_params(self, name_model, params):
        try:
            self.dict_parameters[name_model] = params
        except ValueError:
            print('The name of the model is not supported')

    def get_params(self, name_model):
        try:
            return self.dict_parameters[name_model]
        except ValueError:
            print('The name of the model is not supported')

    def fit(self, X, y, normalize=True, feat_select=True):
        # save nested cv results for each split (n_outer_split),
        # each model (len(name_modes)) and for each metrics
        # (AUC, accuracy, balanced, accuracy, sensitivity, specificity)

        rmv_zero_var = VarianceThreshold()
        X = rmv_zero_var.fit_transform(X)
        raw_results = np.zeros(
            (self.n_split_outer, len(self.names_models), len(self.list_metrics)))
        k = 0
        for idx_train, idx_test in self.cv_outer.split(X, y):
            print('Split number %i' % (k + 1))
            # split data
            X_train, X_test = X[idx_train, :], X[idx_test, :]
            y_train, y_test = y[idx_train], y[idx_test]

            for j, model in enumerate(self.names_models):
                steps = list()
                if normalize:
                    steps.append(('scaler', StandardScaler()))
                if feat_select:
                    if X.shape[1] < 10:
                        raise ValueError(
                            'Not enough features to perform feature selection')
                    selecter = Lasso()
                    if normalize:
                        alpha_max = np.max(np.abs(
                            StandardScaler().fit_transform(X_train).T @ y_train))
                    else:
                        alpha_max = np.max(np.abs(X_train.T @ y_train))

                    self.dict_parameters[model]['selecter__estimator__alpha'] = np.geomspace(
                        alpha_max * 0.95, alpha_max / 1_000, num=50)
                    steps.append((
                        'selecter', SelectFromModel(selecter)))
                steps.append(('model', self.dict_models[model]))
                pipeline = Pipeline(steps)
                search = GridSearchCV(
                    pipeline, self.dict_parameters[model],
                    scoring='neg_root_mean_squared_error', n_jobs=-1, cv=self.cv_inner)

                # Perform GridSearch for the current model over the parameters
                search.fit(X_train, y_train)

                # get the best performing model fit on the whole training set
                best_model = search.best_estimator_

                estimated = best_model.predict(X_test)
                mse = mean_squared_error(y_test, estimated)
                norm_mse = normalized_mse(y_test, estimated)
                r2 = r2_score(y_test, estimated)
                raw_results[k, j, :] = np.array(
                    [mse, norm_mse, r2])

            k += 1

        self.raw_results = raw_results

    def get_results(self):
        results = pd.DataFrame(columns=self.names_models)
        name_row = list()
        for i, metric in enumerate(self.list_metrics):
            df_mean = pd.DataFrame(
                data=np.mean(self.raw_results[:, :, i], axis=0)).T
            df_mean.columns = self.names_models
            results = results.append(df_mean)
            name_row.append("%s" % 'mean' + "_%s" % metric)

            df_std = pd.DataFrame(
                data=np.std(self.raw_results[:, :, i], axis=0)).T
            df_std.columns = self.names_models
            results = results.append(df_std)
            name_row.append("%s" % 'std' + "_%s" % metric)

            df_max = pd.DataFrame(
                data=np.max(self.raw_results[:, :, i], axis=0)).T
            df_max.columns = self.names_models
            results = results.append(df_max)
            name_row.append("%s" % 'max' + "_%s" % metric)

            df_min = pd.DataFrame(
                data=np.min(self.raw_results[:, :, i], axis=0)).T
            df_min.columns = self.names_models
            results = results.append(df_min)
            name_row.append("%s" % 'min' + "_%s" % metric)

        results.index = name_row
        return results

    def fit_single_model(self, X, y, name_model, normalize=True,
                         feat_select=True):
        steps = list()
        if normalize:
            steps.append(('scaler', StandardScaler()))
        if X.shape[1] < 10:
            raise ValueError(
                'Not enough features to perform feature selection')
        if feat_select:
            if X.shape[1] < 10:
                raise ValueError(
                    'Not enough features to perform feature selection')
            selecter = Lasso()
            if normalize:
                alpha_max = np.max(np.abs(
                    StandardScaler().fit_transform(X).T @ y))
            else:
                alpha_max = np.max(np.abs(X.T @ y))

            self.dict_parameters[name_model]['selecter__estimator__alpha'] = np.geomspace(
                alpha_max * 0.95, alpha_max / 1_000, num=50)
            steps.append((
                'selecter', SelectFromModel(selecter)))
        steps.append(('model', self.dict_models[name_model]))
        pipeline = Pipeline(steps)
        search = GridSearchCV(
            pipeline, self.dict_parameters[name_model],
            scoring='r2', n_jobs=-1, cv=self.cv_inner)

        # Perform GridSearch for the current model over the parameters
        search.fit(X, y)
        # get the best performing model fit on the whole training set
        best_model = search.best_estimator_
        return best_model
