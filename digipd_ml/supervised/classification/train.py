import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, \
     GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from imodels import FIGSClassifier
from imblearn.under_sampling import RandomUnderSampler
import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


# Define a class that performs comparison of various ML algorithms based
#  on nested CV
class NestedCV:
    def __init__(self, n_split_outer=10, n_split_inner=5, random_state=42):
        self.n_split_outer = n_split_outer
        self.n_split_inner = n_split_inner
        self.cv_outer = StratifiedKFold(
            n_splits=n_split_outer, shuffle=True, random_state=random_state)
        self.cv_inner = StratifiedKFold(
            n_splits=n_split_inner, shuffle=True, random_state=random_state)
        self.names_models = [
            "linearSVM", "RBFSVM", "RandomForest", "GradientBoosting",
            "Adaboost", "LogisticRegression"]
        self.dict_models = {}
        self.tol = 1e-3
        self.max_iter = 10_000

        self.dict_models["linearSVM"] = SVC(
            kernel='linear', tol=self.tol, max_iter=self.max_iter,
            probability=True)
        self.dict_models["RBFSVM"] = SVC(
            kernel='rbf', tol=self.tol, max_iter=self.max_iter,
            probability=True)
        self.dict_models["RandomForest"] = RandomForestClassifier(
            n_estimators=500)
        self.dict_models["LogisticRegression"] = LogisticRegression(
            penalty='none', tol=self.tol, max_iter=self.max_iter,
            solver='saga')
        self.dict_models["Neural_network"] = MLPClassifier()
        self.dict_models["KNN"] = KNeighborsClassifier()
        self.dict_models["Adaboost"] = AdaBoostClassifier()
        self.dict_models["GradientBoosting"] = GradientBoostingClassifier()
        self.dict_models["FIGS"] = FIGSClassifier()

        self.dict_parameters = {}
        self.dict_parameters["linearSVM"] = {
            'model__C': np.logspace(-4, 4, 50)}
        self.dict_parameters["RBFSVM"] = {'model__C': np.logspace(-4, 4, 25)}
        self.dict_parameters["RandomForest"] = {
            'model__n_estimators': [100, 250, 500, 1000]}
        self.dict_parameters["Adaboost"] = {
            'model__n_estimators': [100, 250, 500, 1000],
            'model__learning_rate': [0.001, 0.01, 0.1, 1.0]}
        self.dict_parameters["Neural_network"] = {'model__alpha': np.logspace(
            -4, 4, 25)}
        self.dict_parameters["KNN"] = {'model__n_neighbors': np.arange(1, 11)}
        self.dict_parameters["LogisticRegression"] = {}
        self.dict_parameters["GradientBoosting"] = {
            'model__n_estimators': [100, 250, 500, 1000],
            'model__learning_rate': [0.001, 0.01, 0.1, 1.0]}
        self.dict_parameters["FIGS"] = {
            'model__max_rules': [6, 8, 10, 12]
        }

        self.list_metrics = ['auc', 'accuracy', 'balanced_accuracy',
                             'sensitivity', 'specificity']

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

    def add_models(self, name_model, model, params):
        self.dict_models[name_model] = model
        self.dict_parameters[name_model] = params

    def fit(self, X, y, normalize=True, feat_select=True, balanced=True):
        # save nested cv results for each split (n_outer_split),
        # each model (len(name_modes)) and for each metrics
        # (AUC, accuracy, balanced, accuracy, sensitivity, specificity)

        rmv_zero_var = VarianceThreshold()
        X = rmv_zero_var.fit_transform(X)

        raw_results = np.zeros(
            (self.n_split_outer, len(self.names_models), self.n_split_inner))
        k = 0
        for idx_train, idx_test in self.cv_outer.split(X, y):
            print('Split number %i' % (k + 1))
            # split data
            X_train, X_test = X[idx_train, :], X[idx_test, :]
            y_train, y_test = y[idx_train], y[idx_test]

            for j, model in enumerate(self.names_models):
                print('Current model: ', model)
                steps = list()
                if not balanced:
                    steps.append(
                        ('sampler',
                         RandomUnderSampler(random_state=42)))
                if normalize:
                    steps.append(('scaler', StandardScaler()))
                if feat_select:
                    if X.shape[1] < 10:
                        raise ValueError(
                            'Not enough features to perform feature selection')
                    if normalize:
                        alpha_max = np.max(
                            np.abs(
                                StandardScaler().fit_transform(X_train).T @ y_train)) / 2
                    else:
                        alpha_max = np.max(np.abs(X_train.T @ y_train)) / 2
                    n_alphas = 25
                    alphas = np.geomspace(alpha_max, alpha_max / 100, n_alphas)
                    self.dict_parameters[model]['selecter__estimator__C'] = 1 / alphas
                    steps.append((
                        'selecter', SelectFromModel(
                            LogisticRegression(
                                penalty="l1", tol=self.tol, max_iter=self.max_iter,
                                solver='liblinear'))))
                steps.append(('model', self.dict_models[model]))
                pipeline = Pipeline(steps)
                search = GridSearchCV(
                    pipeline, self.dict_parameters[model],
                    scoring='roc_auc', n_jobs=-1, cv=self.cv_inner)

                # Perform GridSearch for the current model over the parameters
                search.fit(X_train, y_train)

                # get the best performing model fit on the whole training set
                best_model = search.best_estimator_

                estimated_proba = best_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, estimated_proba)
                estimated_classes = best_model.predict(X_test)

                confusion = confusion_matrix(y_test, estimated_classes)

                true_neg = confusion[0][0]
                false_neg = confusion[1][0]
                true_pos = confusion[1][1]
                false_pos = confusion[0][1]
                sensitivity = true_pos / (true_pos + false_neg)
                specificity = true_neg / (true_neg + false_pos)

                acc = accuracy_score(y_test, estimated_classes)
                balanced_acc = balanced_accuracy_score(
                    y_test, estimated_classes)

                raw_results[k, j, :] = np.array(
                    [auc, acc, balanced_acc, sensitivity, specificity])
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

        for j in range(self.raw_results.shape[0]):
            df_temp = pd.DataFrame(
                data=self.raw_results[j, :, 0])
            df_temp = df_temp.T
            df_temp.columns = self.names_models
            results = results.append(df_temp)
            name_row.append("Split %i" % (j+1))

        results.index = name_row
        return results

    def fit_single_model(self, X, y, name_model, normalize=True,
                         feat_select=True, balanced=True):
        steps = list()
        if not balanced:
            steps.append(('sampler', RandomUnderSampler(random_state=42)))
        if normalize:
            steps.append(('scaler', StandardScaler()))
        if feat_select:
            if normalize:
                alpha_max = np.max(np.abs(StandardScaler().fit_transform(X).T @ y)) / 2
            else:
                alpha_max = np.max(np.abs(X.T @ y)) / 2
            n_alphas = 10
            alphas = np.geomspace(alpha_max, alpha_max / 1_000, n_alphas)
            self.dict_parameters[name_model]['selecter__estimator__C'] = 1 / alphas
            steps.append((
                'selecter', SelectFromModel(
                    LogisticRegression(
                        penalty="l1", tol=self.tol, max_iter=self.max_iter,
                        solver='liblinear'))))
        steps.append(('model', self.dict_models[name_model]))
        pipeline = Pipeline(steps)
        search = GridSearchCV(
            pipeline, self.dict_parameters[name_model],
            scoring='roc_auc', n_jobs=-1, cv=self.cv_inner)

        # Perform GridSearch for the current model over the parameters
        search.fit(X, y)

        # get the best performing model fit on the whole training set
        best_model = search.best_estimator_
        return best_model
