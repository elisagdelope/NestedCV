from libsvmdata import fetch_libsvm
from digipd_ml.supervised.classification import NestedCV

X, y = fetch_libsvm("diabetes")
nested_cv = NestedCV()
nested_cv.names_models = ["linearSVM"]
nested_cv.fit(X, y, normalize=True, feat_select=False)
