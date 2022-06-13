from libsvmdata import fetch_libsvm
from train import NestedCV

X, y = fetch_libsvm("diabetes")
nested_cv = NestedCV()
nested_cv.fit(X, y, normalize=True, feat_select=False)