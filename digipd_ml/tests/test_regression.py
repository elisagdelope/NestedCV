from libsvmdata import fetch_libsvm
from train import NestedCV

X, y = fetch_libsvm("bodyfat")
nested_cv = NestedCV()
nested_cv.fit(X, y, normalize=False, feat_select=False)
