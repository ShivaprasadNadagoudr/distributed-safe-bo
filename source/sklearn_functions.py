from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
import numpy as np


def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=1000,
        n_features=45,
        n_informative=12,
        n_redundant=7,
        random_state=134985745,
    )
    return data, targets


data, targets = get_data()

# [C, gamma]
SVC_BOUNDS = [(-3, 2), (-4, -1)]
SVC_THRESHOLD = 0.85
SVC_NAME = "svc"


def svc_cv(C, gamma, data, targets):
    """SVC cross validation.

    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.

    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    estimator = SVC(C=C, gamma=gamma, random_state=2)
    cval = cross_val_score(estimator, data, targets, scoring="roc_auc", cv=4)
    return cval.mean()


def svc_crossval(X):
    """Wrapper of SVC cross validation.

    Notice how we transform between regular and log scale. While this
    is not technically necessary, it greatly improves the performance
    of the optimizer.
    """
    global data
    global targets
    X = np.atleast_2d(X)
    Y = []
    for [expC, expGamma] in X:
        C = 10 ** expC
        gamma = 10 ** expGamma
        y = svc_cv(C=C, gamma=gamma, data=data, targets=targets)
        Y.append(y)
    return np.reshape(Y, (-1, 1))


# pbounds = {
#     "n_estimators": (10, 250),
#     "min_samples_split": (2, 25),
#     "max_features": (0.1, 0.999),
# }
RFC_BOUNDS = [(10, 250), (2, 25), (0.1, 0.999)]
RFC_THRESHOLD = -0.4
RFC_NAME = "rfc"


def rfc_cv(n_estimators, min_samples_split, max_features, data, targets):
    """Random Forest cross validation.

    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.

    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    estimator = RFC(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=2,
    )
    cval = cross_val_score(estimator, data, targets, scoring="neg_log_loss", cv=4)
    return cval.mean()


def rfc_crossval(X):
    """Wrapper of RandomForest cross validation.

    Notice how we ensure n_estimators and min_samples_split are casted
    to integer before we pass them along. Moreover, to avoid max_features
    taking values outside the (0, 1) range, we also ensure it is capped
    accordingly.
    """
    global data
    global targets
    X = np.atleast_2d(X)
    Y = []
    for [n_estimators, min_samples_split, max_features] in X:
        y = rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=max(min(max_features, 0.999), 1e-3),
            data=data,
            targets=targets,
        )
        Y.append(y)
    return np.reshape(Y, (-1, 1))
