import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV


#ARIMA
def rolling_arima(train, test, endog, order, exog=None, dates=None, freq=None, missing='none'):
    """ Perform rolling forecast on ARIMA

    :param train: training set
    :param test: test set
    :param order:
    :return:
    """
    history = list(train)
    predictions = []
    for row in test:
        model = sm.tsa.ARIMA(endog=history, order=order, exog=exog, dates=dates, freq=freq, missing=missing)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        predictions.append(yhat)
        history.append(row)

    return predictions
