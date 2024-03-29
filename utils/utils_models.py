import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import scipy.signal as sg
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import utils_gn
from config.definitions import ROOT_DIR
import importlib

importlib.reload(utils_gn)
xgb.set_config(verbosity=0)


def knee_elbow_detection(x_data,
                         y_data,
                         type,
                         want_clean_data=False,
                         p0=None,
                         p0_db=None,
                         p0_exp=None,
                         plot=False,
                         ylabel=None,
                         ylim=None,
                         title=None,
                         point_name1=None,
                         point_name2=None
                         ):
    """
    Function that detect knees and elbows by fitting Bacon-Watts and Double Bacon-Watts to a given data.

    Args:
    ----
        -x_data:      an array of independent variable values
        -y_data:      an array of dependent variable values
        -type:        specifies which to detect: "knee" or "elbow"
        -p0:          an array of initial values for Bacon-Watts model
        -p0_db:       an array of initial values for Double Bacon-Watts model
        -p0_exp:      an array of initial values for exponential model
        -plot:        a boolean, either to plot the results or not
        -ylabel:      y-axis label
        -ylim:        y_axis limit
        -title:       figure title
        -point_name1: name of the marked point in Bacon-Watt
        -point_name2: name of the marked point in Double Bacon-Watt

    Returns:
    -------
           cleaned data/knees/elbows
    """

    # Define the Bacon-Watts and Double Bacon-Watts models
    def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
        return alpha0 + alpha1 * (x - x1) + alpha2 * (x - x1) * np.tanh((x - x1) / 1e-8)

    def double_bacon_watts_model(x, alpha0, alpha1, alpha2, alpha3, x0, x2):
        return alpha0 + alpha1 * (x - x0) + alpha2 * (
            x - x0) * np.tanh((x - x0) / 1e-8) + alpha3 * (x - x2) * np.tanh((x - x2) / 1e-8)

    # Define the exponential model for data transformation
    def exponential_model(x, a, b, c, d, e):
        return a * np.exp(b * x - c) + d * x + e

    # Remove outliers from y_data
    clean_data = sg.medfilt(y_data, 5)

    # Get the length of clean data
    cl = len(clean_data)

    # Fit isotonic regression to data to obtain monotonic data
    if type == 'knee':
        isotonic_reg = IsotonicRegression(increasing=False)
    elif type == 'elbow':
        isotonic_reg = IsotonicRegression()
    clean_data = isotonic_reg.fit_transform(x_data, clean_data)

    # Force convexity on the cleaned y_data to prevent early detection of onset
    if (p0_exp is None) and type == 'knee':
        p0_exp = [-4, 5e-3, 10, 0, clean_data[0]]
        bounds = ([-np.inf, 0, 0, -1, 0], [0, np.inf, np.inf, 0, np.inf])
    elif (p0_exp is None) and type == 'elbow':
        p0_exp = [4, 0.03, 22, 0, clean_data[0]]
        bounds = (0, np.inf)
    popt_exp, _ = curve_fit(exponential_model, x_data,
                            clean_data, p0=p0_exp, bounds=bounds)
    clean_data = exponential_model(x_data, *popt_exp)

    if want_clean_data:
        return clean_data

    # Fit the Bacon-Watts model
    if (p0 is None) and type == 'knee':
        p0 = [1, -1e-4, -1e-4, cl * .7]
        bw_bounds = ([-np.inf, -np.inf, -np.inf, cl / 4],
                     [np.inf, np.inf, np.inf, cl])
    elif (p0 is None) and type == 'elbow':
        p0 = [1, 1, 1, cl / 1.5 + 1]
        bw_bounds = ([-np.inf, -np.inf, -np.inf, cl / 1.5],
                     [np.inf, np.inf, np.inf, cl])
    popt, pcov = curve_fit(bacon_watts_model, x_data,
                           clean_data, p0=p0, maxfev=50000, bounds=bw_bounds)
    confint = [popt[3] - 1.96 * np.diag(pcov)[3],
               popt[3] + 1.96 * np.diag(pcov)[3]]

    # Fit the Double Bacon-Watts
    if (p0_db is None) and type == 'knee':
        p0_db = [popt[0], popt[1] + popt[2] / 2, popt[2],
                 popt[2] / 2, .8 * popt[3], 1.1 * popt[3]]
        dbw_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, cl / 4,
                      cl / 2], [np.inf, np.inf, np.inf, np.inf, cl, cl])
    elif (p0_db is None) and type == 'elbow':
        p0_db = [1, 1, 1, 1, cl / 1.5 + 1, cl / 1.5 + 3]
        dbw_bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, cl / 4,
                      cl / 4], [np.inf, np.inf, np.inf, np.inf, cl, cl])
    popt_db, pcov_db = curve_fit(double_bacon_watts_model, x_data, clean_data, p0=p0_db, maxfev=50000,
                                 bounds=dbw_bounds)
    confint_db = [popt_db[4] - 1.96 * np.diag(pcov_db)[4],
                  popt_db[4] + 1.96 * np.diag(pcov_db)[4]]

    if plot:
        # Plot results
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(x_data, y_data, 'b--', label='True data', alpha=.7)
        ax[0].plot(x_data, clean_data, 'g-', label='Cleaned data')
        ax[0].plot(x_data, bacon_watts_model(x_data, *popt),
                   'r-', linewidth=2, label='Bacon-Watts')
        ax[0].plot([popt[3]], [bacon_watts_model(popt[3], *popt)], marker="o", markersize=5, markeredgecolor="black",
                   markerfacecolor="black", label=point_name1)
        ax[0].axvline(x=popt[3], color='black', linestyle='--')
        ax[0].fill_betweenx(ylim, x1=confint[0], x2=confint[1],
                            color='k', alpha=.3, label='95% C.I')
        ax[0].set_xlabel('Cycle number', fontsize=16)
        ax[0].set_ylabel(ylabel, fontsize=16)
        ax[0].grid(alpha=.3)
        ax[0].set_ylim(ylim)
        ax[0].set_title(title, fontsize=16)
        ax[0].legend()

        ax[1].plot(x_data, y_data, 'b--', label='True data', alpha=.7)
        ax[1].plot(x_data, clean_data, 'g-', label='Cleaned data')
        ax[1].plot(x_data, double_bacon_watts_model(
            x_data, *popt_db), 'r-', label='Double Bacon-Watts')
        ax[1].plot([popt_db[4]], [double_bacon_watts_model(popt_db[4], *popt_db)], marker="o", markersize=5,
                   markeredgecolor="black", markerfacecolor="black", label=point_name2)
        ax[1].axvline(x=popt_db[4], color='black', linestyle='--')
        ax[1].fill_betweenx(
            ylim, x1=confint_db[0], x2=confint_db[1], color='k', alpha=.3, label='95% C.I')
        ax[1].set_xlabel('Cycle number', fontsize=16)
        ax[1].set_ylabel(ylabel, fontsize=16)
        ax[1].grid(alpha=.3)
        ax[1].set_ylim(ylim)
        ax[1].set_title(title, fontsize=16)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    if type == 'knee':
        # Calculate values at knee-point and knee-onset
        ttk_o = int(popt_db[4] - 1)  # knee-onset
        ttk_p = int(popt[3] - 1)  # knee-point
        rul = int(len(y_data) - popt_db[4])  # remaining useful life
        q_at_k_o = double_bacon_watts_model(
            popt_db[4], *popt_db)  # capacity at knee-onset
        q_at_k_p = bacon_watts_model(popt[3], *popt)  # capacity at knee-point

        return ttk_o, ttk_p, rul, q_at_k_o, q_at_k_p

    if type == 'elbow':
        # Calculate values at knee-point and knee-onset
        tte_o = int(popt_db[4] - 1)  # elbow-onset
        tte_p = int(popt[3] - 1)  # elbow-point
        ir_at_e_o = double_bacon_watts_model(
            popt_db[4], *popt_db)  # resistance at elbow-onset
        # resistance at elbow-point
        ir_at_e_p = bacon_watts_model(popt[3], *popt)

        return tte_o, tte_p, ir_at_e_o, ir_at_e_p


def metrics_calculator(y_true, y_pred, multi=False):
    """
    A function that calculates regression metrics.

    Arguments:
              y_true:  an array containing the true values of y
              y_pred:  an array containing the predicted values of y
              multi:   a boolean to specify multi-output option
    Returns:
            MAE, RMSE
    """
    if multi:
        return {'MAE': mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
                'MAPE': mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values'),
                'RMSE': mean_squared_error(y_true, y_pred, multioutput='raw_values', squared=False)
                }

    return {'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False)
            }


def axis_to_fig(axis):
    fig = axis.figure

    def transform(coord):
        return fig.transFigure.inverted().transform(
            axis.transAxes.transform(coord))

    return transform


def add_sub_axes(axis, rect):
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width, height]) - trans([0, 0])
    return fig.add_axes([figleft, figbottom, figwidth, figheight])


def kfold_cross_validation(X, y, model, cv):
    """
    A function that performs k-fold cross validation.

    Ars:
    ---
              X, y:               training set
              model:              fitted model to be validated
              cv:                 int or cv object like RepeatedKFold
    Returns:
    -------
            a dictionary with key as test score and value as (score value, std of score value).
    """

    # define metrics to be used
    metrics = {'MAE': 'neg_mean_absolute_error',
               'RMSE': 'neg_root_mean_squared_error',
               'MAPE': 'neg_mean_absolute_percentage_error'
               }
    # metrics = {'MAE': 'neg_mean_absolute_error', 'RMSE': 'neg_root_mean_squared_error'}

    # calculate scores
    scores = cross_validate(model, X, y, scoring=metrics, cv=cv, n_jobs=-1)
    scores_summary = {
        key: abs(val).mean()
        for key, val in scores.items()
        if key in ['test_' + metric for metric in metrics]
    }
    scores_raw = {
        key: abs(val)
        for key, val in scores.items()
        if key in ['test_' + metric for metric in metrics]
    }

    return scores_summary, scores_raw


def antilog(x):
    return 10 ** x


class ModelPipeline:

    def __init__(self, params, transform_target):
        self.params = params
        self.best_model = None
        self.transform_target = transform_target

    def fit(self, X, y):
        if self.transform_target:
            self.best_model = TransformedTargetRegressor(
                MultiOutputRegressor(XGBRegressor(**self.params)),
                func=np.log10,
                inverse_func=antilog
            )

            self.best_model.fit(X, y)
            return self.best_model

        self.best_model = MultiOutputRegressor(XGBRegressor(**self.params))
        self.best_model.fit(X, y)
        return self.best_model


class ModifiedQuadraticSpline:

    def __init__(self):
        self.sol = None
        self.points = None

    def fit(self, x, y):
        A = np.zeros((9, 9))
        A[0:2, 0:3] = np.array([[1, x[0], x[0] ** 2], [1, x[1], x[1] ** 2]])
        A[2:4, 3:6] = np.array([[1, x[1], x[1] ** 2], [1, x[2], x[2] ** 2]])
        A[4:6, 6:9] = np.array([[1, x[2], x[2] ** 2], [1, x[3], x[3] ** 2]])
        A[6, 1], A[6, 2], A[6, 4], A[6, 5] = 1, 2 * x[1], -1, -2 * x[1]
        A[7, 4], A[7, 5], A[7, 7], A[7, 8] = 1, 2 * x[2], -1, -2 * x[2]
        A[8, 2] = 1

        b = np.array([y[0], y[1], y[1], y[2], y[2], y[3], 0., 0., 0.])

        self.sol = np.linalg.solve(A, b)
        self.points = x

    def transform(self, x):
        if x[0] < self.points[0] or x[-1] > self.points[-1]:
            return "Out of range of interpolation"
        res = []
        for el in x:
            if self.points[0] <= el < self.points[1]:
                res.append(self.sol[0] + self.sol[1] *
                           el + self.sol[2] * el ** 2)
            elif self.points[1] <= el < self.points[2]:
                res.append(self.sol[3] + self.sol[4] *
                           el + self.sol[5] * el ** 2)
            elif self.points[2] <= el <= self.points[3]:
                res.append(self.sol[6] + self.sol[7] *
                           el + self.sol[8] * el ** 2)
        return res


def confidence_interval_estimate(prediction, variance, confidence_level=0.95):
    """
    Function that estimates a confidence interval for a point prediction.

    Args:
    ----
         prediction:        predicted value
         variance:          estimated variance
         confidence_level:  level of certainty

    Returns:
    -------
           confidence interval for a given prediction.
    """
    tail_prob = (1 - confidence_level) / 2

    upper_z = stats.norm.ppf(1 - tail_prob)
    lower_z = stats.norm.ppf(tail_prob)

    return np.sqrt(variance) * prediction * np.array([lower_z, upper_z]) + prediction


def prediction_interval(X, y, model, n_bootstraps, target_list, predictions, confidence_level=0.95, plot_dist=False):
    """
    Function that calculates prediction interval for given
    predictions using the idea of bootstrapping.

    Args:
    ----
        X, y:          training set
        model:         unfitted model
        n_bootstraps:  number of bootstraps
        target_list:   list of target variables
        predictions:   predicted values
        plot_dist:     specify whether to plot distribution of residuals or not

    Returns:
    -------
            prediction intervals, variances of residuals
    """
    residuals = []

    for _ in range(n_bootstraps):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
        md = model.fit(X_tr, y_tr)
        pred = md.predict(X_val)
        residuals.append(((y_val - pred) / y_val).tolist())

    residuals = np.array(residuals)

    temp = []
    var_list = []

    if plot_dist:
        fig, ax = plt.subplots(1, len(target_list), figsize=(20, 4))

    for j in range(len(target_list)):
        for i in range(n_bootstraps):
            temp.append(residuals[i, :, j].tolist())
        temp = np.array(temp)
        if plot_dist:
            ax[j].set_title(target_list[j], fontsize=16)
            ax[j].grid()
            sns.distplot(temp.ravel(), kde=True, ax=ax[j])
            if j > 0:
                ax[j].set_ylabel('')
            else:
                ax[j].set_ylabel('Density of prediction errors', fontsize=14)
        var_list.append(np.var(temp.ravel()))
        temp = []

    if plot_dist:
        plt.show()

    return [
        [
            confidence_interval_estimate(el, var_list[j], confidence_level)
            for el in predictions[:, j]
        ]
        for j in range(len(target_list))
    ], var_list


def confidence_interval_metrics(
    actual,
    predictions,
    n_bootstraps,
    target_list,
    metric_type,
    alpha=0.05
):
    """
    Function that set up a confidence interval for model metrics.

    Args:
    ----
        actual:      actual values
        predictions: predicted values
        n_bootstraps: number of bootstraps
        metric_type: type of metric

    Returns:
    -------
            a list of metrics for the targets.
    """
    target_metric_ci = []
    errors = actual - predictions
    alpha_tail = alpha / 2
    for i in range(len(target_list)):
        metric_estimates = []

        for _ in range(n_bootstraps):
            re_sample_idx = np.random.randint(
                0, len(errors[:, i]), errors[:, i].shape
            )

            if metric_type == 'mae':
                metric_estimates.append(
                    np.mean(np.abs(errors[:, i][re_sample_idx])))
            elif metric_type == 'rmse':
                metric_estimates.append(
                    np.sqrt(np.mean((errors[:, i][re_sample_idx]) ** 2)))
            elif metric_type == 'mape':
                metric_estimates.append(
                    np.mean(abs((errors[:, i][re_sample_idx]) / actual[:, i][re_sample_idx])))

        sorted_estimates = np.sort(np.array(metric_estimates))
        conf_interval = [np.round(sorted_estimates[int(alpha_tail * n_bootstraps)], 6),
                         np.round(sorted_estimates[int((1 - alpha_tail) * n_bootstraps)], 6)]

        target_metric_ci.append(np.array(conf_interval))

    return target_metric_ci


def confidence_interval_any(values, n_bootstraps, metric_type=None, alpha=0.05):
    alpha_tail = alpha / 2
    metric_estimates = []
    values = np.array(values)

    for _ in range(n_bootstraps):
        re_sample_idx = np.random.randint(0, len(values), values.shape)

        if metric_type == 'rmse':
            metric_estimates.append(
                np.sqrt(np.mean((values[re_sample_idx]) ** 2)))
        else:
            metric_estimates.append(np.mean(values[re_sample_idx]))

    sorted_estimates = np.sort(np.array(metric_estimates))
    return [
        np.round(sorted_estimates[int(alpha_tail * n_bootstraps)], 6),
        np.round(sorted_estimates[int((1 - alpha_tail) * n_bootstraps)], 6),
    ]


def modified_spline_evaluation(x, y, eval_points):
    """
    Function that fits and evaluate spline at given points.

    Args:
    ----
         x, y:         arrays of points to be used to fit the spline
         eval_points:  points of evaluation

    Returns:
    -------
            array of evaluations.
    """
    spl = ModifiedQuadraticSpline()
    spl.fit(x, y)

    return spl.transform(eval_points)

def feature_importance_analysis(
    model,
    feature_names,
    target_list
):
    """
    Calculate feature importance for a fitted model.

    Args:
    ----
         model:         model object
         feature_names: name of the features 
         target_list:   list of targets
    
    Returns:
    -------
            a pandas dataframe of feature importance.
    """

    # Create a lambda function to scale importance values to the interval [0, 1]
    scaler = lambda x: (x-x.min()) / (x.max()-x.min())
    feature_importance = [scaler(model.regressor_.estimators_[i].feature_importances_) for i in range(len(target_list))]

    # Cast feature importance list to a 2D numpy array
    feature_importance = np.array(feature_importance)

    return pd.DataFrame(
        data=feature_importance.T,
        columns=target_list,
        index=feature_names
    )

def plot_feature_importance(df: pd.DataFrame, fname: str) -> None:

    fig = plt.figure(figsize=(18, 3))
    df_index = np.array(df.index)

    for i, item in enumerate(df.columns):
        
        ax = fig.add_subplot(1, 5, i+1)
        ax.text(0.6, 0.95, item, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
        
        this_importance = df[item].values
        sort_index = np.argsort(this_importance)

        this_importance = this_importance[sort_index]
        this_index = df_index[sort_index]

        ax.bar(this_index[::-1][:10], this_importance[::-1][:10], color='blue', ec='black', alpha=0.78)
        ax.tick_params(axis='x', rotation=90, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        if i != 0:
            ax.set_yticklabels([])
        
        if i == 0:
            ax.set_ylabel('Feature importance', fontsize=16)
    
    plt.show()
    plt.savefig(fname=f"{ROOT_DIR}/plots/{fname}", bbox_inches='tight')

    return None
        

