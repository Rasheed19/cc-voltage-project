import numpy as np
import pandas as pd
from utils import utils_gn
import importlib
from scipy.signal import find_peaks

importlib.reload(utils_gn)


def strings_multi_cycle_features(n=50):
    """
    Create feature names corresponding to cycle number n
    """
    return 'f0', f'f{int(n / 2)}', f'f{str(n)}', f'f{str(n)}-0', 'fdiff'
  

def multi_cycle_features(feature_values, n=50):
    """
    Generate cross-cycle features for array of feature values 
    (feature_values) corresponding to a given cycle number n.
    """
    try:

        # Take 10% of n
        i = int(0.1 * n)

        # Create features corresponding to n
        y_0 = np.median(feature_values[:i])
        y_med = np.median(feature_values[int((n / 2) - i):int((n / 2) + i)])
        y_end = np.median(feature_values[-i:])
        y_endm0 = y_end - y_0
        y_diff = (y_end - y_med) - (y_med - y_0)

        return [y_0, y_med, y_end, y_endm0, y_diff]
    
    except TypeError:
        print('n must be integer and >= 10')


def get_charge_discharge_values(data_dict, col_name, cell, cycle, option):
    """
    Function that extract only charge/discharge values of a given observed quantity.

    Args:
    ----
        data_dict (dict): a dictionary of battery cycling data
        col_name (str):   a string denoting name of observed quantity; e.g, 'I' for current
        cell (str):       a string denoting name of cell
        cycle (str):      a string denoting cycle number; e.g, '2'
        option (str):     a string specifying either pulling up charge/discharge values;
                          "ch": charge, "di": discharge
    
    Returns:
    -------
           returns extracted charge/discharge values
    """
    # An outlier in b1c2 at cycle 2176, measurement is in seconds and thus divide it by 60
    if cell == 'b1c2' and cycle == '2176':
        summary_charge_time = data_dict[cell]['summary']['chargetime'][int(cycle) - 2] / 60
    else:
        summary_charge_time = data_dict[cell]['summary']['chargetime'][int(cycle) - 2]

    values = data_dict[cell]['cycle_dict'][cycle][col_name]

    if option == 'ch':
        return np.array(values[data_dict[cell]['cycle_dict'][cycle]['t'] - summary_charge_time <= 1e-10])
    if option == 'di':
        return np.array(values[data_dict[cell]['cycle_dict'][cycle]['t'] - summary_charge_time > 1e-10])


def get_constant_indices(feature, option):
    """
    This function generates indices corresponding to the start
    and the end of constant values of a given feature.

    Args:
    ----
             feature (list/array):     a list of considered feature, e.g. current, voltage
             option (str):             a string to provide option for charge ('ch') and discharge ('di') indices
    
    Returns:
    -------
            tuple; start, end indices constant values for a given feature. 
    """

    constant_feature_list = []
    constant_feature_index = []

    for i in range(1, len(feature)):
        if abs(feature[i - 1] - feature[i]) <= 1e-2:
            constant_feature_list.append(feature[i - 1])
            constant_feature_index.append(i - 1)

    if option == 'ch':
        det_value = np.max(constant_feature_list)
        opt_list = [i for i, element in zip(constant_feature_index, constant_feature_list) if
                    np.round(det_value - element, 2) <= 0.5]

        return opt_list[0], opt_list[-1]

    if option == 'di':
        det_value = np.min(constant_feature_list)
        opt_list = [i for i, element in zip(constant_feature_index, constant_feature_list) if
                    np.round(element - det_value, 2) <= 0.5]
        return opt_list[0], opt_list[-1]



def cycle_life(data_dict):
    """
    Function that returns the cycle life/eol of cells.

    Args:
    ----
         data_dict (dict): a dictionary of battery cycling data

    Returns:
    -------
           returns a list of cycle life/eol of cells.
    """

    cycle_life = []

    for cell in data_dict.keys():
        qd = data_dict[cell]['summary']['QDischarge']
        qd_eol = qd >= 0.88  # we focus on the definition of eol: cycle number at 80% of nominal capacity
        qd = qd[qd_eol]
        cycle_life.append(len(qd))

    return pd.DataFrame(data=cycle_life, columns=['cycle_life'], index=data_dict.keys())
