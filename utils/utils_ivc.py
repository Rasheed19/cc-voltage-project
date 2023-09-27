import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats
from utils import utils_noah
from config.definitions import ROOT_DIR
import importlib

importlib.reload(utils_noah)


def gradient_estimate(f, x=None):
    """
    A function that estimates the gradient of f at values in x.

    Args:
        f:  an array of the values of f
        x:  an array of points of evaluation

    Returns an array of gradient of f.
    """
    return np.gradient(f) if x is None else np.gradient(f, x)


def ccv_features(data_dict, step_size=1, return_ccv=False, n=50):
    CCV_multi_features = []
    CCV_dict = {}

    for cell in data_dict.keys():
        CCV_features = []

        # initialize a dictionary to store CCV for each cycle
        this_cycle = {}

        for cycle in list(data_dict[cell]['cycle_dict'].keys())[:n]:
            # get the discharge values
            i_values = utils_noah.get_charge_discharge_values(
                data_dict, 'I', cell, cycle, 'di')
            v_values = utils_noah.get_charge_discharge_values(
                data_dict, 'V', cell, cycle, 'di')
            t_values = utils_noah.get_charge_discharge_values(
                data_dict, 't', cell, cycle, 'di')

            # get the indices of the start and end of CC
            start_I, end_I = utils_noah.get_constant_indices(i_values, 'di')

            # get the corresponding voltages
            ccv = v_values[start_I:end_I + 1]

            # get the corresponding time
            cct = t_values[start_I:end_I + 1]

            # Interpolation of voltage curve
            actual_length = len(cct)
            interested_length = int((1 / step_size) * actual_length)

            ccv_intp = interp1d(cct, ccv)
            a, b = min(cct), max(cct)
            ccv = ccv_intp(np.linspace(a, b, interested_length))
            h = (b - a) / interested_length

            # calculate the gradient of ccv with respect to time
            grad_ccv = gradient_estimate(ccv, h)

            CCV_features.append([ccv.min(), ccv.max(), ccv.mean(), ccv.var(), scipy.stats.skew(ccv),
                                 scipy.stats.kurtosis(ccv, fisher=False),
                                 np.trapz(
                                     ccv, dx=h), grad_ccv[0], grad_ccv[-1], grad_ccv.min(), grad_ccv.max()
                                 ])

            this_cycle[cycle] = [cct, ccv]

        # get the multi-cycle feature
        CCV_features = np.array(CCV_features)
        union = []
        for i in range(len(CCV_features[0])):
            union += utils_noah.multi_cycle_features(CCV_features[:, i], n)

        CCV_multi_features.append(union)
        CCV_dict[cell] = this_cycle

    if return_ccv:
        return CCV_dict

    feature_names = (
        'min-ccv-', 'max-ccv-', 'mean-ccv-', 'var-ccv-', 'skew-ccv-', 'kurt-ccv-', 'area-ccv-', 'grad-ccv-start-',
        'grad-ccv-end-', 'grad-ccv-min-', 'grad-ccv-max-'
    )
    return pd.DataFrame(
        data=np.array(CCV_multi_features),
        columns=[
            ft + item
            for ft in feature_names
            for item in utils_noah.strings_multi_cycle_features(n)
        ],
        index=data_dict.keys(),
    )


def plot_CCV_features(data_dict, ylabel=None, ylim=None, sample_cells=None, option=1):
    # sourcery skip: low-code-quality
    if option == 1:
        # get the cells belonging to the same batch
        b1 = [cell for cell in data_dict.keys() if cell[:2] == 'b1']
        b2 = [cell for cell in data_dict.keys() if cell[:2] == 'b2']
        b3 = [cell for cell in data_dict.keys() if cell[:2] == 'b3']

        x_labels = dict(zip(data_dict['b1c0'].keys(),
                            ['Cycles', r'Internal resistance ($\Omega$)', 'Min of CCV (V)', 'Max of CCV (V)',
                             'Mean of CCV (V)', 'Variance of CCV (V)', 'Skewness of CCV', 'Kurtosis of CCV',
                             'Area under CC Voltage Curve', 'Capacity (Ah)']))

        for batch in [b1, b2, b3]:
            fig, ax = plt.subplots(3, 3, figsize=(20, 15))
            i = 0
            for feature in data_dict['b1c0'].keys():
                if feature not in [ylabel]:
                    for cell in batch:
                        ax[i // 3, i % 3].plot(data_dict[cell][ylabel], data_dict[cell][feature], 'o', linewidth=1,
                                               markersize=2)
                        ax[i // 3, i %
                            3].set_ylabel(x_labels[feature], fontsize=14)
                        ax[i // 3, i %
                            3].set_xlabel(x_labels[ylabel], fontsize=14)
                        ax[i // 3, i % 3].set_ylim(ylim)
                    i += 1
            i = 0

            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles, labels)

            plt.show()

    elif option == 2:
        fig = plt.figure(figsize=(15, 3.5))
        for i, cell in enumerate(sample_cells):
            ax = fig.add_subplot(1, 4, i + 1)
            # ax.text(0.05, 0.1, cell, transform=ax.transAxes,
            #       fontsize=16, fontweight='bold', va='top')

            if i == 0:
                ax.set_ylabel('CC discharge voltage (V)', fontsize=16)

            if i != 0:
                ax.set_yticklabels([])

            ax.set_xlabel('Time (minutes)', fontsize=16)

            cycles = [int(cyc) for cyc in data_dict[cell].keys()]
            cmap = plt.get_cmap('gist_heat', len(cycles))

            for cycle in data_dict[cell].keys():
                # x_axis = np.arange(len(data_dict[cell][cycle][1])) + 1 ax.plot(data_dict[cell][cycle][0],
                # data_dict[cell][cycle][1], c=cmap(int(cycle)), linewidth=1, alpha=0.5)
                ax.plot(data_dict[cell][cycle][0] - min(data_dict[cell][cycle][0]), data_dict[cell][cycle][1],
                        c=cmap(int(cycle)), linewidth=1, alpha=0.5)
            ax.set_xlim(0, 16)

            """
            # Normalizer
            vmin, vmax = 1, len(cycles)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # creating ScalarMappable
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                                ticks=range(1, len(cycles) + int((vmax - vmin) / 2), int((vmax - vmin) / 2)))
            cbar.set_label('Cycles', fontsize=16)
            """

            # Add text with an arrow pointing to a specific point on the plot
            ax.annotate('Decreasing', xy=(10, 2.6), xytext=(8, 3.3),
                        arrowprops=dict(facecolor='black', linewidth=0.1), size=14)

            # ax.text(0.02, 0.1, cell, transform=ax.transAxes,
            #                       fontsize=16, fontweight='bold', va='top')

        plt.savefig(fname=f"{ROOT_DIR}/plots/ccv_over_cycles", bbox_inches='tight')
