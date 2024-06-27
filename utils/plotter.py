import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Callable

from .definitions import ROOT_DIR
from .data_wrangler import (
    ccv_signature_features,
    create_knee_elbow_data,
)

from .analyser import PredictedFullCurve


def plot_ccv_evolution(
    data_dict: dict[str, dict],
    sample_cells: list[str],
) -> None:
    """
    Plot the evolution of constant-current voltage for sample cells.
    """

    data_dict = ccv_signature_features(
        data_dict=data_dict, num_cycles=-1, return_ccv=True
    )  # get constant-current voltage at discharge for all cycles (except the last)

    fig = plt.figure(figsize=(16, 5))
    for i, cell in enumerate(sample_cells):
        ax = fig.add_subplot(1, 4, i + 1)
        if i == 0:
            ax.set_ylabel("CC discharge voltage (V)", fontsize=16)

        if i != 0:
            ax.set_yticklabels([])

        ax.set_xlabel("Time (minutes)", fontsize=16)

        cycles = [int(cyc) for cyc in data_dict[cell].keys()]
        cmap = plt.get_cmap("gist_heat", len(cycles))

        for cycle in data_dict[cell].keys():
            # x_axis = np.arange(len(data_dict[cell][cycle][1])) + 1 ax.plot(data_dict[cell][cycle][0],
            # data_dict[cell][cycle][1], c=cmap(int(cycle)), linewidth=1, alpha=0.5)
            ax.plot(
                data_dict[cell][cycle][0] - min(data_dict[cell][cycle][0]),
                data_dict[cell][cycle][1],
                c=cmap(int(cycle)),
                linewidth=1,
                alpha=0.5,
            )
        ax.set_xlim(0, 16)

        # create normalizer
        vmin, vmax = 1, len(cycles)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # create ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            ticks=range(
                1, len(cycles) + int((vmax - vmin) / 2), int((vmax - vmin) / 2)
            ),
        )
        cbar.set_label("Cycles", fontsize=16)

        ax.text(
            0.02,
            0.1,
            cell,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

    plt.tight_layout()
    plt.savefig(fname=f"{ROOT_DIR}/plots/cc_voltage_evolution.png", bbox_inches="tight")

    return None


def plot_feature_target_correlation(data_dict: dict[str, dict]) -> None:

    features = ccv_signature_features(
        data_dict=data_dict,
        step_size=1,
        num_cycles=50,
        return_ccv=False,
    )
    targets = create_knee_elbow_data(data_dict=data_dict)
    merged_df = features.join(targets)

    target_cols = [
        "k-o",
        "k-p",
        "Qatk-o",
        "Qatk-p",
        "e-o",
        "e-p",
        "IRate-o",
        "IRate-p",
        "IRatEOL",
        "EOL",
    ]
    feature_cols = merged_df.drop(target_cols, axis=1).columns

    corr_matrix = merged_df.corr(method="pearson")
    corr_for_heatmap = corr_matrix.loc[target_cols, feature_cols]

    _, ax = plt.subplots(figsize=(4, 18))
    cax = inset_axes(
        ax,
        width="100%",
        height="2%",
        loc="lower left",
        bbox_to_anchor=(0, -0.09, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    sns.heatmap(
        corr_for_heatmap.T,
        vmin=-1,
        vmax=1,
        xticklabels=target_cols,
        yticklabels=feature_cols,
        linewidth=1,
        cmap="seismic",
        alpha=0.7,
        linecolor="black",
        ax=ax,
        cbar_ax=cax,
        cbar_kws={"orientation": "horizontal", "label": "Correlation"},
    )
    ax.figure.axes[-1].set_xlabel(r"Correlation ($\rho$)", size=14)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/correlation_heatmap_num_cycles_50.png",
        bbox_inches="tight",
    )

    return None


def plot_cc_discharge_profile(data_dict: dict[str, dict], sample_cell: str) -> None:

    current = data_dict[sample_cell]["cycle_dict"]["2"]["I"]
    voltage = data_dict[sample_cell]["cycle_dict"]["2"]["V"]
    time = data_dict[sample_cell]["cycle_dict"]["2"]["t"]
    discharge_time_filter = np.logical_and((time >= 31), (time <= 45.5))

    ylabels = [r"Current ($A$)", r"Voltage (V)"]
    fig = plt.figure(figsize=(11, 3))

    for i, profile in enumerate([current, voltage]):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.plot(time, profile)

        if i == 0:
            ax.hlines(
                y=current[discharge_time_filter][-1],
                xmin=31,
                xmax=45.5,
                linewidth=4,
                color="red",
                zorder=100,
                label="Discharge CC",
            )

        if i == 1:
            ax.plot(
                time[discharge_time_filter],
                profile[discharge_time_filter],
                linewidth=4,
                color="red",
                label="CC discharging voltage",
            )

        ax.axvline(x=31, color="red", linestyle="--")
        ax.axvline(x=45.5, color="red", linestyle="--")
        ax.set_xlabel("Time (minutes)", fontsize=16)
        ax.set_ylabel(ylabel=ylabels[i], fontsize=16)

        ax.legend(loc="lower left")

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/cc_dischrage_profiles.png", bbox_inches="tight"
    )

    return None


def axis_to_fig(axis: Any) -> Callable[[tuple], Any]:
    """
    Converts axis to fig object.

    Args:
    ----
         axis (object): axis object

    Returns:
    -------
            transformed axis oject.
    """

    fig = axis.figure

    def transform(coord: tuple | list):
        return fig.transFigure.inverted().transform(axis.transAxes.transform(coord))

    return transform


def add_sub_axes(axis: Any, rect: tuple | list) -> Any:
    """
    Adds sub-axis to existing axis object.

    Args:
    ----
         axis (object):        axis object
         rect (list or tuple): list or tuple specifying axis dimension

    Returns:
    -------
           fig object with added axis.
    """
    fig = axis.figure
    left, bottom, width, height = rect
    trans = axis_to_fig(axis)
    figleft, figbottom = trans((left, bottom))
    figwidth, figheight = trans([width, height]) - trans([0, 0])

    return fig.add_axes([figleft, figbottom, figwidth, figheight])


def plot_parity_history(parity_history: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(12, 20))

    for i, (target, history) in enumerate(parity_history.items()):

        ax = fig.add_subplot(5, 2, i + 1)
        ax.text(
            0.05,
            0.95,
            target,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        ax.scatter(
            history["y_train"],
            history["y_train_pred"],
            s=50,
            color="blue",
            alpha=0.5,
            label="Train",
        )
        ax.scatter(
            history["y_test"],
            history["y_test_pred"],
            s=50,
            color="red",
            alpha=0.5,
            label="Test",
        )
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against each other
        ax.plot(lims, lims, "k--", alpha=0.75, zorder=100)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if i % 2 == 0:
            ax.set_ylabel("Predicted values", fontsize=16)

        if i in [8, 9]:
            ax.set_xlabel("Observed values", fontsize=16)

        # embed histogram of residuals
        res_train = history["y_train"] - history["y_train_pred"]
        res_test = history["y_test"] - history["y_test_pred"]
        res = np.concatenate((res_train, res_test), casting="unsafe", dtype=float)

        subaxis = add_sub_axes(ax, [0.62, 0.17, 0.35, 0.2])
        subaxis.hist(res, bins=20, color="black", alpha=0.75, ec="black")
        subaxis.set_xlim(res.min(), -res.min())
        subaxis.set_xlabel("Residuals", fontsize=10)
        subaxis.set_ylabel("Frequency", fontsize=10)

        if i == 8:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                fontsize=16,
                bbox_to_anchor=(1.0, -0.2),
            )

    plt.savefig(fname=f"{ROOT_DIR}/plots/parity_plot.png", bbox_inches="tight")

    return None


def plot_predicted_full_curve(
    predicted_curve_history: dict[str, dict[str, PredictedFullCurve]], curve_name: str
) -> None:

    if curve_name == "QDischarge":
        x, y = (0.02, 0.2)
        figure_save_name = "capacity_fade"
        ylabel = "Capacity (Ah)"
        ylim = [0.85, 1.1]
    elif curve_name == "IR":
        x, y = (0.05, 0.95)
        figure_save_name = "ir_rise"
        ylabel = r"Internal Resistance ($\Omega$)"
        ylim = [0.014, 0.022]

    else:
        raise ValueError(
            "curve_name must be either 'QDischarge' or 'IR', "
            f"but {curve_name} is privided"
        )

    fig = plt.figure(figsize=(17, 9))

    for i, (cell, history) in enumerate(predicted_curve_history.items()):
        ax = fig.add_subplot(4, 4, i + 1)
        ax.text(
            x,
            y,
            cell,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        ax.plot(
            history[curve_name].actual_cycle,
            history[curve_name].actual_curve,
            "b--",
            label="Actual curve",
            linewidth=1.0,
        )
        ax.plot(
            history[curve_name].predicted_cycle,
            history[curve_name].predicted_curve,
            color="crimson",
            label="Predicted curve",
            linewidth=2.0,
        )
        ax.fill(
            np.append(
                history[curve_name].predicted_cycle_lb,
                history[curve_name].predicted_cycle_ub[::-1],
            ),
            np.append(
                history[curve_name].predicted_curve_lb,
                history[curve_name].predicted_curve_ub[::-1],
            ),
            color="crimson",
            label=r"90% CI",
            alpha=0.13,
        )

        if i == 13:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                fontsize=16,
                bbox_to_anchor=(0.8, -0.4),
            )

        ax.set_ylim(ylim)

    fig.text(
        0.08,
        0.5,
        ylabel,
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    fig.text(0.5, 0.07, "Cycles", ha="center", va="center", fontsize=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/{figure_save_name}_curve.png",
        bbox_inches="tight",
    )

    return None


def plot_feature_importance_analysis_history(
    analysis_df: pd.DataFrame, figure_save_name: str
) -> None:

    fig = plt.figure(figsize=(18, 3))

    df_index = np.array(analysis_df.index)

    for i, col in enumerate(analysis_df.columns):

        ax = fig.add_subplot(1, 5, i + 1)
        ax.text(
            0.6,
            0.95,
            col,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )

        col_importance = analysis_df[col].values
        sorted_index = np.argsort(col_importance)

        col_importance = col_importance[sorted_index]
        temp_index = df_index[sorted_index]

        ax.bar(
            temp_index[::-1][:10],
            col_importance[::-1][:10],
            color="blue",
            ec="black",
            alpha=0.78,
        )
        ax.tick_params(axis="x", rotation=90, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)

        if i != 0:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_ylabel("Feature importance", size=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/feature_importance_{figure_save_name}_bar.png",
        bbox_inches="tight",
    )

    return None


def plot_cycle_number_effect_history(
    list_of_cycles: list[int] | np.ndarray,
    history: dict[str, dict],
    model_type: str,
    evaluation_type: str,
) -> None:

    if evaluation_type == "crossval":
        ylabels = (
            ["Cross-validated errors (cycles)", "Cross-validated errors (cycles)"]
            if model_type == "cycle_at_model"
            else ["Cross-validated errors (Ah)", r"Cross-validated errors ($\Omega$)"]
        )

    elif evaluation_type == "test":
        ylabels = (
            ["Average test errors (cycles)", "Average test errors (cycles)"]
            if model_type == "cycle_at_model"
            else ["Average test errors (Ah)", r"Average test errors ($\Omega$)"]
        )

    _, ax = plt.subplots(1, 2, figsize=(16, 4))

    for i, (split, data) in enumerate(history.items()):
        ax[i].plot(
            list_of_cycles,
            data[f"{evaluation_type}_mae"],
            label="MAE",
            color="blue",
            markersize=5,
        )
        ax[i].fill_between(
            list_of_cycles,
            data[f"{evaluation_type}_mae_ci"][:, 0],
            data[f"{evaluation_type}_mae_ci"][:, 1],
            color="blue",
            alpha=0.1,
            label="MAE: 90% CI",
        )

        ax[i].set_xlabel("Cycle number threshold", fontsize=16)
        ax[i].set_ylabel(ylabels[i], fontsize=16)
        ax[i].set_title(split, fontsize=18)

        ax[i].plot(
            list_of_cycles,
            data[f"{evaluation_type}_rmse"],
            "s-",
            color="crimson",
            label="RMSE",
            markersize=5,
        )
        ax[i].fill_between(
            list_of_cycles,
            data[f"{evaluation_type}_rmse_ci"][:, 0],
            data[f"{evaluation_type}_rmse_ci"][:, 1],
            color="crimson",
            alpha=0.1,
            label="RMSE: 90% CI",
        )

    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=16,
        bbox_to_anchor=(1.0, -0.15),
    )

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/{evaluation_type}_{model_type}_num_cycles_effect.png",
        bbox_inches="tight",
    )

    return None


def plot_subsampling_time_effect_history(
    history: dict[str, list[str] | np.ndarray]
) -> None:

    _, ax = plt.subplots(1, 2, figsize=(16, 4.5))

    ax[0].plot(
        history["time_steps"],
        history["crossval_mae"],
        "D--",
        label="EOL: MAE",
        color="blue",
    )
    ax[0].fill_between(
        history["time_steps"],
        history["crossval_mae_ci"][:, 0],
        history["crossval_mae_ci"][:, 1],
        color="blue",
        alpha=0.15,
        label="MAE: 90% CI",
    )

    ax[1].plot(
        history["time_steps"],
        history["crossval_rmse"],
        "s-",
        label="EOL: RMSE",
        color="crimson",
    )
    ax[1].fill_between(
        history["time_steps"],
        history["crossval_rmse_ci"][:, 0],
        history["crossval_rmse_ci"][:, 1],
        color="crimson",
        alpha=0.15,
        label="RMSE: 90% CI",
    )

    ax[0].legend(loc="lower right")
    ax[1].legend(loc="lower right")

    ax[0].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)
    ax[0].set_ylabel("Cross-validation errors (cycles)", fontsize=16)
    ax[1].set_xlabel("Sub-sampling time steps (mins)", fontsize=16)

    plt.savefig(
        fname=f"{ROOT_DIR}/plots/subsampling_time_effect.png", bbox_inches="tight"
    )

    return None
