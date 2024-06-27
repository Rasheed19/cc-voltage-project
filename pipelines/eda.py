import logging

from steps import data_loader, data_splitter
from utils.plotter import (
    plot_ccv_evolution,
    plot_feature_target_correlation,
    plot_cc_discharge_profile,
)
from utils.helper import config_logger


def eda_pipeline(
    not_loaded: bool,
    test_size: float,
    no_proposed_split: bool,
) -> None:

    config_logger()
    logger = logging.getLogger(__name__)

    loaded_data = data_loader(not_loaded=not_loaded)
    train_data, _ = data_splitter(
        loaded_data=loaded_data,
        test_size=test_size,
        no_proposed_split=no_proposed_split,
    )

    logger.info("Plotting CCV evolution...")
    sample_cells = [
        "b1c30",
        "b2c30",
        "b3c27",
        "b8c7",
    ]
    plot_ccv_evolution(
        data_dict={k: train_data[k] for k in sample_cells},
        sample_cells=sample_cells,
    )

    logger.info("Plotting feature-target correlation...")
    plot_feature_target_correlation(data_dict=train_data)

    logger.info("Plotting constant-current discharge profiles...")
    plot_cc_discharge_profile(data_dict=train_data, sample_cell="b1c30")

    logger.info(
        "EDA pipeline finished successfully. "
        "Check the 'plots' folder for the results."
    )

    return None
