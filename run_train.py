import click

from pipelines import eda_pipeline, training_pipeline
from utils.helper import load_yaml_file
from utils.definitions import ROOT_DIR


@click.command(
    help="""
    Entry point for running training pipeline.
    """
)
@click.option(
    "--not-loaded",
    is_flag=True,
    default=False,
    help="""If given, raw data will be loaded and 
    cleaned.
        """,
)
@click.option(
    "--no-proposed-split",
    is_flag=True,
    default=False,
    help="""If given, train-test split used 
    in this study will not be used for modelling.
        """,
)
@click.option(
    "--num-cycles",
    default=50,
    type=click.IntRange(min=10, max=100, clamp=False),
    help="""The number of cycles of CCV profiles
    to use for modelling.
        """,
)
@click.option(
    "--exclude-b8",
    is_flag=True,
    default=False,
    help="""If given, cells from batch 8
    will be excluded from the data used for
    modelling.
        """,
)
@click.option(
    "--include-curve-prediction",
    is_flag=True,
    default=False,
    help="""If given, full curve predictions
    of sample cells in the configs/data_config.yaml
    will be part of pipeline run.
        """,
)
@click.option(
    "--include-analysis",
    is_flag=True,
    default=False,
    help="""If given, parity analysis (which 
    generates parity plots) and feature 
    importance analysis will be part of
    pipeline run.
        """,
)
@click.option(
    "--only-eda",
    is_flag=True,
    default=False,
    help="""If given, only eda pipeline will be run.
        """,
)
def main(
    not_loaded: bool = False,
    no_proposed_split: bool = False,
    num_cycles: int = 50,
    step_size: int = 1,
    exclude_b8: bool = False,
    include_curve_prediction: bool = False,
    include_analysis: bool = False,
    only_eda: bool = False,
) -> None:

    MODEL_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/configs/model_config.yaml")
    DATA_CONFIG = load_yaml_file(path=f"{ROOT_DIR}/configs/data_config.yaml")

    if only_eda:
        eda_pipeline(
            not_loaded=not_loaded,
            no_proposed_split=no_proposed_split,
            test_size=MODEL_CONFIG["test_size"],
        )

        return None

    training_pipeline(
        not_loaded=not_loaded,
        no_proposed_split=no_proposed_split,
        num_cycles=num_cycles,
        step_size=step_size,
        test_size=MODEL_CONFIG["test_size"],
        param_space=MODEL_CONFIG["param_space"],
        exclude_b8=exclude_b8,
        include_curve_prediction=include_curve_prediction,
        include_analysis=include_analysis,
        sample_cells=DATA_CONFIG["sample_cells"],
    )

    return None


if __name__ == "__main__":
    main()
