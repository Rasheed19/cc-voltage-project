# cc-voltage-project
This repository contains the codes for all the experiments performed in the paper [R. Ibraheem, C. Strange, G. dos Reis, Capacity and internal resistance of lithium-ion batteries: Full degradation curve prediction from voltage response at constant current at discharge, Journal of Power Sources 556 (2023)232477.](https://www.sciencedirect.com/science/article/pii/S0378775322014549)

## Folder analysis

   - **config**: configuration files
   - **experiments**: jupyter notebook files for all experiments 
   - **utils**: custom modules used in the project

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/cc-voltage-project.git
    ```
1. Navigate to the root folder, i.e., `cc-voltage-project` and create a python virtual environment by running
    ```
    python3 -m venv .venv
    ``` 
    on a linux machine or

    ```
    python -m venv .venv
    ```
    on a window machine
1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Prepare all modules by running
    ```
    pip install -e .
    ```
1. Create a folder named **data** in the root directory **cc-voltage-project**. Download the following data and put them in this folder:
    - all the batches of data in this link https://data.matr.io/1/ which are the data for the papers [Data driven prediciton of battery cycle life before capacity degradation by K.A. Severson, P.M. Attia, et al](https://www.nature.com/articles/s41560-019-0356-8) and [Attia, P.M., Grover, A., Jin, N. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020).](https://doi.org/10.1038/s41586-020-1994-5)
    - the internal resistance data used to complement batch 8 can be downloaded from https://doi.org/10.7488/ds/2957 which is published in the paper [Strange, C.; Li, S.; Gilchrist, R.; dos Reis, G. Elbows of Internal Resistance Rise Curves in Li-Ion Cells. Energies 2021, 14, 1206.](https://doi.org/10.3390/en14041206)

1. Create folders named **plots** and **models** in the root directory **cc-voltage-project** to store the generated figures and models respectively.

1. Start running jupyter notebooks in the **experiments** folder.

1. When you are done experimenting, deactivate the virtual environment by running
    ```
    deactivate
    ```

If you use this work in your project, please cite:

    @article{IBRAHEEM2023232477,
        title = {Capacity and Internal Resistance of lithium-ion batteries: Full degradation curve prediction from Voltage response at constant Current at discharge},
        journal = {Journal of Power Sources},
        volume = {556},
        pages = {232477},
        year = {2023},
        issn = {0378-7753},
        doi = {https://doi.org/10.1016/j.jpowsour.2022.232477},
        url = {https://www.sciencedirect.com/science/article/pii/S0378775322014549},
        author = {Rasheed Ibraheem and Calum Strange and Gonçalo {dos Reis}},
        keywords = {Capacity degradation, Internal resistance degradation, Prediction of full degradation curve, Voltage response under constant current at discharge, Lithium-ion cells, Machine learning, Remaining useful life},
        abstract = {The use of minimal information from battery cycling data for various battery life prognostics is in high demand with many current solutions requiring full in-cycle data recording across 50–100 cycles. In this research, we propose a data-driven, feature-based machine learning model that predicts the entire capacity fade and internal resistance curves using only the voltage response from constant current discharge (fully ignoring the charge phase) over the first 50 cycles of battery use data. This approach is applicable where the discharging component is controlled and consistent, but sufficiently general to be applicable to settings with controlled charging but noisy discharge as is the case of electric vehicles. We provide a detailed analysis of the impact of the generated features on the model. We also investigate the impact of sub-sampling the voltage curve on the model performance where it was discovered that taking voltage measurements at every 1 minute is enough for model input without loss of quality. Example performance includes Capacity’s and Internal Resistance’s end of life being predicted with a mean absolute error (MAE) of 71 cycles and 1.5×10−5Ω respectively.}
    }

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
