# cc-voltage-project
This repository contains the codes for all the experiments performed in the paper [R. Ibraheem, C. Strange, G. dos Reis, Capacity and internal resistance of lithium-ion batteries: Full degradation curve prediction from voltage response at constant current at discharge, Journal of Power Sources 556 (2023)232477.](https://www.sciencedirect.com/science/article/pii/S0378775322014549)

![Paper abstract](assets/paper_abstract.png)

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
        author = {Rasheed Ibraheem and Calum Strange and Gonçalo {dos Reis}},
        journal = {Journal of Power Sources},
        volume = {556},
        pages = {232477},
        year = {2023},
        issn = {0378-7753},
        doi = {https://doi.org/10.1016/j.jpowsour.2022.232477},
        url = {https://www.sciencedirect.com/science/article/pii/S0378775322014549},
    }

_Licence: [CC BY 4.0.](https://creativecommons.org/licenses/by/4.0/legalcode)_
