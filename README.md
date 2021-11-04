# Master-Thesis
All codes used for my master thesis
## Setup and Operation
1. Open a Terminal and navigate to the directory where the algorithms should be
2. Clone this repository (as soon as it's public)

    ```
    git clone https://github.com/Zepp3/Master-Thesis
    ```
    
### Discriminative Guided Warping (DGW)
DGW runs on two different ways for the two datasets. CashierData in a Jupyter Notebook, and Schachtschneider via command line, as the computation duration is too long to be executed in an open notebook.
#### CashierData
Just go to the DGW folder and let the DGW.ipynb run in a Jupyter Notebook

```
run_DTW(n_trials=50, database_name="default", *shift_numbers)
```

**Arguments**

**n_trials** : *int*
> Number of trials the optimization should run.

**database_name** : *str*
> Name that the project should have.

**shift_numbers** : *int*
> Number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative.

**Returns**

*Dataframe and .txt-file*
> Fake data in a Dataframe and performance measures.

#### Schachtschneider
1. Open a Terminal and navigate to the directory in which DGW_schachtschneider.py can be found
2. Run the optimization of DGW with the Schachtschneider dataset

    ```
    python3 DGW_schachtschneider -n_trials 50 -database_name DGW_default -shift_numbers 0
    ```
    
    **Arguments**
    
    **n_trials** : *int*
    > Number of optimization trials.
    
    **database_name** : *str*
    > Name of the project.

    **shift_numbers** : *int*
    > Number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative.
    
    **Returns**
    
    *Dataframe and .txt-file*
    > Fake data in a Dataframe and performance measures.
    
## Citation
**DGW**: Iwana, Brian Kenji; Uchida, Seiichi (2020): Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. Available online at http://arxiv.org/pdf/2004.08780v1.
