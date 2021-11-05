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
Just go to the DGW folder and let the `DGW.ipynb` run in a Jupyter Notebook

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

To test the similarity of the fake data to the real data just run the `test_samples_cashierdata.ipynb` in a Jupyter Notebook

```
test_samples(database_name="default", print_data=True)
```

**Arguments**

**database_name** : *str*
> Name that the project should have.

**print_data** : *bool*
> Weather to print the test and fake dataset or not.

**Returns**

*Plots and texts in sdtout*
> Visual and statistical evaluation of the fake data compared to the real test data.

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
3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`
4. Return to step 2. 

### CTGAN

1. Open a Terminal and navigate to the directory in which CTGAN_cashierdata and CTGAN_schachtschneider.py can be found
2. Run the optimization of CTGAN with

    ```
    python3 CTGAN_schachtschneider -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
  
    or 
    
    ```
    python3 CTGAN_schachtschneider  -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    **Arguments**
    
    **num_samples** : *int*
    > Number of fake samples that should be generated.
    
    **n_trials** : *int*
    > Number of optimization trials.
    
    **database_name** : *str*
    > Name of the project.

    **shift_numbers** : *int*
    > Number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative.
    
    **Returns**
    
    *Dataframe and .txt-file*
    > Fake data in a Dataframe and performance measures.

## Contribution

All codes in this repository were written by myself with exeption of the following scripts which were written by the authors of corresponing paper:

### DGW

`augmentation.py`
`helper.py`

## Citation
**DGW**: Iwana, Brian Kenji; Uchida, Seiichi (2020): Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. Available online at http://arxiv.org/pdf/2004.08780v1.

**CTGAN**: Xu, Lei; Skoularidou, Maria; Cuesta-Infante, Alfredo; Veeramachaneni, Kalyan (2019): Modeling Tabular data using Conditional GAN. In H. Wallach, H. Larochelle, A. Beygelzimer, F. Alché-Buc, E. Fox, R. Garnett (Eds.): Advances in Neural Information Processing Systems, vol. 32: Curran Associates, Inc. Available online at https://proceedings.neurips.cc/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf.
