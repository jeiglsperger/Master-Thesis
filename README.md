# Master-Thesis
All codes used for my master thesis.
## Setup and Operation
1. Open a Terminal and navigate to the directory where the algorithms should be.
2. Clone this repository.

    ```
    git clone https://github.com/Zepp3/Master-Thesis
    ```

### DataIngestSchachtschneider
1. It is possible to download and locally run `DataIngest_Schachtschneider.py` in PyCharm for example.
2. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
3. Return to step 1.
4. Open a Terminal and navigate to the directory in which add_externals.py can be found.
5. Add external data with

    ```
    python3 add_externals.py
    ```
    
6. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
7. Return to step 5.

### Discriminative Guided Warping (DGW)
DGW runs on two different ways for the two datasets. CashierData in a Jupyter Notebook, and Schachtschneider via command line, as the computation duration is too long to be executed in an open notebook.
#### CashierData, MonthlyMilkProduction
1. Just go to the DGW folder
2. Open `DGW.ipynb` in a Jupyter Notebook
3. Import either `from DGW_cashierdata` or `from DGW_MonthlyMilkProduction`
4. Let it run.

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
5. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
6. Return to step 1.

#### Schachtschneider
1. Open a Terminal and navigate to the directory in which DGW_schachtschneider.py can be found.
2. Run the optimization of DGW with the Schachtschneider dataset.

    ```
    python3 DGW_schachtschneider.py -n_trials 50 -database_name DGW_default -shift_numbers 0
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
3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
4. Return to step 2. 

#### Test the similarity of the fake data to the real data of DGW
1. Just go to the DGW folder.
2. Open the `test_samples.ipynb` or in a Jupyter Notebook.
3. Import either `from evaluate_cashierdata` or `from evaluate_MonthlyMilkProduction` or `from evaluate_schachtschneider`
4. Choose the right test data file in `test, fake = load_data('test_data_XXX.csv', 'fake_data_' + database_name + '.csv')`
5. Let it run.

    ```
    test_samples(database_name="default", print_data=True)
    ```

    **Arguments**

    **database_name** : *str*
    > Name of the project that you want to evaluate.

    **print_data** : *bool*
    > Weather to print the test and fake dataset or not.

    **Returns**

    *Plots and texts in sdtout*
    > Visual and statistical evaluation of the fake data compared to the real test data.

5. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
6. Return to step 4. 
### CTGAN
1. Open a Terminal and navigate to the directory in which CTGAN_cashierdata.py and CTGAN_schachtschneider.py can be found
2. Run the optimization of CTGAN with

    ```
    python3 CTGAN_cashierdata.py -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
  
    or 
    
    ```
    python3 CTGAN_schachtschneider.py  -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    or 
    
    ```
    python3 CTGAN_MonthlyMilkProduction.py  -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
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
3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`
4. Return to step 2.
5. To test the similarity of the fake data to the real data just run the `test_samples_cashierdata.ipynb` or `test_samples_schachtschneider.ipynb` in a Jupyter Notebook.

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

6. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
7. Return to step 5.  
### timeGAN
1. Open a Terminal and navigate to the directory in which `timeGAN_opt_seq_len_cashierdata.py`, `timeGAN_year_cashierdata.py`, `timeGAN_opt_seq_len_schachtschneider.py` and `timeGAN_year_schachtschneider.py` can be found
2. Run the optimization of timeGAN with

    ```
    python3 timeGAN_opt_seq_len_cashierdata.py -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    or

    ```
    python3 timeGAN_opt_seq_len_schachtschneider.py -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
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

    ```
    python3  timeGAN_year_cashierdata.py -seq_len 500  -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    or 
    
    ```
    python3  timeGAN_year_schachtschneider.py -seq_len 500  -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    **Arguments**
    
    **seq_len** : *int*
    > Sequence length.
    
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
3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`
4. Return to step 2. 
5. To test the similarity of the fake data to the real data just run the `test_samples_cashierdata.ipynb` or `test_samples_schachtschneider.ipynb` in a Jupyter Notebook.

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

6. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
7. Return to step 5. 
## Contribution
All codes in this repository were written by myself with exeption of the following scripts which were written by the authors of corresponing paper:
### DataIngestSchachtschneider
The scripts are based on scripts of [Florian Haslbeck](https://bit.cs.tum.de/team/florian-haselbeck/).
### DGW
`augmentation.py`, `dtw.py`, `helper.py`
### timeGAN
`timegan.py`, `cut_data(ori_data, seq_len)` in `own_data_loading_cashierdata` and `own_data_loading_schachtschneider`, `utils.py`
## Citation
**CTGAN**: Xu, Lei; Skoularidou, Maria; Cuesta-Infante, Alfredo; Veeramachaneni, Kalyan (2019): Modeling Tabular data using Conditional GAN. In H. Wallach, H. Larochelle, A. Beygelzimer, F. Alché-Buc, E. Fox, R. Garnett (Eds.): Advances in Neural Information Processing Systems, vol. 32: Curran Associates, Inc. Available online at https://proceedings.neurips.cc/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Paper.pdf.

**DGW**: Iwana, Brian Kenji; Uchida, Seiichi (2020): Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. Available online at http://arxiv.org/pdf/2004.08780v1.

**timeGAN**
Yoon, Jinsung; Jarrett, Daniel; van der Schaar, Mihaela (2019): Time-series Generative Adversarial Networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. Alché-Buc, E. Fox, R. Garnett (Eds.): Advances in Neural Information Processing Systems, vol. 32: Curran Associates, Inc. Available online at https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf.
