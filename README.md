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
1. Open a Terminal and navigate to the directory in which DGW.py can be found.
2. Run the optimization of DGW.

    ```
    python3 DGW.py -data CashierData.csv -n_trials 50 -database_name DGW_default -shift_numbers 0
    ```
    
    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.

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

#### Test the similarity of the fake data to the real data of DGW and visualize the optuna optimization
1. Just go to the DGW folder.
2. Open the `test_samples.ipynb` in a Jupyter Notebook.
3. Let it run.

    ```
    test_samples(data, database_name="default", print_data=True)
    ```

    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.

    **database_name** : *str*
    > Name of the project that you want to evaluate.

    **print_data** : *bool*
    > Weather to print the test and fake dataset or not.

    **Returns**

    *Plots and texts in sdtout*
    > Visual and statistical evaluation of the fake data compared to the real test data.

4. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
5. Return to step 3.
 
### CTGAN
1. Open a Terminal and navigate to the directory in which CTGAN.py can be found.
2. Run the optimization of CTGAN with.

    ```
    python3 CTGAN.py -data CashierData.csv -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```
    
    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.
    
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
3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
4. Return to step 2.

#### Test the similarity of the fake data to the real data of CTGAN and visualize the optuna optimization
1. Just go to the CTGAN folder.
2. Open the `test_samples.ipynb` in a Jupyter Notebook.
3. Let it run.

    ```
    test_samples(data, database_name="default", print_data=True)
    ```

    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.

    **database_name** : *str*
    > Name of the project that you want to evaluate.

    **print_data** : *bool*
    > Weather to print the test and fake dataset or not.

    **Returns**

    *Plots and texts in sdtout*
    > Visual and statistical evaluation of the fake data compared to the real test data.

4. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
5. Return to step 3. 
##### Show the optuna optimization plots
1. Insert the right study name and storage.
2. Run the cell.

### timeGAN
1. Open a Terminal and navigate to the directory in which `timeGAN.py` can be found.
2. Run the optimization of timeGAN with

    ```
    python3 timeGAN.py -data CashierData.csv -num_samples 500 -n_trials 100 -database_name CTGAN_default -shift_numbers 0
    ```

    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.
    
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

3. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
4. Return to step 2. 

#### Test the similarity of the fake data to the real data of timeGAN and visualize the optuna optimization
1. Just go to the timeGAN folder.
2. Open the `test_samples.ipynb` in a Jupyter Notebook.
3. Let it run.

    ```
    test_samples(data, database_name="default", print_data=True)
    ```

    **Arguments**
    
    **data** : *str*
    > Dataset that should be used.

    **database_name** : *str*
    > Name of the project that you want to evaluate.

    **print_data** : *bool*
    > Weather to print the test and fake dataset or not.

    **Returns**

    *Plots and texts in sdtout*
    > Visual and statistical evaluation of the fake data compared to the real test data.

4. If `ModuleNotFoundError: No module named 'package'` occurs, install these missing packages with `pip3 install package`.
5. Return to step 3. 

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
