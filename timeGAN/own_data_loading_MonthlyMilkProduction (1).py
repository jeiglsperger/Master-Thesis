import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib


def load_datasets():
    """
    Loads the dataset; adds date, quarter and weekday
    :return: raw dataset
    """
    dataset_raw = pd.read_csv("data/Public_MonthlyMilkProduction.csv", sep=';', decimal=',')
    
    dataset_raw = dataset_raw.drop("Date", axis=1)
    
    dataset_raw.name = "MonthlyMilkProduction"

    return dataset_raw


def shift_dataset(preprocessed_dataset, *shift_numbers):
    """
    Adds shifted dataset
    :param preprocessed_dataset: Dataset to which the shifted dataset should be added
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: Dataset with the shifted dataset(s) added
    """
    to_shift_dataset = preprocessed_dataset
    
    for shift_number in shift_numbers:
        shifted_dataset = to_shift_dataset.shift(periods=shift_number)
        print("Shifted Dataset")
        print(np.shape(shifted_dataset))
        shifted_dataset.columns = shifted_dataset.columns.astype(str) + "_shifted" + str(shift_number)
        preprocessed_dataset = pd.concat([preprocessed_dataset, shifted_dataset], axis=1)
        preprocessed_dataset = preprocessed_dataset.iloc[:preprocessed_dataset.shape[0]+shift_number, :]
        preprocessed_dataset = preprocessed_dataset.reset_index(drop=True)
        to_shift_dataset = to_shift_dataset.drop(to_shift_dataset.tail(abs(shift_number)).index)

    return preprocessed_dataset, shift_numbers


def real_data_loading():
    """
    Loads and preprocessed the dataset
    :return: dataset
    """
    dataset_raw = load_datasets()

    # test-train-split
    dataset, dataset_test = train_test_split(dataset_raw, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('test_data_MonthlyMilkProduction.csv', index=False)
        
    return dataset


def preprocess_data(dataset, database_name, *shift_numbers):
    """
    Preprocesses the dataset
    :param dataset: dataset that should get preprocessed
    :param database_name: name that the project should have
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: training data, column names, validation data, data to train the optimized model, number of categorical columns, 
    number of days for the shifted dataset 
    """       
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=False)
    

    columns = list(dataset_train.columns)
    

    if shift_numbers != (0,):
        dataset_train, shift_numbers = shift_dataset(dataset_train, *shift_numbers)
    if shift_numbers != (0,):
        dataset, shift_numbers = shift_dataset(dataset, *shift_numbers)
        
    # convert to numpy array
    dataset_train = dataset_train.to_numpy()
    dataset = dataset.to_numpy()
    
    return dataset_train, columns, dataset_val, dataset, shift_numbers


def cut_data(ori_data, seq_len):
    """
    cut and mix datato make it similar to i.i.d
    :param ori_data: data that should be cutted and mixed
    :param seq_len: sequence length
    :return: cutted and mixed data
    """
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
        
    # Mix the datasets
    idx = np.random.permutation(len(temp_data))    
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
        
    return data
