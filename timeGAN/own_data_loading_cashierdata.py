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
    dataset_raw = pd.read_csv("/Josef/TimeGAN/data/CashierData.csv", sep=';', decimal=',')

    dataset_raw["Date"] = pd.to_datetime(dataset_raw["Date"], format='%Y-%m-%d')

    dataset_raw['quarter'] = dataset_raw['Date'].dt.quarter
    dataset_raw['weekday'] = dataset_raw['Date'].dt.weekday

    dataset_raw.name = "CashierData"

    return dataset_raw


def impute_dataset(dataset):
    """
    Imputes missing values
    :param dataset: Dataset that should get imputed
    :return: Imputed dataset
    """
    imputer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
    imputed_dataset = pd.DataFrame(imputer.fit_transform(dataset))
    imputed_dataset.columns = dataset.columns
    imputed_dataset.index = dataset.index

    return imputed_dataset


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
    
    # dates not important in case of this study
    dataset = dataset_raw.drop(columns=["Date"])


    imputed_dataset = impute_dataset(dataset)

    # test-train-split
    dataset, dataset_test = train_test_split(imputed_dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('/Josef/TimeGAN/test_data.csv', index=False)
        
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
       
    # delete columns that can be derived from others
    dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    

    columns = list(dataset_train.columns)
    
    # one-hot-encoding
    opt_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_dataset = opt_enc.fit_transform(dataset_train[["public_holiday", "school_holiday", "quarter", "weekday"]])
    num_columns_cat = np.shape(encoded_dataset)[1]
    encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset_train.index)
    dataset_train = pd.concat([dataset_train, encoded_dataset], axis=1).drop(["public_holiday", "school_holiday", "quarter", "weekday"],
                                                                             axis=1)
    joblib.dump(opt_enc, "opt_enc_" + database_name + ".gz")
    
    eval_enc = OneHotEncoder(sparse=False)
    encoded_dataset = eval_enc.fit_transform(dataset[["public_holiday", "school_holiday", "quarter", "weekday"]])
    encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset.index)
    dataset = pd.concat([dataset, encoded_dataset], axis=1).drop(["public_holiday", "school_holiday", "quarter", "weekday"], axis=1)
    joblib.dump(eval_enc, "eval_enc_" + database_name + ".gz")
    

    if shift_numbers != (0,):
        dataset_train, shift_numbers = shift_dataset(dataset_train, *shift_numbers)
    if shift_numbers != (0,):
        dataset, shift_numbers = shift_dataset(dataset, *shift_numbers)
        
    # convert to numpy array
    dataset_train = dataset_train.to_numpy()
    dataset = dataset.to_numpy()
    
    return dataset_train, columns, dataset_val, dataset, num_columns_cat, shift_numbers


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
