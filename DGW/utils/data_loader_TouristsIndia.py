import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib


def load_datasets():
    """
    Loads the dataset; removes dates
    :return: raw dataset
    """
    dataset_raw = pd.read_csv("Data/Public_QuarterlyTouristsIndia.csv", sep=';', decimal=',')

    dataset_raw = dataset_raw.drop("index", axis=1)
    
    dataset_raw.name = "QuarterlyTouristsIndia"

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
        shifted_dataset.columns = shifted_dataset.columns.astype(str) + "_shifted" + str(shift_number)
        preprocessed_dataset = pd.concat([preprocessed_dataset, shifted_dataset], axis=1)
        preprocessed_dataset = preprocessed_dataset.iloc[:preprocessed_dataset.shape[0]+shift_number, :]
        preprocessed_dataset = preprocessed_dataset.reset_index(drop=True)
        to_shift_dataset = to_shift_dataset.drop(to_shift_dataset.tail(abs(shift_number)).index)
        
    return preprocessed_dataset


def real_data_loading(database_name, *shift_numbers):
    """
    Loads and preprocessed the dataset
    :param database_name: name that the project should have
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: training data, training labels, number of categorical columns, validation data, data to train the optimized model, 
    labels to train the optimized model, column names, number of days for the shifted dataset 
    """
    dataset = load_datasets()    
    
    # test-train-split
    dataset, dataset_test, = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('test_data_QuarterlyTouristsIndia.csv', index=False)
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=False)
    
    # save column names to later reconstruct a dataframe
    columns = list(dataset_train.columns)

    
    if shift_numbers != (0,):
        dataset_train = shift_dataset(dataset_train, *shift_numbers)
    if shift_numbers != (0,):
        dataset = shift_dataset(dataset, *shift_numbers)

    # scaling
    opt_scaler = MinMaxScaler()
    dataset_train[dataset_train.columns] = opt_scaler.fit_transform(dataset_train[dataset_train.columns])
    joblib.dump(opt_scaler, "opt_scaler_" + database_name + ".gz")

    eval_scaler = MinMaxScaler()
    dataset[dataset.columns] = eval_scaler.fit_transform(dataset[dataset.columns])
    joblib.dump(eval_scaler, "eval_scaler_" + database_name + ".gz")    
             
    # convert to numpy array
    ori_data = dataset_train.to_numpy()
    dataset = dataset.to_numpy()
        
    # devide to labels and data
    labels_train = ori_data[:,0]
    dataset_train = ori_data[:,0:]
    labels = dataset[:,0]
    dataset = dataset[:,0:]
        
    return dataset_train, labels_train, dataset_val, dataset, labels, columns, shift_numbers


def postprocess_data(eval_or_opt, generated_data, columns, database_name, *shift_numbers):
    """
    Postprocesses synthesized data
    :param eval_or_opt: "opt" for the optimization loop, "eval" for training the optimized model
    :param generated_data: synthesized data from the model
    :param columns: Names of the columns of the dataset
    :param num_columns_cat: number of categorical columns
    :param database_name: name of the project
    :shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: postprocessed fake data
    """
    # reshape to 2-dimensional array
    generated_data = generated_data.reshape(generated_data.shape[0], generated_data.shape[1])
    
    # rescale
    scaler = joblib.load(eval_or_opt + "_scaler_" + str(database_name) + ".gz")
    generated_data = scaler.inverse_transform(generated_data)
    
    # delete shifted data
    if shift_numbers != (0,):
        generated_data = generated_data[:,:len(shift_numbers) * -42]
                
    # transform to a pandas dataframe
    generated_data_df = pd.DataFrame(data=generated_data, columns=columns)


    fake_data = generated_data_df
        
    return fake_data
