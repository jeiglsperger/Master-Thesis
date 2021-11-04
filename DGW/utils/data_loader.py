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
    Loads the dataset; adds date, quarter and weekday and then removes dates
    :return: raw dataset
    """
    dataset_raw = pd.read_csv("/JOsef/DTW/time_series_augmentation/Data/CashierData.csv", sep=';', decimal=',')

    dataset_raw["Date"] = pd.to_datetime(dataset_raw["Date"], format='%Y-%m-%d')
    dataset_raw['quarter'] = dataset_raw['Date'].dt.quarter
    dataset_raw['weekday'] = dataset_raw['Date'].dt.weekday
    
    dataset_raw.name = "CashierData"

    # dates not important in case of this study
    dataset_raw = dataset_raw.drop(columns=["Date"])

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
    # shift dataset
    to_shift_dataset = preprocessed_dataset
    print(to_shift_dataset.shape[1])
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
    dataset_raw = load_datasets()
    
    
    dataset = impute_dataset(dataset_raw)
    
    # test-train-split
    dataset, dataset_test, = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('/JOsef/DTW/time_series_augmentation/test_data.csv', index=False)
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=False)
    
    # delete columns that can be derived from others
    dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    
    # save column names to later reconstruct a dataframe
    columns = list(dataset_train.columns)

    # one-hot-encoding
    opt_enc = OneHotEncoder(sparse=False)
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
        
    return dataset_train, labels_train, num_columns_cat, dataset_val, dataset, labels, columns, shift_numbers


def postprocess_data(eval_or_opt, generated_data, columns, num_columns_cat, database_name, *shift_numbers):
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
        generated_data = generated_data[:,:len(shift_numbers) * -44]

    # reverse one-hot-encoding
    enc = joblib.load(eval_or_opt + "_enc_" + database_name + ".gz")
    to_inverse_transform_data = generated_data[:,np.shape(generated_data)[1]-num_columns_cat:np.shape(generated_data)[1]]
    inverse_transformed_data = []
    for i in range(np.shape(generated_data)[0]):
        generated_data_no_cat = generated_data[i,0:np.shape(generated_data)[1]-num_columns_cat]
        inverse_transformed_data.append(np.concatenate(([generated_data_no_cat], enc.inverse_transform([to_inverse_transform_data[i]])), 
                                                       axis=None))      
    inverse_transformed_data = np.array(inverse_transformed_data)
                
    # transform to a pandas dataframe
    generated_data_df = pd.DataFrame(data=inverse_transformed_data, columns=columns)
    
    # insert columns again that can be derived from others
    total_sun_dur_h = (generated_data_df["mean_sun_dur_min"] / 60) * 24
    generated_data_df.insert(11, 'total_sun_dur_h', total_sun_dur_h)
    total_prec_height_mm = generated_data_df["mean_prec_height_mm"] * 24
    generated_data_df.insert(9, 'total_prec_height_mm', total_prec_height_mm)
    total_prec_flag = generated_data_df['mean_prec_flag'].apply(lambda x: 'True' if x > 0 else 'False')
    generated_data_df.insert(11, 'total_prec_flag', total_prec_flag)

    fake_data = generated_data_df
        
    return fake_data
