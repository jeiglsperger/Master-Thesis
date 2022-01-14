import numpy as np
import os
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt
from sdv.evaluation import evaluate
import optuna
import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse


def load_datasets(data):
    """
    Loads the dataset; adds date, quarter and weekday and then removes dates
    :param data: define which dataset should be used
    :return: raw dataset
    """
    if data == "Schachtschneider_externals_cut.csv":
        sep=','
        decimal = '.'
    else:
        sep=';'
        decimal=','
    dataset_raw = pd.read_csv("data/" + data, sep=sep, decimal=decimal)

    if data == "CashierData.csv":
        dataset_raw["Date"] = pd.to_datetime(dataset_raw["Date"], format='%Y-%m-%d')
        dataset_raw['quarter'] = dataset_raw['Date'].dt.quarter
        dataset_raw['weekday'] = dataset_raw['Date'].dt.weekday
        dataset_raw = dataset_raw.drop(columns=["Date"])
    elif data == "Schachtschneider_externals_cut.csv":
        dataset_raw["Auf. Datum"] = pd.to_datetime(dataset_raw["Auf. Datum"], format='%Y-%m-%d')
        dataset_raw = dataset_raw.drop(columns=["school_holiday"])
        dataset_raw = dataset_raw.drop(columns=["Auf. Datum"])
    elif data == "Public_MonthlyMilkProduction.csv":
        dataset_raw = dataset_raw.drop("Date", axis=1)
    elif data == "Public_QuarterlyTouristsIndia.csv":
        dataset_raw = dataset_raw.drop("index", axis=1)
    
    dataset_raw.name = data[:-4]

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
        
    return preprocessed_dataset


def real_data_loading(database_name, *shift_numbers):
    """
    Loads and preprocessed the dataset
    :param database_name: name that the project should have
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: training data, training labels, number of categorical columns, validation data, data to train the optimized model, 
    labels to train the optimized model, column names, number of days for the shifted dataset 
    """
    dataset_raw = load_datasets(data)
    
    
    dataset = impute_dataset(dataset_raw)
    
    # test-train-split
    dataset, dataset_test, = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('test_data_' + data, index=False)
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=False)
    
    # delete columns that can be derived from others
    if data == "CashierData.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    elif data == "Schachtschneider_externals_cut.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])
    
    # save column names to later reconstruct a dataframe
    columns = list(dataset_train.columns)

    # one-hot-encoding
    if data == "CashierData.csv":
        opt_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_dataset = opt_enc.fit_transform(dataset_train[["public_holiday", "school_holiday", "quarter", "weekday"]])
        num_columns_cat = np.shape(encoded_dataset)[1]
        encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset_train.index)
        dataset_train = pd.concat([dataset_train, encoded_dataset], axis=1).drop(["public_holiday", "school_holiday", "quarter", "weekday"], 
                                                                                 axis=1)
        joblib.dump(opt_enc, "opt_enc_" + database_name + ".gz")

        eval_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_dataset = eval_enc.fit_transform(dataset[["public_holiday", "school_holiday", "quarter", "weekday"]])
        encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset.index)
        dataset = pd.concat([dataset, encoded_dataset], axis=1).drop(["public_holiday", "school_holiday", "quarter", "weekday"], axis=1)
        joblib.dump(eval_enc, "eval_enc_" + database_name + ".gz")
    elif data == "Schachtschneider_externals_cut.csv":
        opt_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_dataset = opt_enc.fit_transform(dataset_train[["public_holiday", "weekday"]])
        num_columns_cat = np.shape(encoded_dataset)[1]
        encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset_train.index)
        dataset_train = pd.concat([dataset_train, encoded_dataset], axis=1).drop(["public_holiday", "weekday"], axis=1)
        joblib.dump(opt_enc, "opt_enc_" + database_name + ".gz")

        eval_enc = OneHotEncoder(sparse=False)
        encoded_dataset = eval_enc.fit_transform(dataset[["public_holiday", "weekday"]])
        num_columns_cat_eval = np.shape(encoded_dataset)[1]
        encoded_dataset = pd.DataFrame(encoded_dataset, index=dataset.index)
        dataset = pd.concat([dataset, encoded_dataset], axis=1).drop(["public_holiday", "weekday"], axis=1)
        joblib.dump(eval_enc, "eval_enc_" + database_name + ".gz")
    else:
        num_columns_cat = 0

    
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
    if data == "CashierData.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:len(shift_numbers) * -44]
    if data == "Schachtschneider_externals_cut.csv":
        if shift_numbers != (0,):
            if eval_or_opt == "opt":
                generated_data = generated_data[:,:len(shift_numbers) * -804]
            else:
                generated_data = generated_data[:,:len(shift_numbers) * -805]
    if data == "Public_MonthlyMilkProduction.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:len(shift_numbers) * -1]
    if data == "Public_QuarterlyTouristsIndia.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:len(shift_numbers) * -42]
    

    # reverse one-hot-encoding
    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        enc = joblib.load(eval_or_opt + "_enc_" + database_name + ".gz")
        to_inverse_transform_data = generated_data[:,np.shape(generated_data)[1]-num_columns_cat:np.shape(generated_data)[1]]
        inverse_transformed_data = []
        for i in range(np.shape(generated_data)[0]):
            generated_data_no_cat = generated_data[i,0:np.shape(generated_data)[1]-num_columns_cat]
            inverse_transformed_data.append(np.concatenate(([generated_data_no_cat], enc.inverse_transform([to_inverse_transform_data[i]])), axis=None))      
        generated_data = np.array(inverse_transformed_data)
                
    # transform to a pandas dataframe
    generated_data_df = pd.DataFrame(data=generated_data, columns=columns)
    
    # insert columns again that can be derived from others
    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        total_sun_dur_h_fake = (generated_data_df["mean_sun_dur_min"] / 60) * 24
        generated_data_df.insert(11, 'total_sun_dur_h', total_sun_dur_h_fake)
        total_prec_height_mm_fake = generated_data_df["mean_prec_height_mm"] * 24
        generated_data_df.insert(9, 'total_prec_height_mm', total_prec_height_mm_fake)
        
    if data == "CashierData.csv":
        total_prec_flag_fake = generated_data_df['mean_prec_flag'].apply(lambda x: 'True' if x > 0 else 'False')
        generated_data_df.insert(11, 'total_prec_flag', total_prec_flag_fake)

    fake_data = generated_data_df
        
    return fake_data


def prepare_train_data(database_name, *shift_numbers):
    """
    Prepares the training data
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: training data, training labels, number of categorical columns, validation data, data to train the optimized model, 
    labels to trian the optimized model, column names, number of days for the shifted dataset
    """
    x_train, y_train, num_columns_cat, dataset_val, dataset, labels, columns, shift_numbers = real_data_loading(database_name, *shift_numbers)

    # reshape as DGW only accepts 3-dimensional data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))
    
    return x_train, y_train, num_columns_cat, dataset_val, dataset, labels, columns, shift_numbers


def objective(trial, x_train, y_train, columns, num_columns_cat, dataset_val, database_name, *shift_numbers):
    """
    Objective function for hyperparameter optimization with optuna
    :param trail: current optimization trial
    :param x_train: training data
    :param y_train: training labels
    :param columns: column names
    :param num_columns_cat: number of categorical columns
    :param dataset_val: validation data
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: score of the optimization
    """
    # hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6])
    slope_constraint = trial.suggest_categorical("slope_constraint", ["symmetric", "asymmetric"])
    use_window = trial.suggest_categorical("use_window", ["True", "False"])
    dtw_type = trial.suggest_categorical("dtw_type", ["normal", "shape"])
    use_variable_slice = trial.suggest_categorical("use_variable_slice", ["True", "False"])
    
    # generate data
    generated_data = aug.discriminative_guided_warp(x_train, y_train, batch_size=batch_size, slope_constraint=slope_constraint, use_window=use_window, dtw_type=dtw_type, use_variable_slice=use_variable_slice)

    # postprocessing
    fake_data = postprocess_data("opt", generated_data, columns, num_columns_cat, database_name, *shift_numbers)

    # calculate score 
    scores = evaluate(fake_data, dataset_val)

    return scores


def run_DTW(data, n_trials=50, database_name="default", *shift_numbers):
    """
    Run DTW
    :param n_trials: number of optimization trials
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    """
    # load and prepare data
    x_train, y_train, num_columns_cat, dataset_val, dataset, labels, columns, shift_numbers = prepare_train_data(database_name, *shift_numbers)
    
    # optimize hyperparameters
    study = optuna.create_study(storage=optuna.storages.RDBStorage("sqlite:///" + database_name + ".db"), 
                                study_name = database_name + "_study", direction="maximize", load_if_exists=True
    study.optimize(lambda trial: objective(trial, x_train, y_train, columns, num_columns_cat, dataset_val, database_name, *shift_numbers),
                   n_trials)

    # save performance parameters
    performance = open("performance_" + database_name + ".txt","w+")
    best_parameters = str(study.best_params)
    performance.write(best_parameters)
    best_values = str(study.best_value)
    performance.write(best_values)
    best_trials = str(study.best_trial)
    performance.write(best_trials)

    
    # generate data and stop time for this task
    start_time = time.time()
    generated_data = aug.discriminative_guided_warp(dataset, labels, **study.best_params)
    performance.write(str("--- %s minutes ---" % ((time.time() - start_time) / 60)))
    performance.close()
    # plot of the result of the generation
    hlp.plot1d(dataset[0], generated_data[0])

    # postprocessing
    fake_data = postprocess_data("eval", generated_data, columns, num_columns_cat, database_name, *shift_numbers)
    fake_data.to_csv('fake_data_' + database_name + '.csv', index=False)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-data", "--data", type=str, default="CashierData.csv", help="define data")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=50, help="specify number of optuna trials")
    parser.add_argument("-database_name", "--database_name", type=str, default='DGW_default', help="specify the database")
    parser.add_argument("-shift_numbers", "--shift_numbers", nargs='*', type=int, default=(0,), help="specify shifts of the data")
    
    args = parser.parse_args()

    data = args.data
    n_trials = args.n_trials
    database_name = args.database_name
    shift_numbers = tuple(args.shift_numbers)

    run_DTW(data, n_trials, database_name, *shift_numbers)