from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time
import optuna
from timegan import timegan
from sdv.evaluation import evaluate
from table_evaluator import load_data, TableEvaluator
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_datasets(data):
    """
    Loads the dataset; adds date, quarter and weekday
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

    return preprocessed_dataset, shift_numbers


def real_data_loading(data):
    """
    Loads and preprocessed the dataset
    :return: dataset
    """
    dataset_raw = load_datasets(data)


    imputed_dataset = impute_dataset(dataset_raw)

    # test-train-split
    dataset, dataset_test = train_test_split(imputed_dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_test.to_csv('test_data_' + data, index=False)
        
    return dataset


def preprocess_data(data, dataset, database_name, *shift_numbers):
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
    if data == "CashierData.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    elif data == "Schachtschneider_externals_cut.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])
    

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


def postprocess_data(data, eval_or_opt, generated_data, columns, num_columns_cat, seq_len, *shift_numbers):
    """
    Postprocesses synthesized data
    :param eval_or_opt: "opt" for the optimization loop, "eval" for training the optimized model
    :param generated_data: synthesized data from the model
    :param columns: Names of the columns of the dataset
    :param num_columns_cat: number of categorical columns
    :param seq_len: sequence length
    :shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: postprocessed fake data
    """ 
    # delete shifted data
    if data == "CashierData.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:,:len(shift_numbers) * -44]
    if data == "Schachtschneider_externals_cut.csv":
        if shift_numbers != (0,):
            if eval_or_opt == "opt":
                generated_data = generated_data[:,:,:len(shift_numbers) * -804]
            else:
                generated_data = generated_data[:,:,:len(shift_numbers) * -805]
    if data == "Public_MonthlyMilkProduction.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:,:len(shift_numbers) * -1]
    if data == "Public_QuarterlyTouristsIndia.csv":
        if shift_numbers != (0,):
            generated_data = generated_data[:,:,:len(shift_numbers) * -42]
        
    # reverse one-hot-encoding    
    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        enc = joblib.load(eval_or_opt + "_enc_" + database_name + ".gz")
        if eval_or_opt == "opt":
            to_inverse_transform_data = generated_data[:,:,np.shape(generated_data)[2]-num_columns_cat:np.shape(generated_data)[2]]
        else:
            if data == "CashierData.csv":
                to_inverse_transform_data = generated_data[:,:,np.shape(generated_data)[2]-num_columns_cat:np.shape(generated_data)[2]] 
            elif data == "Schachtschneider_externals_cut.csv":
                to_inverse_transform_data = generated_data[:,:,np.shape(generated_data)[2]-(num_columns_cat+1):np.shape(generated_data)[2]]
        inverse_transformed_data = []
        for i in range(np.shape(generated_data)[0]):
            if data == "CashierData.csv":
                generated_data_no_cat = generated_data[:,:,0:np.shape(generated_data)[2]-num_columns_cat]
                inverse_transformed_data.append(np.concatenate((generated_data_no_cat[i],
                                                            enc.inverse_transform(to_inverse_transform_data[i],)),
                                                               axis=1))
            elif data == "Schachtschneider_externals_cut.csv":
                if eval_or_opt == "opt":
                    generated_data_no_cat = generated_data[:,:,0:np.shape(generated_data)[2]-num_columns_cat]
                else:
                    generated_data_no_cat = generated_data[:,:,0:np.shape(generated_data)[2]-(num_columns_cat+1)]
                inverse_transformed_data.append(np.concatenate((generated_data_no_cat[i], 
                                                             enc.inverse_transform(to_inverse_transform_data[i],))
                                                               , axis=1))
        generated_data = np.array(inverse_transformed_data)


    conditioned_dataset = []
    for i in range(np.shape(generated_data)[0]):
        for j in range(np.shape(generated_data)[1]):
            conditioned_dataset.append(generated_data[i,j,:])
    
    # transform to a pandas dataframe
    generated_data_df = pd.DataFrame(data=conditioned_dataset, columns=columns)
    
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


best_scores = 0

def objective(trial, dataset_train, dataset_val, num_samples, columns, num_columns_cat, database_name, *shift_numbers):
    """
    Objective function for hyperparameter optimization with optuna
    :param trail: current optimization trial
    :param dataset_train: training data
    :param dataset_val: validation data
    :param num_samples: number of fake samples that should be generated
    :param columns: column names
    :param num_columns_cat: number of categorical columns   
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: score of the optimization
    """    
    if trial.number == 60:
        trial.study.stop()
    
    # hyperparameters
    module = trial.suggest_categorical("module", ["gru", "lstm", "lstmLN"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [6, 12, 24])
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    num_layer = trial.suggest_int("num_layer", 3, 6, 1)
    iterations = trial.suggest_categorical("iterations", [1, 10, 100])
    seq_len = trial.suggest_categorical("seq_len", [5, 10, 20])
    

    data_cut = cut_data(dataset_train, seq_len)
        
    
    # set timeGAN parameters
    parameters = dict()
    parameters['module'] = module
    parameters['hidden_dim'] = hidden_dim
    parameters['num_layer'] = num_layer
    parameters['iterations'] = iterations
    parameters['batch_size'] = batch_size
    parameters['seq_len'] = seq_len
    
    # generate data
    generated_data = timegan(data_cut, parameters)

    # postprocessing
    fake_data = postprocess_data(data, "opt", generated_data, columns, num_columns_cat, seq_len, *shift_numbers)

    # calculate score
    scores = evaluate(fake_data, dataset_val)
    
    # save best model
    global best_scores
    if scores > best_scores:
        joblib.dump(timegan, database_name + '.gz')
        objective.seq_len = seq_len
        objective.parameters = parameters
        best_scores = scores

    return scores


def run_TimeGAN(data, num_samples, n_trials, database_name, *shift_numbers):
    """
    Run timeGAN
    :param data: define which dataset should be used
    :param num_samples: number of fake samples that should be generated
    :param n_trials: number of optimization trials
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    """
    # load data
    dataset = real_data_loading(data)

    # preprocess data
    dataset_train, columns, dataset_val, dataset, num_columns_cat, shift_numbers = preprocess_data(data, dataset, database_name, *shift_numbers)
    
    # optimize hyperparameters
    study = optuna.create_study(storage=optuna.storages.RDBStorage("sqlite:///" + database_name + ".db"), 
                                study_name = database_name + "_study", direction="maximize", load_if_exists=True)  #  use GP
    study.optimize(lambda trial: objective(trial,dataset_train, dataset_val, num_samples, columns, num_columns_cat, database_name, 
                                           *shift_numbers), n_trials)
    
    # save performance parameters
    performance = open("performance_" + database_name + ".txt","w+")
    best_parameters = str(study.best_params)
    performance.write(best_parameters)
    best_values = str(study.best_value)
    performance.write(best_values)
    best_trials = str(study.best_trial)
    performance.write(best_trials)
   

    # generate data and stop time for this task
    timegan = joblib.load(database_name + '.gz')
    data_cut = cut_data(dataset, objective.seq_len)
    start_time = time.time()
    generated_data = timegan(data_cut, objective.parameters)
    performance.write(str("--- %s minutes ---" % ((time.time() - start_time) / 60)))
    performance.close()
    
    # postprocessing
    fake_data = postprocess_data(data, "eval", generated_data, columns, num_columns_cat, objective.seq_len, *shift_numbers)

    fake_data.to_csv('fake_data_' + database_name + '.csv', index=False)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", "--data", type=str, default="CashierData.csv", help="define data")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=1000, help="specify number of samples")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=100, help="specify number of optuna trials")
    parser.add_argument("-database_name", "--database_name", type=str, default="timeGAN_default", help="specify the database")
    parser.add_argument("-shift_numbers", "--shift_numbers", nargs='*', type=int, default=(0,), help="specify shifts of the data")
    
    args = parser.parse_args()

    data = args.data
    num_samples = args.num_samples
    n_trials = args.n_trials
    database_name = args.database_name
    shift_numbers = args.shift_numbers
    
    run_TimeGAN(data, num_samples, n_trials, database_name, *shift_numbers)
