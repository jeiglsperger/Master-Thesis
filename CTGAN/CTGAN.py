import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from ctgan import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from table_evaluator import load_data, TableEvaluator
import time
from sdv.evaluation import evaluate
import optuna
import joblib
import matplotlib.pyplot as plt
import warnings
import argparse
import torch 
warnings.filterwarnings("ignore")

print("Is available:") 
print(torch.cuda.is_available())
print("Current Device:")
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print("Device Count:")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


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
    :param data: Dataset that should get imputed
    :return: Imputed dataset
    """
    imputer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
    imputed_dataset = pd.DataFrame(imputer.fit_transform(dataset))
    imputed_dataset.columns = dataset.columns
    imputed_dataset.index = dataset.index

    return imputed_dataset


def shift_dataset(data, preprocessed_dataset, *shift_numbers):
    """
    Adds shifted dataset
    :param preprocessed_dataset: Dataset to which the shifted dataset should be added
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: Dataset with the shifted dataset(s) added
    """
    if data == "CashierData.csv":
        to_shift_dataset = preprocessed_dataset.drop(columns=["quarter"])
    else:
        to_shift_dataset = preprocessed_dataset
    for shift_number in shift_numbers:
        shifted_dataset = to_shift_dataset.shift(periods=shift_number)
        shifted_dataset.columns = shifted_dataset.columns + "_shifted" + str(shift_number)
        preprocessed_dataset = pd.concat([preprocessed_dataset, shifted_dataset], axis=1)
        preprocessed_dataset = preprocessed_dataset.iloc[:preprocessed_dataset.shape[0]+shift_number, :]
        preprocessed_dataset = preprocessed_dataset.reset_index(drop=True)
        to_shift_dataset = to_shift_dataset.drop(to_shift_dataset.tail(abs(shift_number)).index)

    return preprocessed_dataset, shift_numbers


def preprocess_dataset(data, *shift_numbers):
    """
    Loads and preprocessed the dataset
    :param data: define which dataset should be used
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: training data, validation data, data to train the optimized model, number of days for the shifted dataset 
    """
    dataset = load_datasets(data)

    
    dataset = impute_dataset(dataset)

    # test-train-split
    dataset, dataset_test = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=False)
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.25, random_state=42, shuffle=False)

    dataset_test.to_csv('test_data_' + data, index=False)

    # delete columns that can be derived from others
    if data == "CashierData.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h", "total_prec_flag"])
    elif data == "Schachtschneider_externals_cut.csv":
        dataset_train = dataset_train.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])
        dataset = dataset.drop(columns=["total_prec_height_mm", "total_sun_dur_h"])

    if shift_numbers != (0,):
        dataset_train, shift_numbers = shift_dataset(data, dataset_train, *shift_numbers)
    if shift_numbers != (0,):
        dataset, shift_numbers = shift_dataset(data, dataset, *shift_numbers)

    return dataset_train, dataset_val, dataset, shift_numbers


def postprocess_dataset(data, eval_or_opt, samples, *shift_numbers):
    """
    Postprocesses synthesized data
    :param data: define which dataset should be used
    :param eval_or_opt: "opt" for the optimization loop, "eval" for training the optimized model
    :param samples: synthesized data from the model
    :shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: postprocessed fake data
    """
    # delete shifted data
    if data == "CashierData.csv":
        if shift_numbers != (0,):
            samples = samples.iloc[:, :len(shift_numbers) * -14]
    if data == "Schachtschneider_externals_cut.csv":
        if shift_numbers != (0,):
            samples = samples.iloc[:, :len(shift_numbers) * -789]
    if data == "Public_MonthlyMilkProduction.csv":
        if shift_numbers != (0,):
            samples = samples.iloc[:, :len(shift_numbers) * -1]
    if data == "Public_QuarterlyTouristsIndia.csv":
        if shift_numbers != (0,):
            samples = samples.iloc[:, :len(shift_numbers) * -42]

    # set all values that can no be negativ in reality to zero
    for column in samples.columns:
        if (column != "mean_temp" and column != "total_prec_flag" and column != "public_holiday" and column != "school_holiday" and column != "weekday"):
            samples[column][samples[column] < 0] = 0

    # insert columns again that can be derived from others
    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        total_sun_dur_h_fake = (samples["mean_sun_dur_min"] / 60) * 24
        samples.insert(11, 'total_sun_dur_h', total_sun_dur_h_fake)
        total_prec_height_mm_fake = samples["mean_prec_height_mm"] * 24
        samples.insert(9, 'total_prec_height_mm', total_prec_height_mm_fake)
            
    if data == "CashierData.csv":
        total_prec_flag_fake = samples['mean_prec_flag'].apply(lambda x: 'True' if x > 0 else 'False')
        samples.insert(11, 'total_prec_flag', total_prec_flag_fake)
            
    fake_data = samples

    return fake_data


best_scores=0

def objective(trial, preprocessed_dataset, dataset_val, num_samples, database_name, *shift_numbers):
    """
    Objective function for hyperparameter optimization with optuna
    :param trail: current optimization trial
    :param preprocessed_dataset: training data
    :param dataset_val: validation data
    :param num_samples: number of fake samples that should be generated
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    :return: score of the optimization
    """    
    if trial.number == 100:
        trial.study.stop()
     
    # hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [2, 4, 8])
    generator_dims_dict = {
      "16, 16": (16, 16), 
      "32, 32": (32, 32), 
      "64, 64": (64, 64), 
    }
    generator_dims_key = trial.suggest_categorical("generator_dims_key", ["16, 16", "32, 32", "64, 64"])
    generator_dim = generator_dims_dict[generator_dims_key]
    discriminator_dims_dict = {
      "16, 16": (16, 16), 
      "32, 32": (32, 32), 
      "64, 64": (64, 64),
    }
    discriminator_dims_key = trial.suggest_categorical("discriminator_dims_key", ["16, 16", "32, 32", "64, 64"])
    discriminator_dim = discriminator_dims_dict[discriminator_dims_key]
    generator_lr = trial.suggest_categorical("generator_lr", [0.5, 1e-1, 1e-2])
    generator_decay = trial.suggest_categorical("generator_decay", [1e-6, 1e-7, 1e-8])
    discriminator_lr = trial.suggest_categorical("discriminator_lr", [0.5, 1e-1, 1e-2])
    discriminator_decay = trial.suggest_categorical("discriminator_decay", [1e-1, 1e-2, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [10, 20, 30]) # must be a multiple of 10
    discriminator_steps = trial.suggest_int("discriminator_steps", 1, 5, 1)
    epochs = trial.suggest_categorical("epochs", [5, 10, 20])

    # load CTGAN
    ctgan = CTGANSynthesizer(embedding_dim=embedding_dim, generator_dim=generator_dim,
                             discriminator_dim=discriminator_dim, generator_lr=generator_lr,
                             generator_decay=generator_decay, discriminator_lr=discriminator_lr,
                             discriminator_decay=discriminator_decay, batch_size=batch_size,
                             discriminator_steps=discriminator_steps, epochs=epochs)
    
    # set discrete columns
    if data == "CashierData.csv":
        discrete_columns = [
            'public_holiday',
            'school_holiday',
            'weekday',
            'quarter']
        if shift_numbers != (0,):
            for shift_number in shift_numbers:
                discrete_columns.append('public_holiday_shifted' + str(shift_number))
                discrete_columns.append('school_holiday_shifted' + str(shift_number))
                discrete_columns.append('weekday_shifted' + str(shift_number))
    elif data == "Schachtschneider_externals_cut.csv":
        discrete_columns = [
            'public_holiday',
            'weekday']
        if shift_numbers != (0,):
            for shift_number in shift_numbers:
                discrete_columns.append('public_holiday_shifted' + str(shift_number))
                discrete_columns.append('weekday_shifted' + str(shift_number))
        

    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        try:
            ctgan.fit(preprocessed_dataset, discrete_columns)
        except:
            pass
    else:
        ctgan.fit(preprocessed_dataset)

    # generate data
    samples = ctgan.sample(num_samples)

    # postprocessing
    fake_data = postprocess_dataset(data, "opt", samples, *shift_numbers)

    # calculate score
    scores = evaluate(fake_data, dataset_val)
    
    # save best model
    global best_scores
    if scores > best_scores:
        joblib.dump(ctgan, database_name + '.gz')
        best_scores = scores

    return scores


def run_CTGAN(data, num_samples: int, n_trials: int, database_name, *shift_numbers):
    """
    Run CTGAN
    :param data: define which dataset should be used
    :param num_samples: number of fake samples that should be generated
    :param n_trials: number of optimization trials
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    """
    # load and prepare data
    preprocessed_dataset, dataset_val, dataset, shift_numbers = preprocess_dataset(data, *shift_numbers)

    # optimize hyperparameters
    study = optuna.create_study(storage=optuna.storages.RDBStorage("sqlite:///" + database_name + ".db"), 
                                study_name = database_name + "_study", direction="maximize", load_if_exists=True)
    study.optimize(lambda trial: objective(trial, preprocessed_dataset, dataset_val, num_samples, database_name, *shift_numbers), n_trials)

    # save performance parameters
    performance = open("performance_" + database_name + ".txt","w+")
    performance.write(str(study.best_params))
    performance.write(str(study.best_value))
    performance.write(str(study.best_trial))


    # load best model
    ctgan = joblib.load(database_name + '.gz')
    if data == "CashierData.csv":
        discrete_columns = [
            'public_holiday',
            'school_holiday',
            'weekday',
            'quarter']
        if shift_numbers != (0,):
            for shift_number in shift_numbers:
                discrete_columns.append('public_holiday_shifted' + str(shift_number))
                discrete_columns.append('school_holiday_shifted' + str(shift_number))
                discrete_columns.append('weekday_shifted' + str(shift_number))
    elif data == "Schachtschneider_externals_cut.csv":
        discrete_columns = [
            'public_holiday',
            'weekday']
        if shift_numbers != (0,):
            for shift_number in shift_numbers:
                discrete_columns.append('public_holiday_shifted' + str(shift_number))
                discrete_columns.append('weekday_shifted' + str(shift_number))
    
    # generate data and stop time for this task
    start_time = time.time()
    if data == "CashierData.csv" or data == "Schachtschneider_externals_cut.csv":
        ctgan.fit(preprocessed_dataset, discrete_columns)
    else:
        ctgan.fit(preprocessed_dataset)
    samples = ctgan.sample(num_samples)
    performance.write(str("--- %s minutes ---" % ((time.time() - start_time) / 60)))
    performance.close()

    # postprocessing
    fake_data = postprocess_dataset(data, "eval", samples, *shift_numbers)
    fake_data.to_csv('fake_data_' + database_name + '.csv', index=False)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-data", "--data", type=str, default="CashierData.csv", help="define data")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=1000, help="specify number of samples")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=100, help="specify number of optuna trials") # has to be at least 4 or higher
    parser.add_argument("-database_name", "--database_name", type=str, default="CTGAN_default", help="specify the database")
    parser.add_argument("-shift_numbers", "--shift_numbers", nargs='*', type=int, default=(0,), help="specify shifts of the data")
    
    args = parser.parse_args()
    
    data = args.data
    num_samples = args.num_samples
    n_trials = args.n_trials
    database_name = args.database_name
    shift_numbers = tuple(args.shift_numbers)

    run_CTGAN(data, num_samples, n_trials, database_name, *shift_numbers)
