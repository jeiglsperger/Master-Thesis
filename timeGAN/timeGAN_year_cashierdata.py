from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import joblib
import numpy as np
import pandas as pd
import warnings
import time
import optuna
from timegan import timegan
from own_data_loading_cashierdata import real_data_loading, preprocess_data, cut_data
from sdv.evaluation import evaluate
from table_evaluator import load_data, TableEvaluator
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def postprocess_data(eval_or_opt, generated_data, columns, num_columns_cat, seq_len, *shift_numbers):
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
    if shift_numbers != (0,):
        generated_data = generated_data[:,:,:len(shift_numbers) * -44]
    
    # reverse one-hot-encoding     
    enc = joblib.load(eval_or_opt + "_enc_" + database_name + ".gz")
    to_inverse_transform_data = generated_data[:,:,np.shape(generated_data)[2]-num_columns_cat:np.shape(generated_data)[2]]
    inverse_transformed_data = []
    for i in range(np.shape(generated_data)[0]):
        generated_data_no_cat = generated_data[:,:,0:np.shape(generated_data)[2]-num_columns_cat]
        inverse_transformed_data.append(np.concatenate((generated_data_no_cat[i], enc.inverse_transform(to_inverse_transform_data[i],))
                                                       , axis=1))
    inverse_transformed_data = np.array(inverse_transformed_data)

    conditioned_dataset = []
    for i in range(np.shape(generated_data)[0]):
        for j in range(seq_len):
            conditioned_dataset.append(inverse_transformed_data[i,j,:])
                
    # transform to a pandas dataframe
    generated_data_df = pd.DataFrame(data=conditioned_dataset, columns=columns)
    
    # insert columns again that can be derived from others
    total_sun_dur_h = (generated_data_df["mean_sun_dur_min"] / 60) * 24
    generated_data_df.insert(11, 'total_sun_dur_h', total_sun_dur_h)
    total_prec_height_mm = generated_data_df["mean_prec_height_mm"] * 24
    generated_data_df.insert(9, 'total_prec_height_mm', total_prec_height_mm)
    total_prec_flag = generated_data_df['mean_prec_flag'].apply(lambda x: 'True' if x > 0 else 'False')
    generated_data_df.insert(11, 'total_prec_flag', total_prec_flag)
    
    fake_data = generated_data_df
    
    return fake_data


best_scores = 0

def objective(trial, dataset_train, dataset_val, num_samples, columns, num_columns_cat, database_name, seq_len, *shift_numbers):
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
    if trial.number == 50:
        trial.study.stop()
    
    # hyperparameters
    module = trial.suggest_categorical("module", ["gru", "lstm", "lstmLN"])
    hidden_dim = trial.suggest_categorical("hidden_dim", [6, 12, 24, 48])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_layer = trial.suggest_int("num_layer", 3, 6, 1)
    iterations = trial.suggest_categorical("iterations", [100, 1000, 10000])
    
    # set timeGAN parameters
    parameters = dict()
    parameters['module'] = module
    parameters['hidden_dim'] = hidden_dim
    parameters['num_layer'] = num_layer
    parameters['iterations'] = iterations
    parameters['batch_size'] = batch_size
    parameters['seq_len'] = seq_len
    
    # generate data
    generated_data = timegan(dataset_train, parameters)

    # postprocessing
    fake_data = postprocess_data("opt", generated_data, columns, num_columns_cat, seq_len, *shift_numbers)

    # calculate score
    scores = evaluate(fake_data, dataset_val)
    
    # save best model
    global best_scores
    if scores > best_scores:
        joblib.dump(timegan, database_name + '.gz')
        objective.parameters = parameters
        best_scores = scores

    return scores


def run_TimeGAN(seq_len, num_samples, n_trials, database_name, *shift_numbers):
    """
    Run timeGAN
    :param num_samples: number of fake samples that should be generated
    :param n_trials: number of optimization trials
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    """    
    # load data
    dataset = real_data_loading()

    # preprocess data
    dataset_train, columns, dataset_val, dataset, num_columns_cat, shift_numbers = preprocess_data(dataset, database_name, *shift_numbers)
    

    dataset_train = cut_data(dataset_train, seq_len)
    
    #  optimize hyperparameters
    study = optuna.create_study(storage=optuna.storages.RDBStorage("sqlite:///" + database_name + ".db"), 
                                study_name = database_name + "_study_test", direction="maximize", load_if_exists=True)  # maybe use GP
    study.optimize(lambda trial: objective(trial, dataset_train, dataset_val, num_samples, columns, num_columns_cat, database_name, seq_len, 
                                           *shift_numbers), n_trials)

    # save performance parameters
    performance = open("performance_" + database_name + ".txt","w+")
    best_parameters = str(study.best_params)
    performance.write(best_parameters)
    best_values = str(study.best_value)
    performance.write(best_values)
    best_trials = str(study.best_trial)
    performance.write(best_trials)
    
    # plots of optuna optimization 
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    # fig2 = optuna.visualization.matplotlib.plot_intermediate_values(study)  # That's for pruning
    fig3 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig4 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig5 = optuna.visualization.matplotlib.plot_contour(study)
    fig6 = optuna.visualization.matplotlib.plot_slice(study)
    fig7 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig8 = optuna.visualization.matplotlib.plot_edf(study)
    plt.show()

    # generate data and stop time for this task
    timegan = joblib.load(database_name + '.gz')
    start_time = time.time()
    generated_data = timegan(dataset, objective.parameters)
    performance.write(str("--- %s minutes ---" % ((time.time() - start_time) / 60)))
    performace.close()
    
    # postprocessing
    fake_data = postprocess_data("eval", generated_data, columns, num_columns_cat, seq_len, *shift_numbers)

    fake_data.to_csv('fake_data_' + database_name + '.csv', index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-seq_len", "--seq_len", type=int, default=365, help="specify the generated sequence length")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=1, help="specify number of samples")
    parser.add_argument("-n_trials", "--n_trials", type=int, default=50, help="specify number of optuna trials")
    parser.add_argument("-database_name", "--database_name", type=str, default="timeGAN_year_default", help="specify the database")
    parser.add_argument("-shift_numbers", "--shift_numbers", nargs='*', type=int, default=(0,), help="specify shifts of the data")
    
    args = parser.parse_args()
        
    seq_len = args.seq_len
    num_samples = args.num_samples
    n_trials = args.n_trials
    database_name = args.database_name
    shift_numbers = args.shift_numbers
    
    run_TimeGAN(seq_len, num_samples, n_trials, database_name, *shift_numbers)
