import numpy as np
import os
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt
from utils.data_loader_schachtschneider import real_data_loading, postprocess_data
from sdv.evaluation import evaluate
import optuna
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


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
    # hyperparameter
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6, 8, 10, 12])
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


def run_DTW(n_trials, database_name, *shift_numbers):
    """
    Run DTW
    :param n_trials: number of optimization trials
    :param database_name: name of the project
    :param shift_numbers: number of days for which the dataset should be shifted. Can be multiple days as well as positive and negative
    """
    # load and prepare data
    x_train, y_train, num_columns_cat, dataset_val, dataset, labels, columns, shift_numbers = prepare_train_data(database_name, 
                                                                                                                 *shift_numbers)
    
    # optimize hyperparameters
    study = optuna.create_study(storage=optuna.storages.RDBStorage("sqlite:///" + database_name + ".db"), 
                                study_name = database_name + "_study", direction="maximize", load_if_exists=True)
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
    
    # plots of optuna optimization
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    # fig2 = optuna.visualization.matplotlib.plot_intermediate_values(study): plot if pruning would be activated
    fig3 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig4 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig5 = optuna.visualization.matplotlib.plot_contour(study)
    fig6 = optuna.visualization.matplotlib.plot_slice(study)
    fig7 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig8 = optuna.visualization.matplotlib.plot_edf(study)
    plt.show()
    
    # generate data and stop time for this task
    start_time = time.time()
    generated_data = aug.discriminative_guided_warp(dataset, labels, **study.best_params)
    performance.write(str("--- %s minutes ---" % ((time.time() - start_time) / 60)))
    performance.close()
    hlp.plot1d(dataset[0], generated_data[0])

    # postprocessing
    fake_data = postprocess_data("eval", generated_data, columns, num_columns_cat, database_name, *shift_numbers)
    fake_data.to_csv('/Josef/DTW/fake_data_' + database_name + '.csv', index=False)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-n_trials", "--n_trials", type=int, default=50, help="specify number of optuna trials")
    parser.add_argument("-database_name", "--database_name", type=str, default='DGW_default', help="specify the database")
    parser.add_argument("-shift_numbers", "--shift_numbers", nargs='*', type=int, default=(0,), help="specify shifts of the data")
    
    args = parser.parse_args()

    n_trials = args.n_trials
    database_name = args.database_name
    shift_numbers = tuple(args.shift_numbers)

    run_DTW(n_trials, database_name, *shift_numbers)
    
