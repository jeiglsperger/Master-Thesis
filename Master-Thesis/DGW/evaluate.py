from table_evaluator import load_data, TableEvaluator
import pandas as pd
from sdv.evaluation import evaluate
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import optuna


def test_samples(data, database_name="default", print_data=False):
    """
    evaluates the fake data of the model
    :param database_name: name of the project
    :param print_data: weather to print the fake dataset or not
    """
    # loads test and fake data
    test, fake = load_data('test_data_' + data, 'fake_data_' + database_name + '.csv')
    
    # prints test and real data if flag print_data is set to true
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(fake)
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(test)
    
    if data == "CashierData.csv":
        target_col = 'CutFlowers'
    if data == "Public_MonthlyMilkProduction":
        target_col='milk'
    if data == "Public_QuarterlyTouristsIndia":
        target_col='TouristsIndia'    
    if data == "Schachtschneider_externals_cut.csv":
        target_col='Lavandula'
     
    if data == "CashierData.csv":
        cat_cols = ["total_prec_flag", "public_holiday", "school_holiday", "weekday", "quarter"]
        table_evaluator = TableEvaluator(test, fake, cat_cols=cat_cols)
    elif data == "Schachtschneider_externals_cut.csv":
        cat_cols = ["public_holiday", "weekday"]
        table_evaluator = TableEvaluator(test, fake, cat_cols=cat_cols)
    else:
        table_evaluator = TableEvaluator(test, fake)
    # prints visual evaluation of the table evaluator
    try:
        table_evaluator.visual_evaluation()
    except:
        pass
    # print statistical evaluation of the table evaluator
    try:
        table_evaluator.evaluate(target_col=target_col, target_type='regr')
    except:
        pass
    # print the calculation values of the similarity score of the sdv evaluation tool
    print(evaluate(fake, test, aggregate=False))
    
    
    study = optuna.load_study(study_name=database_name+"_study", storage="sqlite:///"+database_name+".db")
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig2 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig3 = optuna.visualization.matplotlib.plot_contour(study)
    fig4 = optuna.visualization.matplotlib.plot_slice(study)
    fig5 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig6 = optuna.visualization.matplotlib.plot_edf(study)
    plt.show()