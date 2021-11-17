from table_evaluator import load_data, TableEvaluator
import pandas as pd
from sdv.evaluation import evaluate
import warnings
warnings.filterwarnings("ignore")


def test_samples(database_name, print_data=False):
    """
    evaluates the fake data of the model
    :param database_name: name of the project
    :param print_data: weather to print the fake dataset or not
    """
    # loads test and fake data
    test, fake = load_data('test_data_MonthlyMilkProduction.csv', 'fake_data_' + database_name + '.csv')
    
    # prints test and real data if flag print_data is set to true
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(fake)
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(test)
    
    
    table_evaluator = TableEvaluator(test, fake)
    # prints visual evaluation of the table evaluator
    try:
        table_evaluator.visual_evaluation()
    except:
        pass
    # print statistical evaluation of the table evaluator
    try:
        table_evaluator.evaluate(target_col='milk', target_type='regr')
    except:
        pass
    # print the calculation values of the similarity score of the sdv evaluation tool
    print(evaluate(fake, test, aggregate=False))