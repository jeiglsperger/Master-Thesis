from table_evaluator import load_data, TableEvaluator
import pandas as pd
from sdv.evaluation import evaluate
import warnings
warnings.filterwarnings("ignore")


def test_samples(database_name="default", print_data=False):
    """
    evaluates the fake data of the model
    :param database_name: name of the project
    :param print_data: weather to print the fake dataset or not
    """
    # loads test and fake data
    test, fake = load_data('/Josef/DTW/test_data_schachtschneider.csv', '/Josef/DTW/fake_data_' + database_name + '.csv')
    
    # prints test and real data if flag print_data is set to true
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(fake)
    with pd.option_context('display.max_columns', None):
        if print_data:
            print(test)
          
        
    cat_cols = ["total_prec_flag", "public_holiday", "school_holiday", "weekday", "quarter"]
    table_evaluator = TableEvaluator(test, fake, cat_cols=cat_cols)
    # prints visual evaluation of the table evaluator
    table_evaluator.visual_evaluation()
    # print statistical evaluation of the table evaluator
    table_evaluator.evaluate(target_col='CutFlowers', target_type='regr')
    
    # print the calculation values of the similarity score of the sdv evaluation tool
    print(evaluate(fake, test, aggregate=False))
