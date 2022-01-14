import pandas as pd
import datetime
import holidays

import ferien


def add_public_holidays(df: pd.DataFrame, federal_state: str):
    """
    Add public holidays in specific federal state
    :param df: DataFrame where data should be added
    :param federal_state: federal_state for retrieving data
    :return: Dataframe with public holidays added
    """
    federal_state_holidays = holidays.CountryHoliday(country='Germany', prov=federal_state, years=[2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 
                                                                                                   2014, 2015, 2016, 2017, 2018, 2019, 2020])
    if 'public_holiday' not in df.columns:
        df['public_holiday'] = 'no'
    for date in df.index:
        if date in federal_state_holidays:
            df.at[date, 'public_holiday'] = federal_state_holidays[date]
    
    return df


def add_school_holidays(df: pd.DataFrame, federal_state: str):
    """
    Add school holidays in specific federal state
    :param df: DataFrame where data should be added
    :param federal_state: federal_state for retrieving data
    :return: Dataframe with school holidays added
    """
    for year in range(2001, 2021):
        federal_state_school_holidays = ferien.state_vacations(state_code=federal_state, year=year)
        if 'school_holiday' not in df.columns:
            df['school_holiday'] = 'no'
        index = df['Auf. Datum']
        for date in index:
            for vac in federal_state_school_holidays:
                if vac.start.date() <= date <= vac.end.date():
                    df.at[date, 'school_holiday'] = vac.name
                
    return df
    
