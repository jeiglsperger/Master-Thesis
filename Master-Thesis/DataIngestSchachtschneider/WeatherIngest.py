import pandas as pd
from datetime import datetime
from geopy.location import Location
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationParameter, DwdObservationDataset
from dwdweather import DwdWeather


def calc_daily_mean_weather_values(location: Location) -> pd.Series:
    request = DwdObservationRequest(parameter = DwdObservationDataset.CLIMATE_SUMMARY,
                                         resolution=DwdObservationResolution.DAILY, 
                                         period=DwdObservationPeriod.HISTORICAL, start_date=datetime(2001, 1, 11), 
                                         end_date=datetime(2020, 10, 28), tidy=True, humanize=True, 
                                         si_units=True).filter_by_station_id(station_id=[963])
    df = request.values.all().df
    
    return df


def add_daily_weather_data(df: pd.DataFrame, location: Location):
    """
    Add weather related daily mean values for specific date at index of DataFrame
    :param df: DataFrame where data should be added
    :param index: index of DataFrame where data should be added
    :param location: Location for requesting weather data
    :param date: Date for requesting weather data
    """
    daily_mean_weather_values = calc_daily_mean_weather_values(location=location)
    
    temperature = daily_mean_weather_values.loc[daily_mean_weather_values['parameter'] == 'temperature_air_200']
    temperature_values = temperature['value'].to_list()
    temperature_values = [x - 273.15 for x in temperature_values]
    df['mean_temp'] = temperature_values
    
    humidity = daily_mean_weather_values.loc[daily_mean_weather_values['parameter'] == 'humidity']
    humidity_values = humidity['value'].to_list()
    df['mean_humid'] = humidity_values
        
    precipitation_height = daily_mean_weather_values.loc[daily_mean_weather_values['parameter'] == 'precipitation_height']
    precipitation_height_values = precipitation_height['value'].to_list()
    df['mean_prec_height_mm'] = precipitation_height_values
    df['total_prec_height_mm'] = [x * 24 for x in precipitation_height_values]
    
    sunshine_duration = daily_mean_weather_values.loc[daily_mean_weather_values['parameter'] == 'sunshine_duration']
    sunshine_duration_values = sunshine_duration['value'].to_list()
    df['mean_sun_dur_min'] = sunshine_duration_values
    df['total_sun_dur_h'] = [x * 24 / 60 for x in sunshine_duration_values] 
    
    return df