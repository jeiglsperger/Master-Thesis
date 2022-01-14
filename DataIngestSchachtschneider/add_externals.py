import pandas as pd
from geopy.geocoders import Nominatim
from tqdm import tqdm
import WeatherIngest, CalendarIngest
from sklearn.impute import SimpleImputer
import numpy as np

    
def add_externals():
    """
    Take data, add external data and save
    """
    geolocator = Nominatim(user_agent='plantgrid', timeout=3)
    location = geolocator.geocode('Schulweg 23, 26203 Wardenburg')
    federal_state = 'NI'
    
    
    df = pd.DataFrame(pd.read_csv('Schachtschneider.csv', header=0))
    
    
    print("----- Adding external data -----")
    df['Auf. Datum'] = pd.to_datetime(df['Auf. Datum'])
    df['Auf. Datum'] = df['Auf. Datum'].dt.date
    df.index = df['Auf. Datum']
    
    
    df = WeatherIngest.add_daily_weather_data(df=df, location=location)
    df = CalendarIngest.add_public_holidays(df=df, federal_state=federal_state)
    df = CalendarIngest.add_school_holidays(df=df, federal_state=federal_state)
    
    
    df['Auf. Datum'] = pd.to_datetime(df['Auf. Datum'])
    df['weekday'] = df['Auf. Datum'].dt.dayofweek
    df = df.drop("Auf. Datum", axis=1)
    
    
    for index, row in df.iterrows():
        cumsum = 0
        for i in range(783):
            cumsum += row[i]
        if cumsum == 0 and row["weekday"] != 0 and row["public_holiday"] == "no":
            for i in range(783):
                df.at[index, df.columns[i]] = np.nan
                
                
    imputer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
    imputed_df = pd.DataFrame(imputer.fit_transform(df))
    imputed_df.columns = df.columns
    imputed_df.index = df.index        
    imputed_df.to_csv('Schachtschneider.csv')
    
    
if __name__ == '__main__':

    add_externals()
