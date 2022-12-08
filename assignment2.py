# Importing Libraries for data handling
import numpy as np
import pandas as pd

# Importing Libraries for Visualisation 
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(filename: str):
    '''
        Takes a string which represents the file name of the dataset in csv format and returns a pandas dataframe in 
        the original and a transposed format
        
        Args:
            filename => str, representing the name of csv file
        Returns:
            original_df => pandas.Dataframe, original format
            transposed_df => pandas.Dataframe, tranposed format     
    '''
    #read csv file
    original_df = pd.read_csv(f'{filename}.csv')
    
    # transpose, rename dataframe columns, drop non-year rows and convert whole df to float64 dtypes
    transposed_df = original_df.set_index('Country Name').T
    second_column_header = transposed_df.iloc[1].tolist()
    transposed_df.columns = [ transposed_df.columns,  second_column_header ]
    transposed_df.drop(['Country Code','Indicator Name', 'Indicator Code'], axis=0, inplace=True)
    transposed_df = transposed_df.apply(pd.to_numeric, errors='coerce')
    
    return original_df, transposed_df

original_df, transposed_df = read_data('API_19_DS2_en_csv_v2_4700503')

print(original_df.shape)
print(original_df.info())

print(original_df.head(), transposed_df.head(), sep='\n')


# Let's explore all the indicators
no_of_indictor = original_df['Indicator Name'].nunique()
print(f'We have {no_of_indictor} Indicators in this data set')

# let look at the Indicators
print(original_df['Indicator Name'].unique())

#GDP Indicator is missing from our dataset we would bring this in later on


# run statistical analysis across indicactors for different countries
def get_stat(country: str, indicators : list):
    '''
        Explore the statistical properties of a few indicators for a country
        
        Args:
            country => str, representing the country
            indicators => list, containing indicators of interest
        Returns:
            summary_stat => pandas.Dataframe, a statistical summary of the selected indicators 
    '''
    return transposed_df[country].describe()[indicators]


ind_of_interest = ['Urban population', 'Total greenhouse gas emissions (kt of CO2 equivalent)', 'Electric power consumption (kWh per capita)', 'Agricultural land (sq. km)','Electricity production from oil sources (% of total)', 'Electricity production from coal sources (% of total)', 'Foreign direct investment, net inflows (% of GDP)' ]

nigeria_stat = get_stat('Nigeria', ind_of_interest)
print(nigeria_stat)

US_stat = get_stat('United States', ind_of_interest)
print(US_stat)

Brazil_stat = get_stat('Brazil', ind_of_interest)
print(Brazil_stat)

Germany_stat = get_stat('Germany', ind_of_interest)
print(Germany_stat)



def compare_stat_of_countries(countries: list, indicator : str):
    '''
        compares the statistical properties of an indicators accross different countries
        
        Args:
            countries => list, representing the list of countries to compare indicator activities
            indicator => str, representing indicators for comparison
        Returns:
            summary_stat => pandas.Dataframe, a statistical summary of the selected countries 
    '''
    country_stat_list = []
    for country in countries:
        stat = transposed_df[country].describe()[indicator]
        stat.name = country
        country_stat_list.append(stat)
    
    return pd.concat(country_stat_list, axis=1)


# compare the following indicators (Electric power consumption, Total greenhouse gas emissions ) stat for the countries below
countries = ['Bangladesh', 'Brazil', 'Canada', 'China', 
                      'Ecuador', 'France', 'India', 'Nigeria', 'South Africa', 'Sweden', 'United Kingdom', 'United States' ]

print(compare_stat_of_countries(countries, 'Electric power consumption (kWh per capita)') )

print(compare_stat_of_countries(countries, 'Total greenhouse gas emissions (kt of CO2 equivalent)') )


#Now lets bring in GDP as we are interested in how it impacted and was impacted by other indicators
gdp_df = pd.read_csv('GDP.csv')
df = pd.concat([original_df, gdp_df])
print(df.head())


