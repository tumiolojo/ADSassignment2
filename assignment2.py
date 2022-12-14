# Importing Libraries for data handling
import numpy as np
import pandas as pd

# Importing Libraries for Visualisation 
import matplotlib.pyplot as plt


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
countries = ['Gabon', 'Oman', 'India', 'Brazil', 'Peru', 'Lithuania', 'Mexico', 'France']

print(compare_stat_of_countries(countries, 'Electric power consumption (kWh per capita)') )

print(compare_stat_of_countries(countries, 'Total greenhouse gas emissions (kt of CO2 equivalent)') )



# Exploring our relationship between different indicator across some countries
def bar_chart(df, countries: list, indicator: str):
    '''
        Returns a bar chart representing the indicator performance of several countries over the years
        
        Args:
            df => pandas.Dataframe, original data frame format
            countries => list, of countries of interest
            indicator => str, selected indicator
        Returns:
            plot => barchart 
    '''
    
    #filter dataframe to contain slected cunties and indicator
    df = df[df['Country Name'].isin(countries)]
    df = df[df['Indicator Name'] == indicator]
    
    df = df[['Country Name', '1990', '1995', '2000', '2005', '2010', '2015']]
    df.set_index('Country Name').plot.bar(figsize = (14,9)) 
    plt.title(indicator)
    plt.savefig("barchart.png")
    

# Plot Energy consumed for the selected countries from 1990 to 2015 at 5 years increment
bar_chart(original_df, countries, 'Energy use (kg of oil equivalent per capita)')


# Plot Access to electricity for the selected countries from 1990 to 2015 at 5 years increment
bar_chart(original_df, countries, 'Access to electricity (% of population)')


# Plot CO2 emission for the selected countries from 1990 to 2015 at 5 years increment
bar_chart(original_df, countries, 'CO2 emissions (kt)')

# We can see that trend is specific to North America, Now lets explore USA
def heatmap(df, country: str, indicators: list):
    '''
        plot the heatmap of country for different indicators
        
        Args:
            df => pandas.Dataframe, original format
            countries => str, selected countries 
            indicator => list, of indicators to check correlation on
            
        Returns:
            plot => line chart 
    '''
    
    coun_df = df[(df['Country Name'] == country) & (df['Indicator Name'].isin(indicators))]
    
    corr_df = coun_df.set_index('Indicator Name').iloc[:, 3:].T
    corr_df.columns.name = None
    corr_df.index.name = None
    corr = corr_df.corr()
    
    # plot heat map
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="rainbow")
    cbar = ax.figure.colorbar(im, ax = ax, shrink=0.5 )
    
    # set x, y axis labels and rotate x-label by 90
    ax.set_xticks(np.arange(len(corr.index.values)), labels=corr.index.values)
    ax.set_yticks(np.arange(len(corr.columns.values)), labels=corr.columns.values)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    
    #annotate data value on heat
    for i in range(len(corr.columns.values)):
        for j in range(len(corr.index.values)):
            text = ax.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center", color="w")
    
    plt.title(country)
    plt.savefig("heatmap.png")
    

ind_of_interest = ['Energy use (kg of oil equivalent per capita)', 'Electricity production from nuclear sources (% of total)', 'Access to electricity (% of population)', 'Forest area (% of land area)', 'Cereal yield (kg per hectare)', 'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)', 'CO2 emissions (kt)'] 

# check indicator correlation for India
heatmap(original_df, 'India', ind_of_interest)




def line_plot(df, countries: list, indicator: str):
    '''
        Plots a line chart showing the indicator performance of certain countries over the years
        
        Args:
            df => pandas.Dataframe, original format
            countries => list, of countries of interest
            indicator => str, selected indicator

        Returns:
            plot => line chart 
    '''
    
    coun_df = df[(df['Country Name'].isin(countries)) & (df['Indicator Name'] == indicator)]
    
    coun_df = coun_df.set_index('Country Name').iloc[:, 3:].T
    coun_df.columns.name = None
    
    plt.style.use('seaborn-white')

    coun_df.plot( figsize=(14, 12), linestyle='--' )

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title(f'{indicator} trend' )
    plt.savefig("lINE PLOT.png")
    plt.show()
  
 

#Exploring trend of cereal yield and forest area   
for ind in ['Forest area (% of land area)', 'Cereal yield (kg per hectare)']:
    line_plot(original_df, ['Oman', 'India', 'France'], ind)
    
#plot heatmap for France
heatmap(original_df, 'France', ind_of_interest)


#plot heatmap for Peru
heatmap(original_df, 'Peru', ind_of_interest)







