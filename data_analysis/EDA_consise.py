#!/usr/bin/env python
# coding: utf-8

# # Data Analysis, RIT Business Competition

# In[1]:


import folium
import requests
import json
import geojson
import numpy as np
import pandas as pd
import seaborn as sns
import typing
from datetime import datetime
from scipy.stats import percentileofscore
from matplotlib import pyplot as plt


import warnings
from IPython.display import HTML, display
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Data cleaning

# ### Shops data

# In[2]:


df_info = pd.read_csv(r'pharm_info.csv')


# In[3]:


df_info.head()


# In[4]:


#Dropping stores with no store numbers and coercing type
df_info = df_info.dropna(subset=['storenum'])

#Changing zipcode dtype
df_info['zipcode'] = df_info['zipcode'].astype(str)


# In[5]:


df_info.info()


# In[6]:


#Dropping columns that I don't think will be useful for data analysis
df_info = df_info.drop(['telephone','address1','address2','state','website','zipcode4'],axis = 1)


# In[7]:


df_info.info()


# ### Sales data

# In[8]:


df_sales = pd.read_csv(r'pharm_sales.csv')


# In[9]:


#Changing date column to datetime object
df_sales['date'] = pd.to_datetime(df_sales['date'])
df_sales.info()


# ### Merging the two dataframes

# In[10]:


#Merging both of them using the storenumber field as that is unique for stores.

df_merged = pd.merge(df_sales, df_info, on='storenum', how = 'outer')


# ### Renaming columns

# In[11]:


df_merged = df_merged.rename(columns = {'sales_num':'revenue_original'})


# In[12]:


df_merged.info()


# In[13]:


# Dropping nulls
df_merged = df_merged.dropna()

df_merged = df_merged.reset_index(drop=True)


# ### Adding new fields and fixing types

# In[14]:


# Original sales figures are off, this is probably a safer bet, might ignore taxes
df_merged['revenue_original'] = df_merged['product_retail_price'] * df_merged['units_sold']

# Total cost per sale.
df_merged['inventory_cost'] = df_merged['productcost'] * df_merged['units_sold']

# inventory costs are 68% of a businesses cost
#https://blog.shelvingdesignsystems.com/what-does-it-really-cost-to-start-a-pharmacy
df_merged['total_cost'] = df_merged['inventory_cost']/0.68

#Cost of non inventory overhead
df_merged['overhead_cost'] = df_merged['total_cost'] - df_merged['inventory_cost']

# Profit per sale
df_merged['profit'] = df_merged['revenue_original'] - df_merged['total_cost']

# Margin per unit
df_merged['margin'] = df_merged['product_retail_price'] - df_merged['productcost']

# Making store number field a string
df_merged['storenum'] = df_merged['storenum'].astype(str)


# In[15]:


df_merged.info()


# #### The plot below will help me see whether I should worry too much about dropping shops or not.

# If there is a huge range in variance between sales, then I should.

# In[16]:


df_merged.groupby(['date'])['units_sold'].sum().plot(color = 'green')


# Sale patterns are random, IE not that much yearly variation, thus I will drop pharmicies which made no sales in 2021, which I will assume is an indication for being closed.

# ### Dropping shops that probably closed.

# Do we know the stores closed beyond a shadow of a doubt? No.
# 
# 
# However doing any analysis would be moot if we don't have the data normalized in some way shape or form. It seems to me the best way to do that would be to normalize over a specific period of time. And to further normalize the data, I'll drop stores that did not have any sales past the year of 2021, then analyze with only those stores over a reasonable timeframe, where the loss of data wouldn't be too high.

# In[17]:


#Making a new dataframe with only sales past 2021

df_merged_open = df_merged.loc[df_merged['date'] >= '2021-01-01']

df_merged_open = df_merged_open.reset_index(drop = True)


# In[18]:


# Finding the shops that had sales in 2021, and keeping the values in a variable to assign to another dataframe later.

shops_still_open = df_merged_open['storenum'].unique()

shops_still_open


# In[19]:


# Number of stores still open.

df_merged_open['storenum'].nunique()


# In[20]:


52/69


# I lost 25% of the stores by only keeping the stores that were open in 2021, I need to see how much of the total dataset I will lose.

# In[21]:


df_merged_final = df_merged[(df_merged['storenum'].isin(shops_still_open)) & (df_merged['date'] >= '2018-01-01')]

df_merged_final = df_merged_final.reset_index(drop = True)


# In[22]:


df_info_final = df_info.loc[df_info['storenum'].isin(shops_still_open)]
df_info_final = df_info_final.reset_index(drop = True)

df_info_final.info()


# In[23]:


df_merged_final.info()


# In[24]:


# Saving as csv

#df_merged_final.to_csv('pharm_sales_final.csv')
#df_info_final.to_csv('pharm_info_final.csv')


# # EDA

# ## Zipcode

# #### Profit

# In[25]:


total_profit_zipcode_groups = df_merged_final.groupby(['zipcode']).sum().sort_values(['profit'], ascending = True)['profit']

total_profit_zipcode_groups.sort_values().head(10)


# In[26]:


total_profit_zipcode_groups.plot(kind = 'bar',
                             color = 'green',
                            ylabel = 'Total ($) ',
                            title = 'Profit per zipcode')


# In[27]:


number_stores_zipcode = df_info_final.groupby(['zipcode']).count()['storenum']

number_stores_zipcode.head(10)


# In[28]:


profit_per_zipcode = total_profit_zipcode_groups/number_stores_zipcode
profit_per_zipcode.sort_values().head()


# In[29]:


profit_per_zipcode.max()/profit_per_zipcode.mean()


# In[30]:


profit_per_zipcode.sort_values().plot(kind = 'bar',
                             color = 'green',
                            ylabel = 'Total ($) ',
                            title = 'Average Profit/ Zipcode')


# ## Choropleth map

# In[31]:


# Link below has all the zipcode boundaries in NY state, I'll only keep the ones I need
url='https://github.com/OpenDataDE/State-zip-code-GeoJSON/raw/master/ny_new_york_zip_codes_geo.min.json'
gj =  requests.get(url).json()


# In[32]:


# Making new geojson file
ziplist = df_merged_final['zipcode'].unique() #list of zips in your dataframe

inziplist = []
for  ft in gj['features']:
    if ft['properties']['ZCTA5CE10'] in ziplist:
        inziplist.append(ft)
        
new_zip_json = {}
new_zip_json['type'] = 'FeatureCollection'
new_zip_json['features'] = inziplist


# In[33]:


#Turning the series above into a dataframe
df_zip = pd.DataFrame({'ZCTA5CE10': [x for x in profit_per_zipcode.index],
                       'average_profit': [x for x in profit_per_zipcode.values]})

# Casting type as string so that parser can parse connect it to geojson file
df_zip['ZCTA5CE10'] = df_zip['ZCTA5CE10']

m = folium.Map(location=[43.191229,-77.5], zoom_start=11, tile = 'stamenwatercolor')



style_function = "font-size: 15px; font-weight: bold"

choropleth = folium.Choropleth(
    geo_data = new_zip_json,
    name = "choropleth",
    data = df_zip,
    columns = ['ZCTA5CE10', 'average_profit'],
    key_on = 'feature.properties.ZCTA5CE10',
    fill_color = "YlGn",
    fill_opacity = 4,
    line_opacity = 2,
    legend_name = "Average profit",
    labels = True
).add_to(m)


# add labels indicating zipcode
style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['ZCTA5CE10'], style=style_function, labels=False))


folium.LayerControl().add_to(m)

m


# ## Heatmap

# In[34]:


df_zip = df_zip.rename(columns = {'ZCTA5CE10': 'zipcode'})
df_zip.info()


# In[35]:


df_demographics = pd.read_csv('zip_demographics.csv')

df_demographics['zipcode'] = df_demographics['zipcode'].astype(str)

df_demographics.head()


# In[36]:


#Label encoding
encode_dict = {"SchoolTestPerformance":     {'Poor': 1, 'Below Average': 2, 'Average':3,
                                            'Above Average':4, 'Excellent':5},"RacialMajority": {'White': 0, 'Black':1}}

df_demographics = df_demographics.replace(encode_dict)


# In[37]:


df_demographics.info()


# In[38]:


df_zips_merged = pd.merge(df_demographics, df_zip, on='zipcode', how = 'outer')
df_zip_num = df_zips_merged.select_dtypes(include = ['float64', 'int64'])


# In[39]:


plt.rcParams['figure.figsize'] = [10,7]
ax = sns.heatmap(df_zip_num.corr(), annot=True, cmap = "Greens")


# ## Store , most profitible

# In[40]:


total_profit_store_num = df_merged_final.groupby(['storenum']).sum().sort_values(['profit'], ascending = True)['profit']

total_profit_store_num.sort_values().head()

#Keep 4641 in mind


# In[41]:


total_profit_store_num.describe()


# In[42]:


total_profit_store_num.sort_values().plot(kind = 'bar',
                             color = 'green',
                            ylabel = 'Total ($) ',
                            title = 'Profit / Store (number)',
                                         figsize = (15,5))


# In[43]:


total_profit_store_num.plot.kde(color = 'green', title = 'Profit distribution (shops)')


# ## Most profitible products

# In[44]:


profit_products = df_merged_final.groupby(['itemnum'])['profit'].sum().sort_values().tail(500)

profit_products.index


# ## Shop reccomendations

# ### Pre-eliminary calculations

# In[45]:


stores_14618 = df_merged_final.loc[df_merged_final['zipcode'] == '14618' ]['storenum'].unique()
stores_14618


# In[46]:


total_profit_store_num['4641'] #61.5th percentile


# In[47]:


percentile_of_4641 = percentileofscore(total_profit_store_num, total_profit_store_num['4641'])
percentile_of_4641


# ### Helper functions

# In[48]:


def sale_timedelta(storenumber,df):
    # Returns number of days between first and last sale in specific shops
    
    start = df.loc[df['storenum'] == storenumber]['date'].describe()['first']
    stop = df.loc[df['storenum'] == storenumber]['date'].describe()['last']

    return int((stop-start).days)

sale_timedelta('4641',df_merged_final)


# In[49]:


# All unique stores.
storenums = df_merged_final['storenum'].unique()
storenums


# In[50]:


# Getting time delta of all stores
stores_daysdelta_list = np.array([(store,sale_timedelta(store,df_merged_final)) for store in storenums])


# In[51]:


stores_daysdelta_list[0:5]


# In[52]:


#Turning array to series, for index matching
idx, values = zip(*stores_daysdelta_list)

stores_timedelta_series = pd.Series((int(x) for x in values), idx)


# In[53]:


(6041.834620 + 62.212278)/2


# In[54]:


# Finding profit per day
store_profit_perday= total_profit_store_num/stores_timedelta_series

store_profit_perday.sort_values()


# In[55]:


for index in store_profit_perday.tail().index:
    print(store_profit_perday[index])


# ### Comparison plots

# In[56]:


# class else write 100 lines of code
class Option:
    def __init__(self, name: str,cost: float,daily_profit: float,fixed_cost: float,n: int,days: int,investment: bool):

        self.name = name
        self.cost = cost
        self.daily_profit = daily_profit
        self.fixed_cost = fixed_cost
        self.n = n
        self.days = days
        self.investment = investment

    def cashplot(self):
        if self.n != 1 or self.n != 0:
            self.cost = (self.cost*self.n)-(self.fixed_cost*(self.n-1))
            self.daily_profit = self.daily_profit*self.n


        cashsum_pharmacy = 0
        pharmacy_profit = [self.daily_profit for i in range(1,self.days)]
        pharmacy_cash = np.append([self.cost+self.daily_profit],pharmacy_profit) if self.investment         else np.append([-self.cost+self.daily_profit],pharmacy_profit)

        summed_list = []
        for v in pharmacy_cash:
            cashsum_pharmacy += v
            summed_list.append(cashsum_pharmacy)

        return summed_list
    
    def breakeven(self):
        if self.investment:
            return 0
        return self.cost//self.daily_profit
        

    def profit(self):
        if self.investment:
            return self.cashplot()[-1] - self.cost
        return self.cashplot()[-1]
    
    def plot(self):
        interval = range(1,self.days+1)
        plt.rcParams['figure.figsize'] = [15, 10] 
        plt.plot(interval, self.cashplot(), label=self.name)
        plt.ylabel('$')
        plt.xlabel('days')
        plt.title('Projected earnings over 10 years')
        plt.legend()
        plt.grid(True)


# In[57]:


Inv_list = [Option(dailyprofit,3e5,store_profit_perday[dailyprofit],0,1,3650,False)            for dailyprofit in store_profit_perday.sort_values().tail(25).index]

Inv_list.append(Option('Building pharmacy',1_764_705,3000,0,1,3650,False))


# In[58]:


for v in Inv_list:
    v.plot()


# In[59]:


breakeven_list = [(store.name,store.profit()) for store in Inv_list]
breakeven_list


# In[60]:


products_4641 = set(df_merged_final.loc[df_merged_final['storenum'] == '4641']['itemnum'].unique())


# In[61]:


len(products_4641)

