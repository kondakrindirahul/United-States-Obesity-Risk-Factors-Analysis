
# coding: utf-8

# In[1]:

# Plotly credentials stuff- run first time only
import plotly
plotly.tools.set_credentials_file(username='xalanxlp', api_key='eqGPEiLp525U75zOHDek')
# plotly.tools.set_credentials_file(username='bdesnoy', api_key='VLNSHIH0dfFWjlXNEbiG')
# plotly.tools.set_credentials_file(username='junyi', api_key='sswCNaXW5ssm3JllQCRq')


# In[2]:

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import plotly.plotly as py
from re import match
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import sklearn.metrics as sm
import statistics as stat
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


# In[3]:

pd.options.mode.chained_assignment = None  # default='warn'


# # Load the Dataset

# In[4]:

data = pd.read_csv('behavioral_risk_factor_dataset.csv')


# In[5]:

# Convert question to a category and rename categories for visibility
data["Question"] = data["Question"].astype("category")
data['Question'].cat.categories = ["% Overweight Adults",
                                   "% Obese Adults",
                                   "% Mod Aerobic", 
                                   "% Mod Aerobic & Strength", 
                                   "% Vigerous Aerobic", 
                                   "% Strength", 
                                   "% No Phys. Activity", 
                                   "% < 1 Fruit / Day", 
                                   "% < 1 Veg / Day"]

# Drop, unnecessary columns to make table human-readable
cols_to_drop = ['Datasource', 'Class', 'Topic', 'Data_Value_Unit', 
                'Data_Value_Type', 'Data_Value_Alt', 'Data_Value_Footnote_Symbol', 
                'Data_Value_Footnote', 'Total', 'Age(years)', 'Education',
                'Gender', 'Income', 'Race/Ethnicity', 'ClassID', 
                'TopicID', 'QuestionID', 'DataValueTypeID', 'LocationID']
data.drop(cols_to_drop, axis = 1, inplace = True)

# remove 'Guam' ,'Puerto Rico', 'Virgin Islands'
data = data[data.LocationAbbr != 'VI']
data = data[data.LocationAbbr != 'PR']
data = data[data.LocationAbbr != 'GU']


# In[6]:

list(data.columns)


# # Preview the Dataset

# In[7]:

data.head(n=20)


# # Grab 2016 Data

# In[8]:

data_2016 = data.query('YearStart == "2016"')

data_2016.head(n=5)


# # Visualizing Obesity by Location

# In[9]:

def get_table_from_query(df, query):
    filtered_data = df.query(query)
    filtered_data = filtered_data.query('LocationAbbr != "US"')
    mean_over_years = filtered_data.groupby(['LocationAbbr', 'Question', 'StratificationCategoryId1', 'StratificationID1'])['Data_Value'].mean()
    filtered_data.drop_duplicates(['LocationAbbr', 'Question', 'StratificationCategoryId1', 'StratificationID1'], keep='last', inplace=True)
    filtered_data.set_index(['LocationAbbr', 'Question', 'StratificationCategoryId1', 'StratificationID1'], inplace=True)
    filtered_data['Data_Value'] = mean_over_years
    filtered_data.reset_index(inplace=True)
    return filtered_data

def get_total_incidence_table(df, question):
    return get_table_from_query(df, 'StratificationCategory1 == "Total" & Question == "%s"' % (question,))


# In[10]:

obesity_data_by_state = get_total_incidence_table(data, "% Obese Adults")
obesity_data_by_state.head(n=5)


# In[11]:

# https://plot.ly/python/matplotlib-colorscales/
# use matplotlib style colorscales
magma_cmap = cm.get_cmap('magma')
viridis_cmap = cm.get_cmap('viridis')

viridis_rgb = []
magma_rgb = []
norm = colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
       k = colors.colorConverter.to_rgb(magma_cmap(norm(i)))
       magma_rgb.append(k)

for i in range(0, 255):
       k = colors.colorConverter.to_rgb(viridis_cmap(norm(i)))
       viridis_rgb.append(k)
    
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

magma = matplotlib_to_plotly(magma_cmap, 255)
viridis = matplotlib_to_plotly(viridis_cmap, 255)


# In[12]:

lat_long_regex = "\((-{0,1}[0-9]+.{0,1}[0-9]*),\s*(-{0,1}[0-9]+.{0,1}[0-9]*)\)"
def extract_lat_long(str):
    lat_long_match = match(lat_long_regex, str)
    return lat_long_match.groups()

def extract_lat(str):
    return float(extract_lat_long(str)[0])

def extract_long(str):
    return float(extract_lat_long(str)[1])

extract_lats = np.vectorize(extract_lat)
extract_longs = np.vectorize(extract_long)


# In[13]:

# See Plotly Support: https://plot.ly/pandas/scatter-plots-on-maps/

def plot_basic(df, plot_title, bar_title, filename, colorscheme=magma):
    # Extract relevant data from dataframe
    locs = df["GeoLocation"]
    state_names = df["LocationDesc"] + ' (' +  df["Data_Value"].astype('str') + '%)'
    values = pd.to_numeric(df["Data_Value"])
    max_value = values.max()
    min_value = values.min()
    
    # Create a heat-map scale w/ colors for percentages
    state_marker = {'size': 8, 
                    'opacity': 0.8, 
                    'symbol': 'square', 
                    'line': {'width': 1, 
                             'color': 'rgba(102, 102, 102)'}, 
                    'cmin': min_value,
                    'cmax': max_value, 
                    'colorbar': {'title': bar_title}}
    state_bar = {
        'title': bar_title
    }
    plot_data = {'type': 'choropleth', 
                 'reversescale': True, 
                 'colorscale': colorscheme,
                 'locationmode': 'USA-states', 
                 'locations': df["LocationAbbr"],
                 'z': values,
                 'text': state_names, 
                 'mode': 'markers', 
                 'marker': state_marker,
                 'colorbar': state_bar}
        
    plot_layout = {'title': '<b>%s</b><br>(hover for state name and value)' % (plot_title,),
                   'colorbar': True, 
                   'width': 1400,
                   'height': 800,
                   'geo': {'scope': 'usa', 
                           'projection': {'type': 'albers usa'}, 
                           'showland': True, 
                           'landcolor': "rgb(250, 250, 250)",
                           'subunitcolor': "rgb(217, 217, 217)",
                           'countrycolor': "rgb(217, 217, 217)", 
                           'countrywidth': 0.5, 
                           'subunitwidth': 0.5}
                  }

    fig = dict(data=[plot_data], layout=plot_layout)
    return fig
  
# direct to a plot.ly page with the plot
def show_on_web(fig):
    url = py.plot(fig, validate=False, filename=filename)

# show the plot
def show_on_notebook(fig):
    py.image.ishow(fig)

# save plot to <filename>.png
def save_plot(fig, filename):
    filename += '.png'
    py.image.save_as(fig, filename=filename)


# In[14]:

fig = plot_basic(obesity_data_by_state, 
          'Percentage of Obese Adults Population', 
          '% Obese Adults', 
          'd3-obesity')
show_on_notebook(fig)


# # Visualizing Overweight Incidence (inc. Obesity) by Location

# In[15]:

# Use grouping to get the actual number of overweight adults by state
overwieght_obesity_data = get_table_from_query(data, 'StratificationCategory1 == "Total" & (Question == "% Overweight Adults" | Question == "% Obese Adults")')
overweight_obesity_sum_by_state = overwieght_obesity_data.groupby(['LocationAbbr'])['Data_Value'].sum()


# In[16]:

# Update the overweight adults table to use the actual number of overweight adults
overweight_data_by_state = get_total_incidence_table(data, "% Overweight Adults")
overweight_data_by_state.set_index('LocationAbbr', inplace=True)
overweight_data_by_state['LocationAbbr'] = overweight_data_by_state.index
overweight_data_by_state['Data_Value'] = overweight_obesity_sum_by_state
overweight_data_by_state.head(n=5)


# In[17]:

fig = plot_basic(overweight_data_by_state, 
           'Overweight and Obese Adults by Location', 
           '% Overwieght Adults', 
           'd3-overweight')
show_on_notebook(fig)
save_plot(fig, 'd3-overweight')


# # Visualizing Inactivity by Location

# In[18]:

inactivity_data_by_state = get_total_incidence_table(data, "% No Phys. Activity")
inactivity_data_by_state.head(n=5)


# In[19]:

fig = plot_basic(inactivity_data_by_state, 
           'Inactivity by Location', 
           '% Adults w/o Phys. Activity', 
           'd3-inactive', viridis)
                
show_on_notebook(fig)
save_plot(fig, 'd3-inactive')


# # Visualizing Vigerous Aerobic Activity by Location

# In[20]:

vig_aerobic_by_state = get_total_incidence_table(data, "% Vigerous Aerobic")
inactivity_data_by_state.head(n=5)


# In[21]:

fig = plot_basic(inactivity_data_by_state, 
           'Vigerous Aerobic Activity by Location', 
           '% Adults w/ Vigerous Aerobic Activity', 
           'd3-vigerous', viridis)
                
show_on_notebook(fig)
save_plot(fig, 'd3-vigerous')


# # Visualizing Vegetable Malnutrition by Location

# In[22]:

veg_data_by_state = get_total_incidence_table(data, "% < 1 Veg / Day")
veg_data_by_state.head(n=5)


# In[23]:

fig = plot_basic(veg_data_by_state, 
           'Vegetable Malnutrition by Location', 
           '% Adults < 1 Veg / Day', 
           'd3-veg', viridis)
show_on_notebook(fig)


# # Visualizing Fruit Malnutrition by Location

# In[24]:

fruit_data_by_state = get_total_incidence_table(data, "% < 1 Fruit / Day")
fruit_data_by_state.head(n=5)
locations = fruit_data_by_state['LocationAbbr'][fruit_data_by_state.LocationAbbr != 'DC']
location_list = list(locations)


# In[25]:

fig = plot_basic(fruit_data_by_state, 
           'Fruit Malnutrition by Location', 
           '% Adults < 1 Fruit / Day', 
           'd3-fruit', viridis)
show_on_notebook(fig)


# # Plotting Correlation between Mean (over Years) Fruit Malnutrition and Obesity

# In[26]:

# X-axis: avg % obesity
# Y-axis: avg % of people consuming less than 1 fruit / day
plt.scatter(obesity_data_by_state.sort_values(by='LocationAbbr', axis=0)['Data_Value'].tolist(),
           fruit_data_by_state.sort_values(by='LocationAbbr', axis=0)['Data_Value'].tolist(),
           marker='^',
           color='g')
plt.xlabel('% Obesity')
plt.ylabel('% of people consuming less than 1 Fruit per Day ')
plt.show()


# # Plotting Correlation between Mean (over Years) Vigerous Aerobic Activity and Obesity

# In[27]:

# X-axis: avg % obesity
# Y-axis: avg % of people performing vigerous aerobic activity
plt.scatter(obesity_data_by_state.sort_values(by='LocationAbbr', axis=0)['Data_Value'].tolist(),
           vig_aerobic_by_state.sort_values(by='LocationAbbr', axis=0)['Data_Value'].tolist(),
           marker='s',
           color='b')
plt.xlabel('% Obesity')
plt.ylabel('% of people performing Vigerous Aerobic Activity')
plt.show()


# # Read Socioeconomic Risk Factors data

# In[28]:

# https://en.wikipedia.org/wiki/List_of_U.S._states_by_educational_attainment
edu_rates_data = pd.read_csv('educationrates.csv')
edu_rates_data.head(n=5)


# In[29]:

# https://en.wikipedia.org/wiki/Demography_of_the_United_States
demo_data = pd.read_csv('demography.csv')
demo_data.head(n=5)


# In[30]:

# https://en.wikipedia.org/wiki/List_of_U.S._states_by_income
income_data = pd.read_csv('medianincome.csv')
income_data.head(n=5)


# In[31]:

# Join tables to get risk factor data
risk_factor_data = edu_rates_data.set_index('State').join(demo_data.set_index('State or territory')).join(income_data.set_index('State'))
risk_factor_data.head(n=5)


# # Socioeconomic Risk Factors - Normalization

# In[32]:

risk_factor_data_norm = pd.DataFrame(MinMaxScaler().fit_transform(risk_factor_data))
risk_factor_data_norm.index = risk_factor_data.index
risk_factor_data_norm.columns = risk_factor_data.columns
risk_factor_data_norm.head(n=5)


# # Socioeconomic Risk Factors - K-Means

# In[33]:

# Find ideal number of clusters
sse = []
k = []

for i in range(1, 25):
    kmeans = KMeans(n_clusters=i).fit(risk_factor_data_norm)
    k.append(i)
    sse.append(kmeans.inertia_)

plt.plot(k, sse)
plt.ylabel('SSE')
plt.xlabel('K')
plt.show()


# In[34]:

# Perform K-means to find clusters
location_series = pd.Series(location_list)

risk_factor_kmeans = KMeans(n_clusters=5).fit(risk_factor_data_norm)
risk_factor_data_w_clusters = risk_factor_data.copy()
risk_factor_data_w_clusters['Cluster'] = risk_factor_kmeans.labels_
risk_factor_data_w_clusters['State_code'] = location_series.values
risk_factor_data_w_clusters.head(n=5)


# In[35]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(risk_factor_data_norm)
pca_2d = pca.transform(risk_factor_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=risk_factor_kmeans.labels_, marker='o')
plt.show()


# In[36]:

cluster1_risk_factor = risk_factor_data_w_clusters.loc[risk_factor_data_w_clusters['Cluster'] == 0]
cluster2_risk_factor = risk_factor_data_w_clusters.loc[risk_factor_data_w_clusters['Cluster'] == 1]
cluster3_risk_factor = risk_factor_data_w_clusters.loc[risk_factor_data_w_clusters['Cluster'] == 2]
cluster4_risk_factor = risk_factor_data_w_clusters.loc[risk_factor_data_w_clusters['Cluster'] == 3]
cluster5_risk_factor = risk_factor_data_w_clusters.loc[risk_factor_data_w_clusters['Cluster'] == 4]


# In[37]:

choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = risk_factor_data_w_clusters['State_code'], 
                   z=risk_factor_data_w_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = risk_factor_data_w_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
                  )]
layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("Socioeconomic Risk Factor Clustering (K-Means)",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # Socioeconomic Risk Factors - DBSCAN

# In[38]:

# Perform DBSCAN to find clusters
risk_factor_db = DBSCAN(eps=0.6, min_samples=6).fit(risk_factor_data_norm)
risk_factor_data_w_DB_clusters = risk_factor_data.copy()
risk_factor_data_w_DB_clusters['Cluster'] = risk_factor_db.labels_
risk_factor_data_w_DB_clusters['State_code'] = location_series.values
risk_factor_data_w_DB_clusters.head(n=5)


# In[39]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(risk_factor_data_norm)
pca_2d = pca.transform(risk_factor_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=risk_factor_db.labels_, marker='o')
plt.show()


# In[40]:

#plotting the socio economic factors clustered using DBSCAN
choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = risk_factor_data_w_DB_clusters['State_code'], 
                   z=risk_factor_data_w_DB_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = risk_factor_data_w_DB_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
#                    colorbar=dict(title=" cluster ")
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("Socioeconomic Risk Factor Clustering (DBSCAN)",),
              colorbar= True, 
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # Socioeconomic Risk Factors - Agglomerative

# In[92]:

# plot dendrogram to find optimum number of clusters
z = linkage(risk_factor_data_norm, 'ward')
plt.figure(figsize=(10,5))
dendrogram(z, leaf_rotation=90., leaf_font_size=10., show_contracted=True, labels = risk_factor_data_norm.index)
plt.title("Dendrogram For Socioeconomic Risk Factor")
plt.xlabel('State')
plt.ylabel('Distance')
plt.axhline(y = 1.75)
plt.axhline(y = 2.75) 
plt.show()


# # Aggregate the % of obese and % of overweight into a single table

# In[42]:

# Extract obesity values
obesity_values = pd.DataFrame(obesity_data_by_state[obesity_data_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
obesity_values.columns = ['% Obesity']
obesity_values.head(n=5)


# In[43]:

# Extract overweight values
oweight_values = pd.DataFrame(overweight_data_by_state[overweight_data_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
oweight_values.columns = ['% Overweight']
oweight_values.head(n=5)


# In[44]:

# Join the obesity and overweight tables together
weight_data = oweight_values.join(obesity_values)
weight_data.head(n=5)


# # Obesity & Overweight - Normalization

# In[45]:

weight_data_norm = pd.DataFrame(MinMaxScaler().fit_transform(weight_data))
weight_data_norm.index = weight_data.index
weight_data_norm.columns = weight_data.columns
weight_data_norm.head(n=5)


# # Obesity & Overweight - K-Means

# In[46]:

# Find ideal number of clusters
# choose number of cluster(k) = 3, based on the elbow method
sse = []
k = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i).fit(weight_data_norm)
    k.append(i)
    sse.append(kmeans.inertia_)

plt.plot(k, sse)
plt.ylabel('SSE')
plt.xlabel('K')
plt.show()


# In[47]:

# Perform K-means to find clusters
weight_kmeans = KMeans(n_clusters=3).fit(weight_data_norm)
weight_data_w_clusters = weight_data.copy()
weight_data_w_clusters['Cluster'] = weight_kmeans.labels_
weight_data_w_clusters['State_code'] = location_series.values
weight_data_w_clusters.head(n=5)


# In[48]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(weight_data_norm)
pca_2d = pca.transform(weight_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=weight_kmeans.labels_, marker='o')
plt.show()


# In[49]:

#plotting the Obesity Statistics clustering using K-Means
choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = weight_data_w_clusters['State_code'], 
                   z=weight_data_w_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = weight_data_w_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("Obesity Statistics clustering using K-Means",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # Obesity & Overweight - DBSCAN

# In[50]:

# Perform DBSCAN to find clusters
weight_db = DBSCAN(eps=0.1, min_samples=5).fit(weight_data_norm) # TODO: Actually tune the DBSCAN parameters from defaults
weight_w_DB_clusters = weight_data.copy()
weight_w_DB_clusters['Cluster'] = weight_db.labels_
weight_w_DB_clusters['State_code'] = location_series.values
weight_w_DB_clusters.head(n=5)


# In[51]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(weight_data_norm)
pca_2d = pca.transform(weight_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=weight_db.labels_, marker='o')
plt.show()


# In[52]:

#plotting the Obesity Statistics clustering using DBSCAN
choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = weight_w_DB_clusters['State_code'], 
                   z=weight_w_DB_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = weight_w_DB_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
#                    colorbar=dict(title=" cluster ")
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("Obesity Statistics clustering using DBSCAN",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # Aggregate CDC Health Behaviour into a single table

# In[53]:

# Extract inactivity values
inact_values = pd.DataFrame(inactivity_data_by_state[inactivity_data_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
inact_values.columns = ['% No Physical Activity']
inact_values.head(n=5)


# In[54]:

# Extract vigerous aerobic activity values
vigaer_values = pd.DataFrame(vig_aerobic_by_state[vig_aerobic_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
vigaer_values.columns = ['% Vigerous Aerobic']
vigaer_values.head(n=5)


# In[55]:

# Extract vegetable malnutriton values
veg_values = pd.DataFrame(veg_data_by_state[veg_data_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
veg_values.columns = ['% < 1 Veg / Day']
veg_values.head(n=5)


# In[56]:

# Extract fruit malnutrition values
fruit_values = pd.DataFrame(fruit_data_by_state[fruit_data_by_state.LocationAbbr != 'DC'].set_index('LocationDesc')['Data_Value'])
fruit_values.columns = ['% < 1 Fruit / Day']
fruit_values.head(n=5)


# In[57]:

# Join the individual tables together
cdc_risk_data = inact_values.join(vigaer_values.join(veg_values.join(fruit_values)))
cdc_risk_data.head(n=5)


# # CDC Health Behaviour - Normalization

# In[58]:

cdc_risk_data_norm = pd.DataFrame(MinMaxScaler().fit_transform(cdc_risk_data))
cdc_risk_data_norm.index = cdc_risk_data.index
cdc_risk_data_norm.columns = cdc_risk_data.columns
cdc_risk_data_norm.head(n=5)


# # CDC Health Behaviour - K-Means

# In[59]:

# Find ideal number of clusters
# k = 3
sse = []
k = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(cdc_risk_data_norm)
    k.append(i)
    sse.append(kmeans.inertia_)

plt.plot(k, sse)
plt.ylabel('SSE')
plt.xlabel('K')
plt.show()


# In[60]:

# Perform K-means to find clusters
cdc_risk_factors_kmeans = KMeans(n_clusters=3).fit(cdc_risk_data_norm)
cdc_risk_factors_w_clusters = cdc_risk_data.copy()
cdc_risk_factors_w_clusters['Cluster'] = cdc_risk_factors_kmeans.labels_
cdc_risk_factors_w_clusters['State_code'] = location_series.values
cdc_risk_factors_w_clusters.head(n=5)


# In[61]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(cdc_risk_data_norm)
pca_2d = pca.transform(cdc_risk_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=cdc_risk_factors_kmeans.labels_, marker='o')
plt.show()


# In[62]:

#plotting the CDC Risk Factor clustering using K-Means
choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = cdc_risk_factors_w_clusters['State_code'], 
                   z=cdc_risk_factors_w_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = cdc_risk_factors_w_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
#                    colorbar=dict(title=" cluster ")
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("CDC Health Behavior Clustering (K-Means)",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # CDC Health Behaviour - DBSCAN

# In[63]:

# Perform DBSCAN to find clusters
cdc_risk_factor_db = DBSCAN(eps=0.2, min_samples=3).fit(cdc_risk_data_norm) # TODO: Actually tune the DBSCAN parameters from defaults
cdc_risk_factor_w_DB_clusters = cdc_risk_data.copy()
cdc_risk_factor_w_DB_clusters['Cluster'] = cdc_risk_factor_db.labels_
cdc_risk_factor_w_DB_clusters['State_code'] = location_series.values
cdc_risk_factor_w_DB_clusters.head(n=5)


# In[64]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(cdc_risk_data_norm)
pca_2d = pca.transform(cdc_risk_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=cdc_risk_factor_db.labels_, marker='o')
plt.show()


# In[65]:

#plotting the CDC Risk Factor clustering using DBSCAN
choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = cdc_risk_factor_w_DB_clusters['State_code'], 
                   z=cdc_risk_factor_w_DB_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = cdc_risk_factor_w_DB_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
#                    colorbar=dict(title=" cluster ")
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("CDC Health Behavior Clustering (DBSCAN)",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # CDC Health Behaviour - Agglomerative

# In[66]:

# plot dendrogram to find optimum number of clusters
z = linkage(cdc_risk_data_norm, 'ward')
plt.figure(figsize=(10,5))
dendrogram(z, leaf_rotation=90., leaf_font_size=10., show_contracted=True, labels = cdc_risk_data_norm.index)
plt.title("Dendrogram For CDC Health Behavior")
plt.xlabel('State')
plt.ylabel('Distance')
plt.axhline(y = 1.75)
plt.axhline(y = 2.75) 
plt.show()


# # Combine Obesity & Risk Factors for CDC Health Behavior and outcomes

# In[67]:

# Join the individual tables together
outcome_data = oweight_values.join(obesity_values.join(inact_values.join(vigaer_values.join(veg_values.join(fruit_values)))))
outcome_data.head(n=5)


# # CDC Health Behavior and outcomes - Normalization

# In[68]:

outcome_data_norm = pd.DataFrame(MinMaxScaler().fit_transform(outcome_data))
outcome_data_norm.index = outcome_data.index
outcome_data_norm.columns = outcome_data.columns
outcome_data_norm.head(n=5)


# # CDC Health Behavior and outcomes - K-Means

# In[69]:

# Find ideal number of clusters
# k = 3
sse = []
k = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(outcome_data_norm)
    k.append(i)
    sse.append(kmeans.inertia_)

plt.plot(k, sse)
plt.ylabel('SSE')
plt.xlabel('K')
plt.show()


# In[70]:

# Perform K-means to find clusters
outcome_kmeans = KMeans(n_clusters=3).fit(outcome_data_norm)
outcome_data_w_clusters = outcome_data.copy()
outcome_data_w_clusters['Cluster'] = outcome_kmeans.labels_
outcome_data_w_clusters['State_code'] = location_series.values
outcome_data_w_clusters.head(n=5)


# In[71]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(outcome_data_norm)
pca_2d = pca.transform(outcome_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=outcome_kmeans.labels_, marker='o')
plt.show()


# In[72]:

choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = outcome_data_w_clusters['State_code'], 
                   z=outcome_data_w_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = outcome_data_w_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("CDC Health Behavior Outcomes Clustering (K-Means)",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # CDC Health Behavior and outcomes - DBSCAN

# In[73]:

# Perform DBSCAN to find clusters
outcome_db = DBSCAN(eps=0.6, min_samples=2).fit(risk_factor_data_norm) # TODO: Actually tune the DBSCAN parameters from defaults
outcome_data_w_DB_clusters = outcome_data.copy()
outcome_data_w_DB_clusters['Cluster'] = outcome_db.labels_
outcome_data_w_DB_clusters['State_code'] = location_series.values
outcome_data_w_DB_clusters.head(n=50)


# In[74]:

# Perform PCA to make 2D risk factor plot
pca = PCA(n_components=2).fit(outcome_data_norm)
pca_2d = pca.transform(outcome_data_norm).transpose()
pca_2d

plt.scatter(pca_2d[0], pca_2d[1], c=outcome_db.labels_, marker='o')
plt.show()


# In[75]:

choropleth = [dict(type='choropleth', 
                   autocolorscale=False, 
                   locations = outcome_data_w_DB_clusters['State_code'], 
                   z=outcome_data_w_DB_clusters['Cluster'], 
                   locationmode='USA-states', 
                   text = outcome_data_w_DB_clusters['State_code'], 
                   colorscale = 'custom-colorscale', 
                   showscale=False
#                    colorbar=dict(title=" cluster ")
                  )]

layout = dict(title='<b>%s</b><br>(hover for state name and value)' % ("CDC Health Behavior Outcomes Clustering (DBSCAN)",),
              width= 1400,
              height= 800,
              geo = dict(scope="usa", 
                         projection=dict(type="albers usa"), 
                         showland = True, 
                         landcolor = "rgb(250, 250, 250)",
                         subunitcolor = "rgb(217, 217, 217)",
                         countrycolor = "rgb(217, 217, 217)", 
                         countrywidth= 0.5, 
                         subunitwidth= 0.5))

fig = dict(data=choropleth, layout=layout)
show_on_notebook(fig)


# # CDC Health Behavior and outcomes - Agglomerative

# In[76]:

# plot dendrogram to find optimum number of clusters
z = linkage(outcome_data_norm, 'ward')
plt.figure(figsize=(10,5))
dendrogram(z, leaf_rotation=90., leaf_font_size=10., show_contracted=True, labels = outcome_data_norm.index)
plt.title("Dendrogram For Combined CDC Data")
plt.xlabel('State')
plt.ylabel('Distance')
plt.axhline(y = 1.75)
plt.axhline(y = 2.75) 
plt.show()


# In[ ]:



