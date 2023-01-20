
#import the standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# the url of urban population
url ='https://api.worldbank.org/v2/en/indicator/SP.URB.TOTL.IN.ZS?downloadformat=excel'
# the years are stored in a column
year1 ='1972'
year2= '2020'

# the file is read using pandas
df = pd.read_excel(url, sheet_name='Data', skiprows=3)
df = df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'])
#extract the required data for clustering
df_cluster = df.loc[df.index, ['Country Name', year1, year2]]
print(df_cluster)


# check number of null vaues
df_cluster.isnull().sum()

# the null values are dropped
df_cluster = df_cluster.dropna()
print(df_cluster)

# the data is stored in array
x = df_cluster[[year1, year2]].values
print(x)

# x is normalised for clustering
min_val = np.min(x)
max_val = np.max(x)
x_scaled = (x-min_val) / (max_val-min_val)
print(x_scaled)


# this is the scatterplot of the original data using matplotlib
df_cluster.plot(year1, year2, kind='scatter')
plt.title('Scatterplot of urban population between 1972 and 2020')
plt.xlabel('Year 1')
plt.ylabel('Year 2')
plt.show()

# the best number of clusters is chosen using sum of squared error
sse = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x_scaled) # the normalised data is fit using KMeans method
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xlabel('clusters')
plt.ylabel('sse')
plt.show()

# according to the elbow, 3 is the best number for clustering
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(x_scaled)

# the centroid of the clusters are determined
center = kmeans.cluster_centers_
print(center)

# this is a plot of the clusters and centroids using matplotlib
plt.figure(figsize=(12,8))
plt.scatter(x_scaled[y_kmeans == 0, 0], x_scaled[y_kmeans == 0, 1], s = 50, c = 'purple',label = 'label 0')
plt.scatter(x_scaled[y_kmeans == 1, 0], x_scaled[y_kmeans == 1, 1], s = 50, c = 'orange',label = 'label 1')
plt.scatter(x_scaled[y_kmeans == 2, 0], x_scaled[y_kmeans == 2, 1], s = 50, c = 'green',label = 'label 2')
plt.scatter(center[:, 0], center[:,1], s = 100, c = 'red', label = 'Centroids')
plt.title('Scatterplot depicting the clusters and centroids of urban population in countries of the world from 1972 to 2020', fontsize=20)
plt.xlabel('Year 1', fontsize=15)
plt.ylabel('Year 2', fontsize=15)
plt.legend()
plt.show()


# the clusters are stored in a column called labels
df_cluster['labels'] = y_kmeans
df_label = df_cluster.loc[df_cluster['labels'] == 2] # the third cluster label is stored into df_label 
df_pie = df_label.iloc[:6, :] # the first 6 countries of the third cluster are used for analysis
print(df_pie)

df_label2 = df_cluster.loc[df_cluster['labels'] == 1] # the second cluster is stored in a variable
df_histogram = df_label2.iloc[:6, :]
print(df_histogram)

# a pie chart of the third cluster for the first 6 countries is plot using matplotlib
pie_data = df_pie['2020']
label = df_pie['Country Name']
title = 'Pie chart showing second cluster in Year 2020'
color = ['red', 'magenta', 'blue', 'indigo', 'green', 'yellow']
explode = (0, 0, 0, 0 , 0, 0.1)


plt.figure(figsize=(10,8))
plt.title(title, fontsize=20)
plt.pie(pie_data, explode = explode, labels=label, colors=color, autopct='%0.2f%%')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# this is a histogram plot
def histogram(datasets, title, color, alpha):
    """
    This function plots a histogram and it accepts the following as parameters:
    datasets: these are the columns which are to be plotted
    title: title of the plot
    color: color of the histograms
    alpha: this depicts the transparency of the plot
    """
    plt.figure(figsize = (12,8))
    
    for i in range(len(datasets)):
        plt.subplot(1,2,i+1).set_title(title[i])
        plt.hist(datasets[i], alpha=alpha[i], color=color[i])
    
    plt.show()
    
    return

# Arrays are created for the histogram 
datasets = [df_histogram['1972'], df_histogram['2020']]
title = ['Year 1972', 'Year 2020']
color = ['red', 'blue']
alpha = [0.3, 0.4]
# The arrays are passed as input parameters into the defined function and the output is displayed below

histogram(datasets, title, color, alpha)

# transpose the original data
df_transpose = df.transpose()
print(df_transpose)

# the columns of the transposed data are refined using the loc function
df_transpose.columns = df_transpose.iloc[0]
df_transpose_clean = df_transpose.iloc[1:,:]
print(df_transpose_clean)

# Nigeria is used for analysis and a dataframe is created for it
df_Nigeria = pd.DataFrame({'Year' : df_transpose_clean.index, 'Nigeria' : df_transpose_clean['Nigeria']})
df_Nigeria.reset_index(drop=True)


# the year column is converted to integer
df_Nigeria['Year'] = np.array(df_Nigeria['Year']).astype(int)


# this is a fitting function
def model_fitting(x, a, b, c, d):
    '''
    The function calculates the exponential function which accepts some parameters:
    x: these are the years of the data column
    a,b,c,d are constants
    
    '''
    return a*x**3 + b*x**2 + c*x + d


# the parameters are passed into the curve_fit method and the parameters and covariance are calculated 
param, covar = curve_fit(model_fitting, df_Nigeria['Year'], df_Nigeria['Nigeria'])
print(param)
print(covar)

# a range of values for years is created for prediction
years = np.arange(1961, 2031)
predictions = model_fitting(years, *param) # the parameters and years are passed into the model_fitting function and the predictions are calculated 

# this is a plot of urban population of Nigeria and the predictions for the next 10 years 
plt.figure(figsize=(12,10))
plt.plot(df_Nigeria["Year"], df_Nigeria["Nigeria"], label="Nigeria")
plt.plot(years, predictions, label="predictions")
plt.title('A Plot showing the predictions of urban population in Nigeria', fontsize=20)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Urban Population", fontsize=15)
plt.legend()
plt.show()

# Here the predictions for the next 10 years are put in a dataframe 
df_predictions = pd.DataFrame({'Year': years, 'Forecast': predictions})
df_ten_years_prediction = df_predictions.iloc[60:,:]
print(df_ten_years_prediction)




