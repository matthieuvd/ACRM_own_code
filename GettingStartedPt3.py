# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import squarify

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# to ignore all warnings temporarily
import warnings
warnings.filterwarnings("ignore")



## 1. IMPORT DATA
# load in all data
customers = pd.read_csv("./data/customers.csv")
geolocation = pd.read_csv("./data/geolocation.csv")
order_items = pd.read_csv("./data/order_items.csv")
order_payments = pd.read_csv("./data/order_payments.csv")
order_reviews = pd.read_csv("./data/order_reviews.csv")
orders = pd.read_csv("./data/orders.csv")
products = pd.read_csv("./data/products.csv")
sellers = pd.read_csv("./data/sellers.csv")

customers.drop(columns=['customer_unique_id'], inplace=True)
customers.head()
#df.to_csv('./data/customers.csv', index=False)
## 2. SPECIAL ZIPCODES IN SELLERS DATASET
# show
#print(sellers.head())

# data types
sellers.dtypes

# type cast to string
sellers["zip_code"] = sellers["zip_code"].astype(str)

# check
#print(sellers.dtypes)

# check for missing values
sellers.isnull().sum()

# let's read in the zip code data from BPOST
bpost = pd.read_excel("./Data/BPOST_Data.xlsx")
# type cast Postcode column to string
bpost["Postcode"] = bpost["Postcode"].astype(str)
# show
#print(bpost.head())

# check data types
#print(bpost.dtypes)

# check for missing values in BPOST dataset
#print(bpost.isnull().sum())

# merge using left_on and right_on
merged_df = sellers.merge(bpost, left_on='zip_code', right_on='Postcode', how='left')

# show
#print(merged_df.head())

# drop original city_name and Postcode column
merged_df.drop(["city_name", "Postcode"], axis=1, inplace=True)
# rename 'Plaatsnaam' (from BPOST data set) to 'city_name'
merged_df.rename(columns={"Plaatsnaam": "city_name"}, inplace=True)

# check for missing values again
#print(merged_df.isnull().sum())

# still 7 missing values for city_name
#print(merged_df[merged_df["city_name"].isnull()])

# replace '1071' by '1070' in column zip_code in original sellers dataset
sellers["zip_code"] = sellers["zip_code"].replace("1051", "1050") \
                                            .replace("1052", "1050") \
                                            .replace("1071", "1070")

# check if the replacement worked
#print("Do we find 1051 in zip_code column:", sellers["zip_code"].isin(['1051']).any())
#print("Do we find 1052 in zip_code column:",sellers["zip_code"].isin(['1052']).any())
#print("Do we find 1071 in zip_code column:",sellers["zip_code"].isin(['1071']).any())


## 3. PLOTTING
# count how many times a customer_id appears in the orders dataframe
count_df = orders['customer_id'].value_counts().reset_index()
# count how many times we observe each order count
count_count_df = count_df.groupby('count').count().reset_index()
# rename columns
count_count_df.rename(columns={"count": "orders_placed", "customer_id": "count"}, inplace=True)
# show
#print(count_count_df)

# visualize the number of purchases per customer using a seaborn bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=count_count_df['orders_placed'], y=count_count_df['count'], hue=count_count_df['orders_placed'], palette="RdPu_r", legend=False)
# add data labels above each bar
for i in range(len(count_count_df)):
    count = count_count_df.iloc[i]
    count_count_id = count_count_df.iloc[i]['count']
    plt.text(i, count_count_id+1, count_count_id, ha='center', va='bottom')
# add title and axis labels
plt.title('Number of Purchases per Customer')
plt.xlabel('Orders Placed')
plt.ylabel('Number of Customers')
#plt.show()

# show customers dataframe
#print(customers.head())

# check missing values
#print(customers.isnull().sum())

# function to retrieve province from zip code
def zip_code_province(zip_code):
    # initialize province
    province = None
    # retrieve first two characters of zip_code
    characters = zip_code[:2]
    # assign province
    if characters in ['10', '11', '12']:
        province = 'Brussel'
    elif characters in ['13', '14']:
        province = 'Waals Brabant'
    elif characters in ['15', '16', '17', '18', '19', '30', '31', '32', '33', '34']:
        province = 'Vlaams Brabant'
    elif characters in ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29']:
        province = 'Antwerpen'
    elif characters in ['35', '36', '37', '38', '39']:
        province = 'Limburg'
    elif characters in ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49']:
        province = 'Luik'
    elif characters in ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59']:
        province = 'Namen'
    elif characters in ['60', '61', '62', '63', '64', '65', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79']:
        province = 'Henegouwen'
    elif characters in ['66', '67', '68', '69']:
        province = 'Luxemburg'
    elif characters in ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']:
        province = 'West-Vlaanderen'
    elif characters in ['90', '91', '92', '93', '94', '95', '96', '97', '98', '99']:
        province = 'Oost-Vlaanderen'

    return province

# before we apply the function: zip_code should be a string in customers (because now incorrect float type)
customers['zip_code'] = customers['zip_code'].astype(int).astype(str)
# show
#print(customers.head())

# apply function on dataframe
customers['province_name'] = customers['zip_code'].apply(lambda x: zip_code_province(x))
# show
#print(customers.head())

# group by province and count number of customers
province_cust = customers.groupby('province_name')['customer_id'].count().sort_values(ascending=False).reset_index()
# rename columns (so that we can easily merge later)
province_cust.rename(columns={'province_name': 'NE_Name', 'customer_id':'count'}, inplace=True)
# show
#print(province_cust)

# read in the .shp file of provinces in geopandas
map_df = gpd.read_file('./Data/BELGIUM_-_Provinces.shp')
# show what the dataset looks like
#print(map_df)

# merge map_df and province_cust so that we have the number of customers per province (i.e., count)
merged = map_df.merge(province_cust, on='NE_Name', how='inner')
#print(merged)

# define colormap (starts from 0 to max value in count column)
cmap = colormaps['Blues']
norm = Normalize(vmin=0, vmax=merged['count'].max())

# plot the map with the clipped colormap (see GettingYouStartedPt2 for more info)
fig, ax = plt.subplots(figsize=(12, 8))
merged.plot(column='count', cmap=cmap, edgecolor='white', linewidth=0.3, ax=ax, norm=norm, legend=True)
ax.set_title('Customers per Province')

# show
#print(plt.show())

# define a list with all columns that we want to convert to the datetime data type
date_cols = ['order_purchase_timestamp', 'order_approved_at','order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
# let's convert them to datetime format
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], format='%Y-%m-%d %H:%M:%S')

# ASSUMPTION: we will only look at orders that actually got delivered
orders = orders[orders['order_status'] == 'delivered']

# let's create the variables described above
orders['purchased_approved'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() # in seconds
orders['approved_carrier'] = (orders['order_delivered_carrier_date'] - orders['order_approved_at']).dt.total_seconds() / 3600. # in hours
orders['carrier_delivered'] = (orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']).dt.total_seconds() / 3600. # in hours
orders['delivered_estimated'] = (orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']).dt.total_seconds() / 3600. # in hours
orders['purchased_delivered'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600. # in hours

# drop rows where approved_carrier is less than 0, i.e. where the order was approved after the carrier picked it up
orders = orders.drop(orders[orders['approved_carrier'] < 0].index)
# drop rows where carrier_delivered is less than 0, i.e. where the carrier delivered the order before picking it up
orders = orders.drop(orders[orders['carrier_delivered'] < 0].index)

# calculate the median of the purchased_approved column
median_purchase_approved = orders['purchased_approved'].median()
#print(median_purchase_approved) # in seconds
# replace missing values in order_approved_at by adding the calculated median to order_purchase_timestamp
orders['order_approved_at'].fillna(orders['order_purchase_timestamp'] + pd.Timedelta(seconds=median_purchase_approved), inplace=True)

# calculate the median of the approved_carrier column
median_approved_carrier = orders['approved_carrier'].median()
#print(median_approved_carrier) # in hours
# replace missing values in order_delivered_carrier_date by adding the calculated median to order_approved_at
orders['order_delivered_carrier_date'].fillna(orders['order_approved_at'] + pd.Timedelta(hours=median_approved_carrier), inplace=True)

# calculate the median of the carrier_delivered column
median_carrier_delivered = orders['carrier_delivered'].median()
#print(median_carrier_delivered) # in hours
# replace missing values in order_delivered_customer_date by adding the calculated median to order_delivered_carrier_date
orders['order_delivered_customer_date'].fillna(orders['order_delivered_carrier_date'] + pd.Timedelta(hours=median_carrier_delivered), inplace=True)

# calculate the median of the carrier_delivered column
median_carrier_delivered = orders['carrier_delivered'].median()
#print(median_carrier_delivered) # in hours
# replace missing values in order_delivered_customer_date by adding the calculated median to order_delivered_carrier_date
orders['order_delivered_customer_date'].fillna(orders['order_delivered_carrier_date'] + pd.Timedelta(hours=median_carrier_delivered), inplace=True)

# we have to re-run our variable creation code from above
orders['purchased_approved'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() # in seconds
orders['approved_carrier'] = (orders['order_delivered_carrier_date'] - orders['order_approved_at']).dt.total_seconds() / 3600. # in hours
orders['carrier_delivered'] = (orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']).dt.total_seconds() / 3600. # in hours
orders['delivered_estimated'] = (orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']).dt.total_seconds() / 3600. # in hours
orders['purchased_delivered'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600. # in hours

# we shouldn't have any missing values anymore
for col in orders.columns:
    missings = len(orders[col][orders[col].isnull()]) / float(len(orders))
    #print(col, missings)

# get the maximum date for each column
max_dates = orders[['order_purchase_timestamp', 'order_approved_at','order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']].max()
# get the maximum date from all columns
reference_date = max(max_dates)
# show
#print(reference_date)

# group by customer_id and get the most recent date of placing the order
recency_df = orders.groupby('customer_id')['order_purchase_timestamp'].max().reset_index()
# calculare recency (in days)
recency_df['recency']= recency_df['order_purchase_timestamp'].apply(lambda x: (reference_date - x).days)
# show
#print(recency_df)

# quick check if we don't have any duplicate customer_ids
orders['customer_id'].nunique() == len(recency_df)

# count the unique order_ids per customer_id
frequency_df = orders.groupby('customer_id').agg({"order_id": "nunique"}).reset_index()
# rename column
frequency_df.rename(columns={"order_id": "frequency"}, inplace=True)
# show
frequency_df

# caculate frequency of the frequency variable (i.e., count the number of customers in each frequency category)
frequency_count = frequency_df['frequency'].value_counts().reset_index()
# visualize the number of purchases per customer
plt.figure(figsize=(10, 6))
sns.barplot(x=frequency_count['frequency'], y=frequency_count['count'], hue=frequency_count['frequency'], palette="Blues_r", legend=False) # reverse palette by adding _r
# add data labels above each bar
for i in range(len(frequency_count)):
    count = frequency_count.iloc[i]
    count_count_id = frequency_count.iloc[i]['count']
    plt.text(i, count_count_id+1, count_count_id, ha='center', va='bottom')
# add title and axis labels
plt.title('Number of Purchases per Customer')
plt.xlabel('Orders Placed')
plt.ylabel('Number of Customers')
#plt.show()

# quick check: is the sum of the counts of each frequency level equal to the number of unique customers
frequency_count['count'].sum() == len(frequency_df)

# collapse the order_payments dataset so that we have one row for each unique order_id
# NOTE: this will drop the payment_sequential and payment_installments columns and the payment_type column will be the last one
grouped_df = order_payments.groupby(['order_id'])['payment_value'].sum().reset_index()
# rename column to 'total_amount_due'
grouped_df = grouped_df.rename(columns={'payment_value': 'total_amount_due'})
# left merge with orders dataset (we add the payment value column)
orders_with_price = pd.merge(orders, grouped_df, on='order_id', how='left')

# check for missing values
orders_with_price.isnull().sum()

# get shape of orders_with_price before removing missing values
#print(orders_with_price.shape)
# drop rows with missing values
orders_with_price = orders_with_price.dropna()
# get shape of orders_with_price after removing missing values
#print(orders_with_price.shape)

# check for missing values again
orders_with_price.isnull().sum()

# group by customer_id and sum the total_amount_due column
monetary_df = orders_with_price.groupby('customer_id').agg({"total_amount_due":"sum"}).reset_index()
# rename column
monetary_df = monetary_df.rename(columns={'total_amount_due': 'monetary_value'})
# show
#print(monetary_df)

# show the number of rows for each dataset
#print(recency_df.shape)
#print(frequency_df.shape)
#print(monetary_df.shape)

# we don't need the order_purchase_timestamp column anymore in the recency_df
recency_df = recency_df.drop('order_purchase_timestamp', axis=1)
# merge the three datasets
rfm_df = pd.merge(recency_df, frequency_df, on='customer_id', how='inner').merge(monetary_df, on='customer_id', how='inner')
# show shape
#print(rfm_df.shape)

# show first 5 rows
#print(rfm_df.head())

# get some quick descriptives
#print(rfm_df.describe())

# visualize the distribution of the three variables
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); sns.distplot(rfm_df['recency'])
plt.subplot(3, 1, 2); sns.distplot(rfm_df['frequency'])
plt.subplot(3, 1, 3); sns.distplot(rfm_df['monetary_value'])
#plt.show()

# use boxplots to visualize the distribution of the three variables
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# define variables to plot
variables = ["recency", "frequency", "monetary_value"]
# plot boxplot for each variable
for i in range(3):
    axes[i].boxplot(rfm_df[variables[i]])
    axes[i].set_title(variables[i])
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlabel(variables[i])
    axes[i].set_xticklabels('')
# adjust spacing between subplots
plt.tight_layout()
# show
#plt.show()

# apply discretization (i.e., create five categories using the qcut() function in pandas)
rfm_df["recency_score"]  = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm_df["frequency_score"]= pd.qcut(rfm_df['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm_df["monetary_score"] = pd.qcut(rfm_df['monetary_value'], 5, labels=[1, 2, 3, 4, 5])
# show
rfm_df.head()

# concatenate the scores to create the RFM score
rfm_df['rfm_score'] = rfm_df.apply(lambda x: str(x['recency_score']) + str(x['frequency_score']) + str(x['monetary_score']), axis=1)
# show
rfm_df.head()

# create assignment function
def assign_segment(rfm_score):

    segment = None

    if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
        segment = 'Champions'
    elif rfm_score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451', '442', '441', '431', '453', '433', '432', '423', '353', '352', '351', '342', '341', '333', '323']:
        segment = 'Potential Loyalist'
    elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
        segment = 'Loyal'
    elif rfm_score in ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
        segment = 'Promising'
    elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
        segment = 'New Customers'
    elif rfm_score in ['331', '321', '312', '221', '213', '231', '241', '251']:
        segment = 'About To Sleep'
    elif rfm_score in ['535', '534', '443', '434', '343', '334', '325', '324']:
        segment = 'Need Attention'
    elif rfm_score in ['155', '154', '144', '214', '215', '115', '114', '113']:
        segment = 'Cannot Lose Them'
    elif rfm_score in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '133', '125', '124']:
        segment = 'At Risk'
    elif rfm_score in ['332', '322', '233', '232', '223', '222', '132', '123', '122', '212', '211']:
        segment = 'Hibernating Customers'
    elif rfm_score in ['111', '112', '121', '131', '141', '151']:
        segment = 'Lost customers'

    return segment

# apply assignment function
rfm_df['segment'] = rfm_df['rfm_score'].apply(lambda x: assign_segment(x))
# show
rfm_df.head()

# check for missing values
rfm_df.isnull().sum()

# get number of unique values (this should be 11)
rfm_df['segment'].nunique()

# get the proportion of customers in each segment
segments_df = rfm_df['segment'].value_counts(normalize=True).reset_index().sort_values("proportion", ascending=False)

# visualize
plt.figure(figsize=(24,8))
per = sns.barplot(x=segments_df['proportion'], y=segments_df['segment'], palette="Blues_r", orient='h') # orient='h' to make horizontal
# add data labels next to each bar
for index, row in segments_df.iterrows():
    per.text(row['proportion']+0.0005, index, f'{row["proportion"]:.2%}', va='center')
# add title and axis labels
plt.title('Distribution of Segments')
plt.xlabel('Proportion')
plt.ylabel('Segment')
# show
#plt.show()

# get an overview for each segment
rfm_stats = rfm_df[["segment", "recency", "frequency", "monetary_value"]].groupby("segment").agg(['mean','median', 'min', 'max', 'count'])
# show
#print(rfm_stats)

# create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# violin plots for recency
sns.violinplot(x='segment', y='recency', data=rfm_df, ax=axes[0], palette='tab10')
axes[0].set_title("Distributions of Recency by Segment")
axes[0].set_xticks(axes[0].get_xticks(), axes[0].get_xticklabels(), rotation=90, ha='right')
# violin plots for frequency
sns.violinplot(x='segment', y='frequency', data=rfm_df, ax=axes[1], palette='tab10')
axes[1].set_title("Distributions of Frequency by Segment")
axes[1].set_xticks(axes[1].get_xticks(), axes[1].get_xticklabels(), rotation=90, ha='right')
axes[1].set_ylim(0, 6)
# violin plots for monetary value
sns.violinplot(x='segment', y='monetary_value', data=rfm_df, ax=axes[2], palette='tab10')
axes[2].set_title("Distributions of Monetary Value by Segment")
axes[2].set_xticks(axes[2].get_xticks(), axes[2].get_xticklabels(), rotation=90, ha='right')
axes[2].set_ylim(0, 1000)
# set labels and title
for ax in axes:
    ax.set_xlabel('Segment')
    ax.set_ylabel('Data Distribution')
# adjust spacing between subplots
plt.tight_layout()
# show
#plt.show()

# visualize proportions of each cluster in customer base
plt.figure(figsize=(24,8))
axis = squarify.plot(sizes=rfm_stats["recency"]["count"], label=rfm_stats.index, color=sns.color_palette("tab20", len(rfm_stats)),
                     text_kwargs={'fontsize': 16})
# set title (optional: remove axes)
axis.set_title("Proportion of each Customers Segment", fontsize=16)
#plt.axis('off')
#plt.show()

# only select rfm variables
rfm_kmeans_df = rfm_df[['recency', 'frequency', 'monetary_value']]
# show
#print(rfm_kmeans_df.head())

# initialize scaler
scaler = StandardScaler()
# fit
scaler.fit(rfm_kmeans_df[['recency', 'frequency', 'monetary_value']])
# transform and convert into a dataframe
rfm_kmeans_df[['recency', 'frequency', 'monetary_value']] = scaler.transform(rfm_kmeans_df[['recency', 'frequency', 'monetary_value']])
# show
#print(rfm_kmeans_df.head())
'''
# create an empty list to store the within-cluster sum of squares (WCSS) for different values of k
wcss = []
# create an empty list to store the silhouette scores
silhouette_scores = []
# iterate through different values of K (here: from 2 to 10) and compute WCSS and silhouette score for each iteration
for k in range(2, 11):
    # print current value of k
    print("k:", k)
    # initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k)
    # fit kmeans on the data
    kmeans.fit(rfm_kmeans_df)
    # get labels of our clustering
    labels = kmeans.labels_
    # append WCSS to list storing WCSS
    wcss.append(kmeans.inertia_)
    # calculate the silhouette score and append to list
    silhouette_avg = silhouette_score(rfm_kmeans_df, labels)
    silhouette_scores.append(silhouette_avg)

# create a subplot to display both the elbow curve and silhouette scores
plt.figure(figsize=(12, 5))
# plot the elbow curve
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
# plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# define Kmeans model with 5 clusters (random_state for reproducibility of results)
kmeans = KMeans(n_clusters=5, random_state=123)
# fit model
kmeans.fit(rfm_kmeans_df)

rfm_kmeans_df['cluster']= kmeans.labels_
rfm_kmeans_df.head()

overview_df = rfm_kmeans_df.groupby(['cluster']).agg({
            'recency'  : ['mean','median', 'min', 'max'],
            'frequency': ['mean','median', 'min', 'max'],
            'monetary_value' : ['mean','median', 'min', 'max', 'count']
        }).round(0)
# show
#print(overview_df)

# use unstandardized for easy interpretation
rfm_df['cluster'] = kmeans.labels_
overview_df2 = rfm_df.groupby(['cluster']).agg({
            'recency'  : ['mean','median', 'min', 'max'],
            'frequency': ['mean','median', 'min', 'max'],
            'monetary_value' : ['mean','median', 'min', 'max', 'count']
        }).round(0)
# show
#print(overview_df2)

# create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# violin plots for recency
sns.violinplot(x='cluster', y='recency', data=rfm_df, ax=axes[0], palette='tab10')
axes[0].set_title("Distributions of Recency by Cluster")
# violin plots for frequency
sns.violinplot(x='cluster', y='frequency', data=rfm_df, ax=axes[1], palette='tab10')
axes[1].set_title("Distributions of Frequency by Cluster")
# violin plots for monetary value
sns.violinplot(x='cluster', y='monetary_value', data=rfm_df, ax=axes[2], palette='Set1')
axes[2].set_title("Distributions of Monetary Value by Cluster")
axes[2].set_ylim(0, 4000)
# set labels and title
for ax in axes:
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Data Distribution')
# adjust spacing between subplots
plt.tight_layout()
# show
#plt.show()

# create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for cluster_value in rfm_df['cluster'].sort_values().unique():
    sns.kdeplot(rfm_df[rfm_df['cluster'] == cluster_value]['recency'], label=f'Cluster {cluster_value}', ax=axes[0], shade=True)
axes[0].set_title("Distribution of Recency by Customer Segment")
axes[0].legend()
for cluster_value in rfm_df['cluster'].sort_values().unique():
    sns.kdeplot(rfm_df[rfm_df['cluster'] == cluster_value]['frequency'], label=f'Cluster {cluster_value}', ax=axes[1], shade=True)
    if cluster_value == 2:
        axes[1].axvline(x=rfm_df[rfm_df['cluster'] == 2]['frequency'].unique(), label='Cluster 2', color='green') # only one value for cluster 2: freq=1
axes[1].set_title("Distribution of Frequency by Customer Segment")
axes[1].legend()
for cluster_value in rfm_df['cluster'].sort_values().unique():
    sns.kdeplot(rfm_df[rfm_df['cluster'] == cluster_value]['monetary_value'], label=f'Cluster {cluster_value}', ax=axes[2], shade=True)
axes[2].set_title("Distribution of Monetary Value by Customer Segment")
axes[2].set_xlim(0, 2000) # limit x-axis
axes[2].legend()
# set labels and title
for ax in axes:
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Data Distribution')
# adjust spacing between subplots
plt.tight_layout()
# show
#plt.show()

# get the proportion of customers in each segment
segments_df = rfm_df['cluster'].value_counts(normalize=True).reset_index().sort_values("proportion", ascending=False)
# type cast to make it categorical
segments_df['cluster'] = segments_df['cluster'].astype(str)
# show
#print(segments_df)

# visualize
plt.figure(figsize=(24,6))
bar = sns.barplot(x='proportion', y='cluster', data=segments_df, palette="Blues_r", orient='h')
# add data labels next to each bar
for index, row in segments_df.iterrows():
    bar.text(row['proportion']+0.0005, index, f'{row["proportion"]:.2%}', va='center')
# add title and axis labels
plt.title('Distribution of Clusters')
plt.xlabel('Proportion')
plt.ylabel('Segment')
# show
#plt.show()

# visualize the proportions of each cluster in you customer base you can use squairfy
# it literally makes a squery out of each cluster
plt.figure(figsize=(24, 8))
axis = squarify.plot(sizes=overview_df2["monetary_value"]["count"], label=overview_df2.index, color=sns.color_palette("tab10", len(overview_df2)), 
                     text_kwargs={'fontsize': 18, 'color':'white'})
# set title (optional: remove axes)
axis.set_title("Proportion of each Customers Segment", fontsize=16)
#plt.axis('off')
# show
#plt.show()
'''

# create a dataframe
order_reviews.head()

# NOTE: inner join between orders_with_price and order_reviews (i.e., we only select observations for which we have full data for both datasets)
# followed by left join with customers dataset to add the customer information
features_df = orders_with_price.merge(order_reviews, on="order_id", how="inner").merge(customers, on='customer_id', how='left')
# drop columns
features_df = features_df.drop(columns=['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'order_status',
                                        'review_id', 'language'], axis=1)
# show
features_df.head()

# missing values
features_df.isnull().sum()

# count number of characters in review and drop original column
features_df['nr_chars_review'] = features_df['review_comment_message'].apply(lambda x: 0 if pd.isnull(x) else len(x))
features_df = features_df.drop('review_comment_message', axis=1)
# count number of characters in review title and drop original column
features_df['nr_chars_title'] = features_df['review_comment_title'].apply(lambda x: 0 if pd.isnull(x) else len(x))
features_df = features_df.drop('review_comment_title', axis=1)

# calculate age from birthday (see previous notebook)
# define reference date
ref_date = orders['order_delivered_customer_date'].max()
print(ref_date)
# convert birthday to datetime
features_df['birthday'] = pd.to_datetime(features_df['birthday'], format='%Y-%m-%d')
# calculate the age of a customer
features_df['age'] = (ref_date - features_df['birthday']).dt.days // 365.25
# drop birthday column
features_df = features_df.drop('birthday', axis=1)

# convert to datetime
features_df['review_creation_date'] = pd.to_datetime(features_df['review_creation_date'], format='%Y-%m-%d %H:%M:%S')
features_df['review_answer_timestamp'] = pd.to_datetime(features_df['review_answer_timestamp'], format='%Y-%m-%d %H:%M:%S')
# calculate the time between sending the satisfaction survey and the answer
features_df['answer_time'] = (features_df['review_answer_timestamp'] - features_df['review_creation_date']).dt.total_seconds() / 86400. # in days
# drop original columns
features_df = features_df.drop(['review_creation_date', 'review_answer_timestamp'], axis=1)

# province and gender are categroical and should be converted to dummy variables
features_df = pd.get_dummies(features_df, columns=['gender'], drop_first=True, dtype=float)

# create dependent variable (binary)
features_df['review_score'] = features_df['review_score'].apply(lambda x: 1 if x >=3 else 0)

# drop identifier columns
features_df = features_df.drop(['order_id', 'customer_id'], axis=1)

#print(features_df.head())

# NOTE: we will use the following variables to predict the review score
independent_features = ['purchased_delivered', 'nr_chars_review', 'nr_chars_title', 'answer_time', 'gender_M', 'age']
# Independent Features (X)
X = features_df[independent_features]
# review_score is the target column / Dependent Variable (y)
y = features_df[['review_score']]

# scale features
# initialize scaler
from sklearn.preprocessing import MinMaxScaler
# instantiate scaler
scaler = StandardScaler()
# fit scaler on numerical features only
numerical_features = X.columns.drop(['gender_M'])
scaler.fit(X[numerical_features])
# apply tansformation
X[numerical_features] = scaler.transform(X[numerical_features])

# check result
#print(X.describe())

# import statsmodels package
import statsmodels.api as sm
# run multivariate logistic regression
X = sm.add_constant(X) # adding a constant: Y = beta0 + beta1*X1 + beta2*X2 + espilon instead of Y = beta1*X1 + beta2*X2 + epsilon
# fit model
model = sm.Logit(y, X).fit()
# show model summary
#print(model.summary())

# NOTE: sklearn implemenation gives the same results as statsmodels package
# If you want to check:
# Independent Features (X)
X = features_df[independent_features]
# review_score is the target column / Dependent Variable (y)
y = features_df[['review_score']]
# scaling
scaler = StandardScaler()
numerical_features = X.columns.drop(['gender_M'])
scaler.fit(X[numerical_features])
X[numerical_features] = scaler.transform(X[numerical_features])
# modeling
logistic_regression = LogisticRegression()
logistic_regression.fit(X, y)
lr_preds = logistic_regression.predict(X)
# get coefficients
#print(logistic_regression.coef_)

# get intercept
#print(logistic_regression.intercept_)