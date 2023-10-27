# import packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import seaborn as sns
import itertools

orders = pd.read_csv("./Data/orders.csv")
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
# get descriptives
#print(orders.describe())
# create a new column month
orders['month_year'] = orders['order_purchase_timestamp'].dt.to_period('M')

# group the data by month and count the number of orders placed per month
monthly_sales = orders.groupby('month_year')['order_id'].count()

# convert the index (Month-Year) to strings for compatibility with plt.bar()
monthly_sales.index = monthly_sales.index.astype(str)
# create a bar plot to visualize the sales volume per month
plt.figure(figsize=(20, 6))

plt.bar(monthly_sales.index, monthly_sales)
plt.plot(monthly_sales.index, monthly_sales, color='orange', marker='x')

plt.title('Evolution of Placed Orders (per Month)')
plt.xlabel('Month')
plt.ylabel('Orders Placed')
plt.xticks(rotation=45)  # rotate x-axis labels for better readability
#plt.show()
# read in the order_payments dataset
order_payments = pd.read_csv("./Data/order_payments.csv")

# collapse the order_payments dataset so that we have one row for each unique order_id
# NOTE: this will drop the payment_sequential and payment_installments columns and the payment_type column will be the last one
grouped_df = order_payments.groupby(['order_id'])['payment_value'].sum().reset_index()

# rename column to 'total_amount_due'
grouped_df = grouped_df.rename(columns={'payment_value': 'total_amount_due'})

# left merge with orders dataset (we add the payment value column)
orders_with_price = pd.merge(orders, grouped_df, on='order_id', how='left')
# create a new column month
orders_with_price['month_year'] = orders_with_price['order_purchase_timestamp'].dt.to_period('M')

# group the data by month and count the number of orders placed per month
monthly_sales2 = orders_with_price.groupby('month_year')['total_amount_due'].sum()

# convert the index (Month-Year) to strings for compatibility with plt.bar()
monthly_sales2.index = monthly_sales2.index.astype(str)
# visualize sales for month (in monetary value)
plt.figure(figsize=(20, 6))

plt.bar(monthly_sales2.index, monthly_sales2)

plt.title('Evolution of Sales (per Month)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # rotate x-axis labels for better readability
#plt.show()
# calculate differences in days
orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_approved_at']).dt.total_seconds() / 86400
orders['estimated_delivery_time'] = (orders['order_estimated_delivery_date'] - orders['order_approved_at']).dt.total_seconds() / 86400
# let's pick a color palette (default)
# documentation: https://seaborn.pydata.org/tutorial/color_palettes.html
sns.color_palette()
# define color palette using itertools.cycle
palette = itertools.cycle(sns.color_palette())

# define data and labels
data = [orders['delivery_time'], orders['estimated_delivery_time']]
labels = ['Delivery time', 'Estimated delivery time']

# create a figure
plt.figure(figsize=(10, 4))

# loop through data and labels to create KDE plots with cycling colors
for data_series, label in zip(data, labels):
    current_palette_color = next(palette)
    sns.kdeplot(data_series, label=label, fill=True, common_norm=False, alpha=0.5, linewidth=0, color=current_palette_color)

plt.title("Delivery time in days")
plt.xlabel("Delivery Time")
plt.legend()
plt.show()

# load in reviews dataset
reviews = pd.read_csv("./data/order_reviews.csv")

# merge with orders dataset
orders = orders.merge(reviews, on='order_id', how='left')

# visualize limited delivery time by review score
sns.catplot(x="review_score", y="delivery_time", kind="box", hue="review_score",
            data=orders[orders["delivery_time"] < 60], height=4, aspect=1.5, palette="tab10")
plt.xlabel("Review Score")
plt.ylabel("Delivery Time")
plt.show()
# last date an order was placed
#print(orders['order_purchase_timestamp'].max())

# last date an order was delivered
#print(orders['order_delivered_customer_date'].max())
# load customers dataset
customers = pd.read_csv("./data/customers.csv")

# convert customer_id to datetime
customers['birthday'] = pd.to_datetime(customers['birthday'], format='%Y-%m-%d')

# show
#print(customers.head())
# we'll use the last date an order was delivered as the reference date
ref_date = orders['order_delivered_customer_date'].max()

# calculate the age of a customer
customers['age'] = (ref_date - customers['birthday']).dt.days // 365.25

# convert to integer
customers['age'] = customers['age'].astype(int)

# show
#print(customers.head())
# create a matplotlib figure and axis
plt.figure(figsize=(10, 4))

# get current axes (gca) so that you can draw on it directly
ax = plt.gca()

# plot histogram
ax.hist(customers['age'], bins=customers['age'].nunique(), edgecolor='white', color='darkgrey', density=True)

# plot KDE plot on the same axis
sns.kdeplot(data=customers['age'], fill=False, common_norm=False, color='black')

# add a vertical line to indicate the age of 18
plt.axvline(18, color='red', linestyle='--', linewidth=1.2)

# Add labels and a title
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")

# Show the plot
#plt.show()

#Geospatial data
# import geopandas
import geopandas as gpd

# to ignore all warnings temporarily
import warnings
warnings.filterwarnings("ignore")

# read in the .shp file in geopandas
map_df = gpd.read_file('./data/BELGIUM_-_Municipalities.shp')

#lowercase column names
map_df.columns = map_df.columns.str.lower()

# show what the dataset looks like
#print(map_df.head())

# plot the map
fig, ax = plt.subplots(figsize=(12, 8))
map_df.plot(ax=ax, color='black', edgecolor='white', linewidth=0.5)
ax.set_title('Municipalities in Belgium')
#plt.axis('off')  # Turn off the axis
#plt.show()

# load in customers dataset
customers = pd.read_csv("./Data/customers.csv")

# NOTE: zip_code should be string without decimals
customers['zip_code'] = customers['zip_code'].astype(int).astype(str)

# put city_name in lower-case
customers['city_name'] = customers['city_name'].str.lower()

# show
#print(customers.head())

# load the external dataset
zip_codes = pd.read_excel('./Data/zipcodes_municipality.xlsx')

# put city_name in lower-case
zip_codes['city_name'] = zip_codes['city_name'].str.lower()

# show
print(zip_codes.head())


