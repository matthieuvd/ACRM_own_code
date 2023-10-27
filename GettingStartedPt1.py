import numpy as np
import pandas as pd
import os


data_dir = "./Data"
dir_list = [obs for obs in os.listdir(data_dir) if ".DS" not in obs]
dir_list
#print(os.listdir(data_dir)) #alle databestandnamen
orders = pd.read_csv("./Data/orders.csv")
#print(orders.dtypes)
#1. Orders
orders[['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']].head()
#alle data die we van string naar tijd willen omzetten
date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], format='%Y-%m-%d %H:%M:%S')
#print(orders.head())
#print(orders.dtypes) #tijd data is nu juist en geen string meer
for col in orders.columns:
    missings = len(orders[col][orders[col].isnull()]) / float(len(orders))
    #print(col, missings)
#let's count the occurence of order_status values in rows where we have no missing values in any column
#print(orders.dropna()['order_status'].value_counts())
#in comparison, let's count the occurence of order_status values in rows that have at least one missing value in any column
#print(orders[orders.isna().any(axis=1)]['order_status'].value_counts())
# we only look at orders that actually got delivered
orders = orders[orders['order_status'] == 'delivered']
# however, we still have to deal with missing values
for col in orders.columns:
    missings = len(orders[col][orders[col].isna()])
    #print(col, missings)
# show the 23 problematic rows
missing_values_indexes = orders.index[orders.isna().any(axis=1)]
rows_with_missing_values = orders[orders.isna().any(axis=1)]
#print(rows_with_missing_values)

#Let's create some new variables to impute these missing values:
#purchased_approved: the seconds taken for an order to get approved after the customer purchases it
#approved_carrier: the hours taken for the order to go to the delivery carrier after it being approved
#carrier_delivered: the hours taken for the order to be delivered to the customer from the date it reaches the delivery carrier
#delivered_estimated: the hours difference between the estimated delivery date and the actual delivery date
#purchased_delivered: the hours taken for the order to be delivered to the customer from the date the customer made the purchase

# let's create the variables described above
orders['purchased_approved'] = (orders['order_approved_at'] - orders['order_purchase_timestamp']).dt.total_seconds() # in seconds
orders['approved_carrier'] = (orders['order_delivered_carrier_date'] - orders['order_approved_at']).dt.total_seconds() / 3600. # in hours
orders['carrier_delivered'] = (orders['order_delivered_customer_date'] - orders['order_delivered_carrier_date']).dt.total_seconds() / 3600. # in hours
orders['delivered_estimated'] = (orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']).dt.total_seconds() / 3600. # in hours
orders['purchased_delivered'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.total_seconds() / 3600. # in hours

# show first 5 rows
#print(orders.head())
# get some statistics regarding the newly created (numeric) variables
#print(orders.describe())
# drop rows where approved_carrier is less than 0, i.e. where the order was approved after the carrier picked it up
orders = orders.drop(orders[orders['approved_carrier'] < 0].index)
#print(len(orders))
# drop rows where carrier_delivered is less than 0, i.e. where the carrier delivered the order before picking it up
orders = orders.drop(orders[orders['carrier_delivered'] < 0].index)
#print(len(orders))
# check if we still have missing values
for col in orders.columns:
    missings = len(orders[col][orders[col].isna()])
    #print(col, missings)
# calculate the median of the purchased_approved column
median_purchase_approved = orders['purchased_approved'].median()
#print(median_purchase_approved) # in seconds

# replace missing values in order_approved_at by adding the calculated median to order_purchase_timestamp
orders['order_approved_at'].fillna(orders['order_purchase_timestamp'] + pd.Timedelta(seconds=median_purchase_approved), inplace=True)
for col in orders.columns:
    missings = len(orders[col][orders[col].isnull()])
    #print(col, missings)
# calculate the median of the approved_carrier column
median_approved_carrier = orders['approved_carrier'].median()
#print(median_approved_carrier) # in hours

# replace missing values in order_delivered_carrier_date by adding the calculated median to order_approved_at
orders['order_delivered_carrier_date'].fillna(orders['order_approved_at'] + pd.Timedelta(hours=median_approved_carrier), inplace=True)
# check missing values
for col in orders.columns:
    missings = len(orders[col][orders[col].isnull()])
    #print(col, missings)
#calculate the median of the carrier_delivered column
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
#print(orders.loc[missing_values_indexes])
# get descriptives on numerical variables
#print(orders.describe())
# get descriptives on non-numerical variables
#print(orders.describe(include=object))
# percentatge of orders that were delivered after the estimated delivery date
len(orders[orders['order_estimated_delivery_date'] <= orders['order_delivered_customer_date']]) / len(orders)
# NOTE: this is the same as checking whether delivered_estimated is negative
len(orders[orders['order_estimated_delivery_date'] <= orders['order_delivered_customer_date']]) / len(orders) == len(orders[orders['delivered_estimated'] < 0]) / len(orders)

#2.Order Payments & Order Items
order_payments = pd.read_csv("./Data/order_payments.csv")
#print(order_payments.head())
# compare the number of unique order_ids in orders with the number of observations
#print(len(order_payments['order_id'].unique()))
#print(len(order_payments))
# count the occurrences of each value in order_id
value_counts = order_payments['order_id'].value_counts()

# get the values with more than one occurrence for order_id
more_than_once = value_counts[value_counts > 1].index.tolist()

# subset based on the order_id
subset_df = order_payments[order_payments['order_id'].isin(more_than_once)].reset_index(drop=True)

# get the shape of the subset
#print(subset_df.shape)
# let's have a look at one specific order_id
#print("order_id under investigation:", more_than_once[0])
specific_order_id = subset_df[subset_df['order_id'] == more_than_once[0]]
#print(specific_order_id.sort_values('payment_sequential')) # order based on sequence of payment
# let's calculate the amount due
specific_order_id['payment_value'].sum()
# read in the order_items dataset
order_items = pd.read_csv("./Data/order_items.csv")
# show
#print(order_items.head())
# let's check the corresping row for our order_id under investigation
#print(order_items[order_items['order_id'] == more_than_once[0]])
# to do so, we will collapse the order_payments dataset so that we have one row for each unique order_id
# NOTE: this will drop the payment_sequential and payment_installments columns and the payment_type column will be the last one
grouped_df = order_payments.groupby(['order_id'])['payment_value'].sum().reset_index()

# rename column to 'total_amount_due'
grouped_df = grouped_df.rename(columns={'payment_value': 'total_amount_due'})

# show
#print(grouped_df.head())
# check to see if we find the same value as before
#print(grouped_df[grouped_df['order_id'] == more_than_once[0]])
# left merge with orders dataset (we add the payment value column)
orders_with_correct_price = pd.merge(orders, grouped_df, on='order_id', how='left')

# number of observations should be equal to the size of the orders dataset
orders_with_correct_price.shape
#print(orders_with_correct_price.head())
# drop price from order_items
order_items = order_items.drop('price', axis=1)

# left merge of order_items with orders_with_correct_price dataset (which includes the payment_value column)
order_items_payments = pd.merge(orders_with_correct_price, order_items, on='order_id', how='left')

# number of observations should be equal to the size of the orders dataset
#print(order_items_payments.shape)
#print(order_items_payments.head())

#3. Products
# import products
products = pd.read_csv("./Data/products.csv")

# show
#print(products.head())
# check descriptives of numerical columns
#print(products.describe())
#4. Customers
# import customers
customers = pd.read_csv("./Data/customers.csv")
# show
#print(customers.head())
# check data types
#print(customers.dtypes)
# convert to string type
customers['zip_code'] = customers['zip_code'].astype(int).astype(str)
#print(customers.dtypes)
# since we only have categorical features, we get count, unique, top, and the frequency of the most common value
#print(customers.describe())
# check for missing values
#print(customers.isnull().sum())
# function to extract province from zip code
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
# apply function on dataframe
customers['province_name'] = customers['zip_code'].apply(lambda x: zip_code_province(x))

# check for missing values again
#print(customers.isnull().sum())
# group by province and count number of customers
province_cust = customers.groupby('province_name')['customer_id'].count().sort_values(ascending=False).reset_index()

#rename columns
province_cust.rename(columns={'province_name': 'NE_Name', 'customer_id':'count'}, inplace=True)

print(province_cust)