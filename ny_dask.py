#!/usr/bin/env python
# coding: utf-8

# NY TAXI FARE ANALYSIS USING DASK

from dask.distributed import Client

client = Client()
client


import dask
import dask.dataframe as dd
import hvplot.dask

df = dd.read_csv('train500K.csv', parse_dates=["pickup_datetime"])

df


print(f"shape : ({len(df)},{len(df.columns)})")


df.hvplot.hist(y='fare_amount', title='Fare distribution')
df.hvplot.box(y='fare_amount')


df = df[(df.fare_amount > 3) & (df.fare_amount < 1000)]

#clean longitude and latitude
df = df[(df.pickup_longitude > -76) & (df.pickup_longitude < -70)]
df = df[(df.pickup_latitude < 43) & (df.pickup_latitude >= 40)]

df.hvplot.box(y='passenger_count')

df = df[(df.passenger_count > 0) & (df.passenger_count <= 8)]

df = df.drop(labels='key', axis=1)

df.head()


# PREPRO

df.isnull().sum().compute()


import numpy as np

def deg_to_rad(deg):
    """Convert degrees into radian, return the result"""
    return deg * (np.pi/180)

def get_distanceBetween(dropLat, pickLat, dropLong, pickLong):
    """Return the distance in km between pickup long/lat and dropoff long/lat"""
    R = 6371 #Radius of the earth in km
    delta_lat = deg_to_rad(dropLat - pickLat)
    delta_long = deg_to_rad(dropLong - pickLong)
    a = np.sin(delta_lat/2) * np.sin(delta_lat/2) + np.cos(deg_to_rad(pickLat)) * np.cos(deg_to_rad(dropLat)) * np.sin(delta_long/2) * np.sin(delta_long/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df['distance_km'] = get_distanceBetween(df.dropoff_latitude, df.pickup_latitude, df.dropoff_longitude, df.pickup_longitude)

df.head()


LAT_JFK = 40.6441666
LON_JFK = -73.7822222
LAT_LAGU = 40.7747222
LON_LAGU = -73.8719444
LAT_NEW = 40.6897222
LON_NEW = -74.175

#**** FROM/TO JFK ****#

df['distance_From_JFK_to_dropoff'] = get_distanceBetween(df.dropoff_latitude, LAT_JFK, df.dropoff_longitude, LON_JFK)
df['distance_From_pickup_to_JFK'] = get_distanceBetween(LAT_JFK, df.pickup_latitude, LON_JFK, df.pickup_longitude)

#**** FROM/TO LAGUARDIA ****#

df['distance_From_LAGU_to_dropoff'] = get_distanceBetween(df.dropoff_latitude, LAT_LAGU, df.dropoff_longitude, LON_LAGU)
df['distance_From_pickup_to_LAGU'] = get_distanceBetween(LAT_LAGU, df.pickup_latitude, LON_LAGU, df.pickup_longitude)

#**** FROM/TO NEWARK ****#

df['distance_From_NEW_to_dropoff'] = get_distanceBetween(df.dropoff_latitude, LAT_NEW, df.dropoff_longitude, LON_NEW)
df['distance_From_pickup_to_NEW'] = get_distanceBetween(LAT_NEW, df.pickup_latitude, LON_NEW, df.pickup_longitude)


import pandas as pd


df['pickup_datetime_us_east'] = (
                        df['pickup_datetime'].map(lambda x: pd.to_datetime(x, errors='coerce'))
)


# Heavy operation, need to persist
df = df.persist()
df = df.set_index('pickup_datetime_us_east')


df.head()


df['year_pickup'] = df['pickup_datetime'].apply(lambda x: x.year)
df['month_pickup'] = df['pickup_datetime'].apply(lambda x: x.month)
df['day_pickup'] = df['pickup_datetime'].apply(lambda x: x.day)
