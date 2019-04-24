# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:02:11 2019

@author: Jozef Tkocz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sklearn

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

plt.close('all')

#Load the data into a pandas dataframe
cars = pd.read_csv("autos.csv", encoding = "ISO-8859-1")

#print a data summary
print(cars.describe())

#Remove columns we don't care about
#cars.replace(["NaN", 'NaT'], np.nan, inplace = True)
cars.dropna(axis = 0, how='any', inplace=True)
print(cars.columns[cars.isna().any()].tolist())
#cars.dropna(axis = 'columns', how='any')

print(cars.describe())
cars.drop(['seller', 'offerType', 'abtest', 'gearbox', 'fuelType', 'notRepairedDamage', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name', 'vehicleType'], axis='columns', inplace=True)

#Rmove all rows with anomalous price and date information
#tooCheap = cars.loc[cars['price']<100]
cars.drop(cars.loc[cars['price']<100].index, axis=0, inplace=True)
cars.drop(cars.loc[cars['price']>20000].index, axis=0, inplace=True)
#print a data summary


#List unique car brands
carBrands = cars.brand.unique()

#Break the data down into unique brands
brandSubSets = {}
for brand in carBrands:
    brandSubSets[brand] = cars.loc[cars['brand'] == brand]

#Break the data down into unique models
modelSubSets = {}
#Loop over all brands    
for brand in brandSubSets:
    #Make a subset for each model in each brand
    carModels = brandSubSets[brand].model.unique()
    for model in carModels:
        modelSubSets[model] = brandSubSets[brand].loc[brandSubSets[brand]['model'] == model]
        
#Choose an example subset of the data - look at Skodas
#List all unique Skoda models
exampleModels = brandSubSets['skoda'].model.unique()

'''for carModel in exampleModels:
    plt.figure(str(carModel))
    plt.plot(modelSubSets[carModel]['kilometer'], modelSubSets[carModel]['price'], 'o')
    plt.xlabel('Odometer Reading (km)')
    plt.ylabel('Car Price (Euros)')'''
    
#Now have a problem = odometer data looks to be binned, not continuous! Model each bin as a continuous (Gaussian) distribution
#Determine the mean and standard error for each bin

'''for carModel in exampleModels:
    #determine number of unique values for attribute of interest
    kilometerBins = modelSubSets[carModel].kilometer.unique()
    #Create an array to store the mean and standard error price for each kilometer bin
    prices = np.zeros(len(kilometerBins))
    priceError = np.zeros(len(kilometerBins))
    i = 0
    #Compute the mean and standard error for the price of each bin
    for kilometer in kilometerBins:
        priceData = modelSubSets[carModel].loc[modelSubSets[carModel]['kilometer'] == kilometer]['price']
        mean = np.mean(priceData)
        stdError = np.std(priceData/np.sqrt(len(priceData)))
        
        prices[i] = mean
        priceError[i] = stdError
        i += 1
        
    plt.figure(str(carModel))
    logErrors = np.abs(np.log(prices+stdError) - np.log(prices))
    plt.errorbar(kilometerBins*1.0e-3, np.log(prices), yerr = logErrors, fmt = 'o', color = 'k')
    #plt.plot(kilometerBins, prices, 'o')
    plt.xlabel('Odometer Reading (km)')
    plt.ylabel('Car Price (Euros)')'''
    
#Data sets look too small to perform meaningful regression analysis on - repeat by brand? Are all datasets binned?
    
'''for brand in brandSubSets:
    plt.figure(str(brand))
    plt.plot(brandSubSets[brand]['kilometer'], brandSubSets[brand]['price'], 'o')'''
    
#This is similarly granular. Maybe the registration date is more continuous?
for carModel in exampleModels:
    #Pull out month and year of registration
    years = modelSubSets[carModel]['yearOfRegistration']
    months = modelSubSets[carModel]['monthOfRegistration']
    prices = modelSubSets[carModel]['price']
    crawlDatesTmp = modelSubSets[carModel]['dateCrawled'].tolist()
    
    crawlDates = []
    for date in crawlDatesTmp:
        crawlDates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
    
    #Calculate time since registration - compute difference between registration date and date scraped
    registrationDates = []
    for i in range(0, len(years)):
        if months.iloc[i] == 0:
            months.iloc[i] = 1
        registrationDates.append(datetime(years.iloc[i], months.iloc[i], 1))
    
    carAges = []
    carAgestmp = np.array(crawlDates) - np.array(registrationDates)
    for i in range(0, len(carAgestmp)):
        carAges.append(carAgestmp[i].days)
        
    carAges = np.array(carAges)
    prices = np.array(prices)
    
    rejectInices = np.where(carAges < 0)
    
    carAges = np.delete(carAges, rejectInices)
    prices = np.delete(prices, rejectInices)
    
    print(len(carAges), len(prices))
    
    plt.figure(carModel)
    plt.plot(carAges, prices, 'o')
    plt.xlabel('Car Age (days)')
    plt.ylabel('ln(Car Price (Euros))')
    
#Using sklearn to fit a linear regression model
#Creat a linear regression object
    
#Pick out the x and y data
carModel = 'fabia'
years = modelSubSets[carModel]['yearOfRegistration']
months = modelSubSets[carModel]['monthOfRegistration']
prices = modelSubSets[carModel]['price']
crawlDatesTmp = modelSubSets[carModel]['dateCrawled'].tolist()
    
crawlDates = []
for date in crawlDatesTmp:    
    crawlDates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
    
#Calculate time since registration - compute difference between registration date and date scraped
registrationDates = []
for i in range(0, len(years)):
    if months.iloc[i] == 0:
        months.iloc[i] = 1
    registrationDates.append(datetime(years.iloc[i], months.iloc[i], 1))
    
carAges = []
carAgestmp = np.array(crawlDates) - np.array(registrationDates)
for i in range(0, len(carAgestmp)):
    carAges.append(carAgestmp[i].days)
        
carAges = np.array(carAges)
prices = np.array(prices)
rejectInices = np.where(carAges < 0)
    
fabiaAges = np.delete(carAges, rejectInices)
fabiaPrices = np.delete(prices, rejectInices)

#Make a pandas dataframe out of the numpy arrays
modelData = np.array([fabiaAges, np.log(fabiaPrices)]).T
model = pd.DataFrame(modelData, columns = ['age','price'])
model.sort_values('age', axis = 'index', inplace=True)

lm = LinearRegression()
X = model.drop('price', axis='columns')
lm.fit(X, model.price)

plt.figure('Fabia Example')
plt.plot(model.age, model.price, 'o')
plt.plot(model.age, lm.predict(X))

#Following from example above, split into test and training sets
X_train, X_test, Y_train, Y_test = train_test_split(X, model.price, test_size = 0.2, random_state = 5)
scores = cross_val_score(lm, X, model.price, cv=15, scoring = 'neg_mean_squared_error')    
print('accuracy is {} +/- {}'.format(np.mean(scores), np.std(scores)))

#Try a polynomial fit
polynomial_features= PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(X)
lm.fit(x_poly, model.price)

plt.figure('Fabia Example poly')
plt.plot(model.age, model.price, 'o')
plt.plot(model.age, lm.predict(x_poly))
#Would also need to run against some other models to determine which is most accurate