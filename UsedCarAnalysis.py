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

class carModel:
    
    def __init__(self, df, model):
        self.cars = df
        self.minPrice = 100
        self.maxPrice = 20000
        self.model = model
        
    def cleanData(self):
        #Pull out all rows corresponding to the model of interest
        self.cars.drop(self.cars.loc[self.cars['model'] != self.model].index, axis=0, inplace=True) 
        #Remove any rows that have nan values
        self.cars.dropna(axis = 0, how='any', inplace=True)
        #Remove any columns we don't care about
        self.cars.drop(['seller', 'offerType', 'abtest', 'gearbox', 'fuelType', 'notRepairedDamage', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen', 'name', 'vehicleType'], axis='columns', inplace=True)
        #Remove all rows with anomalous price information
        self.cars.drop(self.cars.loc[self.cars['price']<self.minPrice].index, axis=0, inplace=True)
        self.cars.drop(self.cars.loc[self.cars['price']>self.maxPrice].index, axis=0, inplace=True)
        
        carAges = self.calculateCarAge()
        cars['age'] = carAges
    
    def calculateCarAge(self):
        #Pull out month and year of registration
        years = self.cars['yearOfRegistration'].tolist()
        months = self.cars['monthOfRegistration'].tolist()
        
        #Make a list of datetime objects with the registration dates of the cars
        #Make datetime objects out of the month and year data
        registrationDates = []
        for i in range(0, len(months)):
            #First set all instances of month = 0 to month = 1
            if months[i] == 0:
                months[i] = 1
            registrationDates.append(datetime(years[i], months[i], 1).date())
        
        #Also pull out the date the data was crawled
        crawlDates = self.cars['dateCrawled'].tolist()
        crawlDates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S').date() for date in crawlDates]
        
        #Calculate the car age at the time of scraping
        carAges = (np.array(crawlDates) - np.array(registrationDates))
        carAges = [carAge.days for carAge in carAges]
        
        return(carAges)
    
    def plotPriceHist(self):
        prices = self.cars['price']
        ages = self.cars['age']
        
        plt.figure()
        plt.plot(ages, np.log(prices), 'o')

##Load the data into a pandas dataframe
cars = pd.read_csv("autos.csv", encoding = "ISO-8859-1")
model = 'fabia'

fabia = carModel(cars, 'micra')
fabia.cleanData()
fabia.plotPriceHist()

#    
##This is similarly granular. Maybe the registration date is more continuous?
#for carModel in exampleModels:
#    #Pull out month and year of registration
#    years = modelSubSets[carModel]['yearOfRegistration']
#    months = modelSubSets[carModel]['monthOfRegistration']
#    prices = modelSubSets[carModel]['price']
#    crawlDatesTmp = modelSubSets[carModel]['dateCrawled'].tolist()
#    
#    crawlDates = []
#    for date in crawlDatesTmp:
#        crawlDates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
#    
#    #Calculate time since registration - compute difference between registration date and date scraped
#    registrationDates = []
#    for i in range(0, len(years)):
#        if months.iloc[i] == 0:
#            months.iloc[i] = 1
#        registrationDates.append(datetime(years.iloc[i], months.iloc[i], 1))
#    
#    carAges = []
#    carAgestmp = np.array(crawlDates) - np.array(registrationDates)
#    for i in range(0, len(carAgestmp)):
#        carAges.append(carAgestmp[i].days)
#        
#    carAges = np.array(carAges)
#    prices = np.array(prices)
#    
#    rejectInices = np.where(carAges < 0)
#    
#    carAges = np.delete(carAges, rejectInices)
#    prices = np.delete(prices, rejectInices)
#    
#    print(len(carAges), len(prices))
#    
#    plt.figure(carModel)
#    plt.plot(carAges, prices, 'o')
#    plt.xlabel('Car Age (days)')
#    plt.ylabel('ln(Car Price (Euros))')
#    
##Using sklearn to fit a linear regression model
##Creat a linear regression object
#    
##Pick out the x and y data
#carModel = 'fabia'
#years = modelSubSets[carModel]['yearOfRegistration']
#months = modelSubSets[carModel]['monthOfRegistration']
#prices = modelSubSets[carModel]['price']
#crawlDatesTmp = modelSubSets[carModel]['dateCrawled'].tolist()
#    
#crawlDates = []
#for date in crawlDatesTmp:    
#    crawlDates.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
#    
##Calculate time since registration - compute difference between registration date and date scraped
#registrationDates = []
#for i in range(0, len(years)):
#    if months.iloc[i] == 0:
#        months.iloc[i] = 1
#    registrationDates.append(datetime(years.iloc[i], months.iloc[i], 1))
#    
#carAges = []
#carAgestmp = np.array(crawlDates) - np.array(registrationDates)
#for i in range(0, len(carAgestmp)):
#    carAges.append(carAgestmp[i].days)
#        
#carAges = np.array(carAges)
#prices = np.array(prices)
#rejectInices = np.where(carAges < 0)
#    
#fabiaAges = np.delete(carAges, rejectInices)
#fabiaPrices = np.delete(prices, rejectInices)
#
##Make a pandas dataframe out of the numpy arrays
#modelData = np.array([fabiaAges, np.log(fabiaPrices)]).T
#model = pd.DataFrame(modelData, columns = ['age','price'])
#model.sort_values('age', axis = 'index', inplace=True)
#
#lm = LinearRegression()
#X = model.drop('price', axis='columns')
#lm.fit(X, model.price)
#
#plt.figure('Fabia Example')
#plt.plot(model.age, model.price, 'o')
#plt.plot(model.age, lm.predict(X))
#
##Following from example above, split into test and training sets
#X_train, X_test, Y_train, Y_test = train_test_split(X, model.price, test_size = 0.2, random_state = 5)
#scores = cross_val_score(lm, X, model.price, cv=15, scoring = 'neg_mean_squared_error')    
#print('accuracy is {} +/- {}'.format(np.mean(scores), np.std(scores)))
#
##Try a polynomial fit
#polynomial_features= PolynomialFeatures(degree=5)
#x_poly = polynomial_features.fit_transform(X)
#lm.fit(x_poly, model.price)
#
#plt.figure('Fabia Example poly')
#plt.plot(model.age, model.price, 'o')
#plt.plot(model.age, lm.predict(x_poly))
##Would also need to run against some other models to determine which is most accurate