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
        self.cars = df.copy()
        self.minPrice = 100
        self.maxPrice = 30000
        self.maxAge = 50*365
        self.minAge = 0.5*365
        self.model = model
        self.decayConstant = 0
        self.initialPrice = 0
        self.linFitScore = -100
        
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
        #Only consider cars older than 6 months, younger than 50 years
        carAges = self.calculateCarAge()
        self.cars['age'] = carAges
        
        self.cars.drop(self.cars.loc[self.cars['age']>self.maxAge].index, axis=0, inplace=True)
        self.cars.drop(self.cars.loc[self.cars['age']<self.minAge].index, axis=0, inplace=True)
    
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
        
    def linRegress(self, plot=False):
        
        X = self.cars['age']
        Y = np.log(self.cars['price'])
        
        #Compute weights for the logarithmic price data - using functional approach
        priceErr = 10
        lnPriceErr = np.log(self.cars['price'] + priceErr) - np.log(priceErr)
        if len(X > 1):
            #Perform a linear regression - use cross validation to determine to measure fit quality
            lm = LinearRegression()
            scores = cross_val_score(lm, np.array(X).reshape(-1, 1), Y, cv=10, scoring = 'neg_mean_squared_error') 
            self.linFitScore = np.mean(scores)
            
            #print('accuracy is {} +/- {}'.format(np.mean(scores), np.std(scores)))
            lm.fit(np.array(X).reshape(-1, 1), Y, lnPriceErr)
            self.decayConstant = lm.coef_[0]
            self.initialPrice = lm.intercept_
            
            #Plot the linear fit if required
            if plot:
                plt.figure(self.model)
                plt.plot(X, Y, 'o')
                plt.plot(X, lm.predict(np.array(X).reshape(-1, 1)))
                plt.xlabel('Age (days)')
                plt.ylabel('ln(Price (EUR))')
                
    def plotPriceHist(self, trend = False):
        plt.figure(self.model + ' Price History')
        plt.plot(self.cars['age'], self.cars['price'], 'o')
        plt.xlabel('Car Age (days)')
        plt.ylabel('Price (EUR)')
        
        if trend:
            xs = np.sort(self.cars['age'])
            A = np.exp(self.initialPrice)
            alpha = self.decayConstant
            predict = A*np.exp(alpha*xs)
            plt.plot(xs, predict)
                
    def dataMetrics(self):
        self.numEntries = len(self.cars)
        
        
##Load the data into a pandas dataframe
cars = pd.read_csv("autos.csv", encoding = "ISO-8859-1")
cars.dropna(inplace=True, axis='rows')
model = 'golf'

models = cars.model.unique()

test = carModel(cars, model)
test.cleanData()
test.linRegress(plot=True)

modelDict = {}
index = np.linspace(0, len(models), len(models))

decayConstants = []
models_for_analysis = []
initialPrices = []

for model in models:
    modelDict[model] = carModel(cars, model)
    modelDict[model].cleanData()
    modelDict[model].dataMetrics()
    
    #Only analyse models for which there are enough entries
    if modelDict[model].numEntries > 100:
        modelDict[model].linRegress()
        #Only include data that are well-described by a linear fit
        if modelDict[model].linFitScore > - 0.5:
            decayConstants.append(modelDict[model].decayConstant)
            initialPrices.append(modelDict[model].initialPrice)
            models_for_analysis.append(model)

models_for_analysis = pd.Series(models_for_analysis)
sortindices= np.argsort(decayConstants)
index = range(0, len(decayConstants))

plt.figure('Decay Constant Bar Chart')
plt.bar(index, np.sort(decayConstants))
plt.xticks(index, models_for_analysis[sortindices], rotation=30)

sortindices = np.argsort(initialPrices)
plt.figure('Initial Price Bar Chart')
plt.bar(index, np.exp(np.sort(initialPrices)))
plt.xticks(index, models_for_analysis[sortindices], rotation=30)