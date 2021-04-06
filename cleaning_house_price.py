##
##       Hulda Nana Nkwenkeu
##   Copyright -  all right reserved - @2021
##

##
##   Cleaning house prices data
##


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm


sns.set()


##
##  Loading data
##

house_data = pd.read_excel('Zoopla_data_info.xls')

##
## we want to delete any rows not starting with the pound sign '£'
##

position_of_prices = house_data['Price'].str.contains('£')
index_to_delete = np.where(position_of_prices == False)
house_data_updated = house_data.drop(house_data.index[index_to_delete])

##
## convert prices column such that it becomes a double
##

house_data_updated['Price'] = house_data_updated['Price'].apply(lambda x: x.replace('£','').replace(',','')).astype('float')


##
## we want to only keep rows which price are between a minPrice and maxPrice
##
minPrice = 200000
maxPrice = 1000000

#starting with min price
position_of_prices_minPrice = house_data_updated['Price'] < minPrice 
index_to_delete_minPrice = np.where(position_of_prices_minPrice == True)
house_data_updated = house_data_updated.drop(house_data_updated.index[index_to_delete_minPrice])

# and then max price
position_of_prices_maxPrice = house_data_updated['Price'] > maxPrice
index_to_delete_maxPrice = np.where(position_of_prices_maxPrice == True)
house_data_updated = house_data_updated.drop(house_data_updated.index[index_to_delete_maxPrice])

##
## check if any bath column is set to Nan. If so, update it to 1 - this is our default value
##

if house_data_updated['Number of Baths'].isnull().values.any():
    house_data_updated['Number of Baths'] = house_data_updated['Number of Baths'].fillna(1)
    
##
## check if any media floor plan is set to Nan. If so, update it to 0
##
if house_data_updated['Media Floor Plan'].isnull().values.any():
    house_data_updated['Media Floor Plan'] = house_data_updated['Media Floor Plan'].fillna(0)
    

##
## check if any media images is set to Nan. If so, update it to 0
##
if house_data_updated['Media Images'].isnull().values.any():
    house_data_updated['Media Images'] = house_data_updated['Media Images'].fillna(0)

##
## check if any media video is set to Nan. If so, update it to 0
##
if house_data_updated['Media Video'].isnull().values.any():
    house_data_updated['Media Video'] = house_data_updated['Media Video'].fillna(0)
    
##
## check if number of beds if set to Nan. If so, delete the corresponding rows
##
if house_data_updated['Number of Beds'].isnull().values.any():
    position = house_data_updated['Number of Beds'].isnull()
    index_position = np.where(position == True)
    house_data_updated = house_data_updated.drop(house_data_updated.index[index_position])

##
## computing the average distance to a subway given the position of the house
## adjusting the number of public transport that are close by
##

# the input data is meant to be a string where values are seperated by semi-column ";"
def average_distance_to_home(val, return_distance = True):
    
    list_of_transport = val.split(';')
    nb_of_transport = 0
    average_distance = 0
    
    for transport in list_of_transport:
        pos = transport.find('miles')
        if pos != -1:
            average_distance += float(transport[0:pos])
            nb_of_transport += 1
    
    if return_distance: 
        return average_distance/nb_of_transport
    else:
        return nb_of_transport


house_data_updated['Number of Transport'] = house_data_updated['Transport Information'].apply(lambda x: average_distance_to_home(x,False))
house_data_updated['Distance to Transport'] = house_data_updated['Transport Information'].apply(lambda x: average_distance_to_home(x,True))



##
## Cleaning data further in order to keep only baths and beds that are lower than 5
##

maxBaths = 4
maxBeds = 4

position_of_prices_maxBaths = house_data_updated['Number of Baths'] > maxBaths 
index_to_delete_maxBaths = np.where(position_of_prices_maxBaths == True)
house_data_updated = house_data_updated.drop(house_data_updated.index[index_to_delete_maxBaths])

position_of_prices_maxBeds = house_data_updated['Number of Beds'] > maxBeds 
index_to_delete_maxBeds = np.where(position_of_prices_maxBeds == True)
house_data_updated = house_data_updated.drop(house_data_updated.index[index_to_delete_maxBeds])


##
##  keep a specific list of columns to proceed
##

list_of_columns = ['Price','Media Images','Media Video','Number of Beds','Media Floor Plan','Number of Baths','Number of Transport','Distance to Transport','Apt Post Code','Listing ID']

house_data_final = house_data_updated[list_of_columns]



##
##  Datat Visualization
##

## Heat Map - Correlation matrix
matrix= house_data_final.corr()
#f, ax = plt.subplots(figsize=(16,12))
sns.heatmap(matrix,vmax=0.7, square=True)


## distribution of prices

f, ax = plt.subplots()
ax.set_xlim(min(house_data_final['Price']), max(house_data_final['Price']))
sns.displot(house_data_final['Price'], ax=ax)


## count plot for numbers of beds and baths
fig, ax = plt.subplots(2,1,figsize=(12,20))
sns.countplot(house_data_final['Number of Beds'], ax=ax[0])
sns.countplot(house_data_final['Number of Baths'],ax=ax[1])


## pair plot of interesting columns
intersting_cols = ['Price', 'Number of Beds','Distance to Transport','Media Images','Number of Baths']
sns.pairplot(house_data_final[intersting_cols], height = 2.5)

## descriptive analytics for each column
data_description = house_data_final.describe()



##
## given the above study of our initial data we can now limit our data to a specific list
##
list_of_columns_updated = ['Price','Media Images','Media Video','Number of Beds','Media Floor Plan','Number of Baths','Distance to Transport','Apt Post Code']
house_data_final = house_data_final[list_of_columns_updated]

## turning the categorial column post code into a numerical one based on unique item
house_data_final_updated = pd.get_dummies(house_data_final)

# dividing our dataset into two set - X and Y
Y = house_data_final_updated['Price']
X = house_data_final_updated.drop(columns=['Price'])


##
##  We will use cross validation technique in order to build a better predictor
##
number_of_folds = 20
number_of_estimators = 200
kf = KFold(n_splits=number_of_folds)
cv_indices = kf.split(X)

##
## developing a funciton to calculate the accuracy
##
def accuracy_of_fit(y_true, prediction):
    
    accuracy = 0
    y_true_list = list(y_true)
    prediction_list = list(prediction)
    for i in list(range(len(y_true_list))):
        accuracy += abs( y_true_list[i] - prediction_list[i])/y_true_list[i]
    
    accuracy = accuracy/len(y_true_list)
    
    return accuracy

##
## development of the random forest model
##

# Create the model we are using
rf = RandomForestRegressor(n_estimators = number_of_estimators, random_state = 42)
train_index = []
test_index = []
errors = list(range(number_of_folds))
accuracy = list(range(number_of_folds))
r2squared = list(range(number_of_folds))
iterator = 0

if True:
    
    for train_index, test_index in cv_indices:
        print("RF : running case ", iterator)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        rf.fit(X_train, Y_train)
        predictions = rf.predict(X_test)
        
        # compute errors
        errors[iterator] = mean_absolute_error(Y_test,predictions)
        
        # r2squared
        r2squared[iterator] = r2_score(Y_test, predictions)
        
        # accuracy
        accuracy[iterator] = 100 * accuracy_of_fit(Y_test, predictions)
        
        iterator += 1
        
    
    # Display the performance metrics RF
    print('RF: accuracy:', round(100-np.mean(accuracy),4), '%')
    print('RF: Mean Absolute Error:', round(np.mean(errors), 4))
    print('RF: Mean R2 Squared:', round(np.mean(r2squared), 4))
    
    # plotting prediction vs real values for the last training set
    plt.figure(figsize=(10,10))
    plt.scatter(Y_test, predictions, c='crimson')
    p1 = max(max(Y_test), max(predictions))
    p2 = min(min(Y_test), min(predictions))
    plt.plot([p1,p2],[p1,p2], 'b-')
    plt.xlabel("True Values", fontsize = 10)
    plt.ylabel("Predicted Values - Random Forest", fontsize = 10)
    plt.show()


##
## development of the Generalized Linear Model using Gaussian Link
##
if True :
    
    cv_indices_glm = kf.split(X)
    train_index = []
    test_index = []
    error_squared_glm = list(range(number_of_folds))
    errors_glm = list(range(number_of_folds))
    accuracy_glm = list(range(number_of_folds))
    r2squared_glm = list(range(number_of_folds))
    iterator = 0
    
    for train_index, test_index in cv_indices_glm:
        print("GLM : running case ", iterator)
        X_train_glm, X_test_glm = X.iloc[train_index], X.iloc[test_index]
        Y_train_glm, Y_test_glm = Y.iloc[train_index], Y.iloc[test_index]
        
        X_train_glm = sm.add_constant(X_train_glm)
        glm_Gauss = sm.GLM(Y_train_glm, X_train_glm, family=sm.families.Gaussian(link = sm.families.links.log()))
        glm_res = glm_Gauss.fit()
        
        X_test_glm = sm.add_constant(X_test_glm)
        predictions_glm = glm_res.predict(X_test_glm)
        
        
        # compute errors
        errors_glm[iterator] = mean_absolute_error(Y_test_glm,predictions_glm)
        
        # r2squared
        r2squared_glm[iterator] = r2_score(Y_test_glm, predictions_glm)
        
        # accuracy
        accuracy_glm[iterator] = 100 * accuracy_of_fit(Y_test_glm, predictions_glm)
        
        iterator += 1
    
    # Display the performance metrics GLM
    print('GLM: Accuracy:', round(100-np.mean(accuracy_glm),2), '%')
    print('GLM: Absolute Error:', round(np.mean(errors_glm), 2))
    print('GLM: R2 Squared:', round(np.mean(r2squared_glm), 2))
    
    # plotting prediction vs real values for the last training set
    plt.figure(figsize=(10,10))
    plt.scatter(Y_test_glm, predictions_glm, c='crimson')
    p1 = max(max(Y_test_glm), max(predictions_glm))
    p2 = min(min(Y_test_glm), min(predictions_glm))
    plt.plot([p1,p2],[p1,p2], 'b-')
    plt.xlabel("True Values", fontsize = 10)
    plt.ylabel("Predicted Values - GLM", fontsize = 10)
    plt.show()