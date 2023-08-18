#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
# ! pip install pandas-datareader
import pandas_datareader.data as web
import numpy as np
import pandas as pd
# !pip install tensorflow
# !pip install keras
# !pip install ipykernel
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


import datetime as dt
ticker="AAPL"
start=dt.datetime(2012,1,1)
end=dt.datetime(2019,12,17)
# !pip install yfinance
import yfinance as yf
data = yf.download('AAPL', start = start, end=end)


# In[3]:


#visualize the closing price
plt.figure(figsize=(16,8))
plt.title('close price history')
plt.plot(data['Close'])
plt.xlabel('Data',fontsize=18)
plt.ylabel('Close Price USD',fontsize=18)
plt.show()


# In[4]:


#creating a datafram with only close column
df=data.filter(['Close'])
#convert the dataframe to a numpy array
dataset=df.values
#get the number of rows to train the model on
training_data_len=math.ceil(len(dataset)*.8)
training_data_len


# In[5]:


#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


# In[6]:


#create the training data set
train_data=scaled_data[0:training_data_len , :]
#split the data into x_train and y_train data set
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<62:
        print(x_train)
        print(y_train)
        print()
    


# In[7]:


#convert the x_train to numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)


# In[8]:


#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],60,1))
x_train.shape


# In[9]:


#build LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[10]:


#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')


# In[11]:


#train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)
 


# In[12]:


#create the testing data set
#creating a new array containing scaled values from index 1544 to 2002
test_data=scaled_data[training_data_len:,:]
#creat the data sets x_train and y_train
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    


# In[13]:


#convert the data to a numpy array
x_test=np.array(x_test)


# In[14]:


#reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[15]:


#get the models predicted price values
prediction=model.predict(x_test)
predictions=scaler.inverse_transform(prediction)


# In[16]:


#get the root mean squared error (RMSE)
RMSE=np.sqrt(np.mean(predictions-y_test[:340,:])**2)
RMSE


# In[23]:


#plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid=predictions[:340,:]
#visualize the data
plt.figure(figsize=(16,8))
plt.title('MODEL')
plt.xlabel('Data',fontsize=18)
plt.ylabel('close price')
plt.plot(train['Close'])
plt.plot(valid["Close"])
plt.plot(valid[['Close','predictions']])
plt.legend(['Train','Val','predictions'],loc='lower right')
plt.show()


# In[18]:


#show the valid and the predicted prices
valid
df=pd.DataFrame(valid,columns=['a'])
df


# In[ ]:




