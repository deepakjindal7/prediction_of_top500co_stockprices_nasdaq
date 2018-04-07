
# coding: utf-8

# In[1]:


from keras.layers.core import Dense,Activation,Dropout


# In[2]:


from keras.layers.recurrent import LSTM


# In[3]:


from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:



    


# In[28]:


def load_data(file_name,seq_len,normalise_window):
    
    fs=open(file_name,'r',encoding='utf-8')
    f=fs.read()
    data=f.split('\n')
    sequence_len=seq_len+1
    result=[]
    for i in range(len(data)-sequence_len):
        result.append(data[i:i+sequence_len])
    
    row=round(int(result[0]),4)
    train=result[:int(row),:]
    np.random.shuffle(train)
    X_train=train[:,:-1]
    Y_train=train[:,:-1]
    X_test=result[int(row):,:-1]
    Y_test=result[int(row):,:-1]
    
    X_train=np.reshape(X_train,(X_train.shape[0].X_train.shape[1],1))
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    
    return [X_train,Y_train,X_test,Y_test] 


# In[20]:


def predict_sequence_full(model,data,window_size,pred_len):
    curr_frame=data[0]
    pred=[]
    for i in xrange(len(data)):
        pred.append(model.predict(curr_frame[newaxis,:,:][0,0]))
        curr_frame=curr_frame[1:]
        curr_frame=np.insert(curr_frame,[window_size-1],pred[-1],axis=0)
    return pred


# In[21]:


def predict_sequence_multiple(model,data,window_size,pred_len):
    pred=[]
    for i in xrange(len(data)/pred_len):
        curr_frame=data[i*pred_len]
        predicted=[]
        for j in xrange(pred_len):
            
            predicted.append(model.predict(curr_frame[newaxis,:,:][0,0]))
            curr_frame=curr_frame[1:]
            curr_frame=np.insert(curr_frame,[window_size-1],pred[-1],axis=0)
            pred.append(predicted)
    return pred


# In[22]:


def plot_results_multiple(pred_data,true_data,pred_len):
    fig=plt.figure(facecolor='white')
    ax=fig.add_subplot(111)
    ax.plot(true_data,label='True Data')
    print('yo')
    for i,data in enumerate(pred_data):
        padding=[None for p in xrange(i*pred_len)]
        plt.plot(padding + data,label='Prediction')
        plt.legend()
    plt.show() 


# In[29]:


[X_train,Y_train,X_test,Y_test]=load_data("all_stocks_5yr.csv",50,True)


# In[ ]:


model=Sequential()


# In[ ]:


d.head()


# In[ ]:


model.add(LSTM(units=50,input_shape=(16,16),return_sequences=True))


# In[ ]:


model.add(Dropout(0.2))


# In[ ]:


model.add(LSTM(units=100,return_sequences=True))


# In[ ]:


model.add(Dropout(0.2))


# In[ ]:


model.add(Dense(units=1))


# In[ ]:


model.add(Activation('linear'))


# In[ ]:


model.compile(loss='mse',optimizer='rmsprop')


# In[ ]:


model.fit(X_train,Y_train,batch_size=512, epochs=1,validation_split=0.02)


# In[ ]:


predictions=predict_sequence_multiple(model,X_test,50,50)
plot_results_multiple(predictions,Y_test,50)

