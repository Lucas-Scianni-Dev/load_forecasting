import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU, RepeatVector, Dropout, Bidirectional
import tensorflow as tf
from datetime import date
from datetime import datetime
from datetime import timedelta
import nbimporter
from sklearn.model_selection import KFold
import timeit

def Run_MLP_Completa(submercado,inp, data_prev, var_norm,graficos,epochs,n_splits,NumHiddenLayers,verbose,horizonte):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    data_treino_MLP = data_prev["data_treino_MLP"]
    data_teste_MLP = data_prev["data_teste_MLP"]
        
    X_input_MLP_treino = inp["X_input_MLP_treino"]
    Y_input_MLP_treino = inp["Y_input_MLP_treino"]
    X_input_MLP_teste = inp["X_input_MLP_teste"]
    Y_input_MLP_teste = inp["Y_input_MLP_teste"]

    tic=timeit.default_timer()  
    NumNeurons=int(0.5*X_input_MLP_treino.shape[1])
    beta=0.1
    VALIDATION_SPLIT=0.25
    learning_rate=1e-3

    class MCDropout(Dropout):
        def call(self, inputs):
            return super().call(inputs, training = True)

    def my_loss_fn(y_true, y_pred):
        return tf.reduce_mean(((y_pred*c_std+c_min)/(y_true*c_std+c_min)-1)**2)


    def create_model():
        model = tf.keras.Sequential()
        for i in range(NumHiddenLayers):
            Layer=Dense(NumNeurons, kernel_initializer=keras.initializers.he_normal(seed=1), activation='relu')
            model.add(Layer)
            model.add(MCDropout(beta))


        OutputLayer=keras.layers.Dense(Y_input_MLP_treino.shape[1], kernel_initializer=keras.initializers.he_normal(seed=1), activation='relu')
        model.add(OutputLayer)

        loss_object = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), loss=my_loss_fn, metrics=["accuracy"])

        return model

    kfold = KFold(n_splits=n_splits, shuffle=True)

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (1000)])
        return pred_stack 

    Prediction = []
    j = 1
    for train, valid in kfold.split(X_input_MLP_treino,Y_input_MLP_treino):
        X_treino, X_valid = X_input_MLP_treino[train], X_input_MLP_treino[valid]
        Y_treino, Y_valid = Y_input_MLP_treino[train], Y_input_MLP_treino[valid]
        MLP_completa = create_model()
        
        #print('split',j)
        for i in range(epochs):  
            MLP_completa.train_on_batch(X_treino, Y_treino)
            if verbose == 1:
                print("Epoch ",i)
        
        MLP_completa.save("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Modelos_Treinados/MLP_completa_"+submercado+"_"+str(j)+"_"+str(horizonte)+"_dias")
        
        j= j+1
        
        prediction_valid = predict(MLP_completa,X_valid)
        prediction_valid = prediction_valid*c_std+c_min
        Y_valid = Y_valid*c_std+c_min

        media_valid = np.mean(prediction_valid, axis = 0)
        MAPE_MLP_valid=(np.abs((media_valid-Y_valid)/Y_valid))*100
        MAPE_AVG_MLP_valid=np.average(MAPE_MLP_valid,axis=1)
        media_mape_valid = np.average(MAPE_AVG_MLP_valid,axis=0)

        print("MAPE medio validacao kfold: " ,media_mape_valid)

        prediction=predict(MLP_completa,X_input_MLP_teste)
        Prediction.append(prediction)


    toc=timeit.default_timer()  

    Prediction = np.asarray(Prediction)
    y_pred_Stack=Prediction.mean(axis=0)
    y_pred_stack=y_pred_Stack*c_std+c_min
    Y_MLP_test = Y_input_MLP_teste*c_std+c_min
    
    media = np.mean(y_pred_stack, axis = 0)
    
    MAPE_MLP=np.abs((media-Y_MLP_test)/Y_MLP_test)
    MAPE_AVG_MLP=np.average(MAPE_MLP,axis=1)*100
    media_mape = np.average(MAPE_AVG_MLP,axis=0)
    
    Y_MLP_test_mean = np.mean(Y_MLP_test,axis = 1)

    MSE = []
    nash_MLP = []
    DM_MLP = []
    for i in range(Y_MLP_test.shape[0]):
        nash_MLP_i=1-np.sum((Y_MLP_test[i]-media[i])**2)/np.sum((Y_MLP_test[i]-Y_MLP_test_mean[i])**2)
        nash_MLP.append(nash_MLP_i)

        DM_MLP_i = np.sqrt(((MAPE_AVG_MLP[i]/100)**2)+(1-nash_MLP[i])**2)
        DM_MLP.append(DM_MLP_i)

        MSE_i = np.sum((y_pred_Stack[i]-Y_input_MLP_teste[i])**2)/Y_MLP_test.shape[1]
        MSE.append(MSE_i) 
    
    print("Tempo de Simulação:",toc - tic)
    print("MAPE médio:",media_mape)
    
    data = data_teste_MLP
    if graficos ==1: 
        for i in range(8):
            plt.figure(figsize=(8,8))
            plt.plot(Y_MLP_test[i],label='Y_test')
            plt.plot(media[i],label='Prediction')
            plt.title("Data: %s \n MLP + MCDropout + CrossValidation \nMAPE=%.2f    NASH= %.3f   DM= %.3f "%(data[i].item().strftime("%d/%m/%Y"),MAPE_AVG_MLP[i] ,nash_MLP[i], DM_MLP[i]))
            plt.legend()
    
    resultados = {"mape":MAPE_MLP,"nash": nash_MLP, "mse":MSE}
    return media,Y_MLP_test,data_teste_MLP,resultados

def Run_LSTM_Kfold(submercado, inp, data_prev, var_norm,graficos,epochs,n_splits,verbose,horizonte):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    data_treino_lstm = data_prev["data_treino_lstm"]
    data_teste_lstm = data_prev["data_teste_lstm"]
        
    X_LSTM_treino = inp["X_LSTM_treino"]
    Y_LSTM_treino = inp["Y_LSTM_treino"]
    X_LSTM_teste = inp["X_LSTM_teste"]
    Y_LSTM_teste = inp["Y_LSTM_teste"]

    tic=timeit.default_timer()
    
    NumHiddenLayers=1
    NumNeurons=100
    beta=0.1
    VALIDATION_SPLIT=0.25
    learning_rate=1e-3
    
    class MCDropout(Dropout):
        def call(self, inputs):
            return super().call(inputs, training = True)
    
    def my_loss_fn(y_true, y_pred):
        return tf.reduce_mean(((y_pred*c_std+c_min)/(y_true*c_std+c_min)-1)**2)

    def create_model():
        model = Sequential()
        for i in range(NumHiddenLayers):
            model.add(Bidirectional(LSTM(100), input_shape = (X_LSTM_treino.shape[1], X_LSTM_treino.shape[2])))
            model.add(Dense(Y_LSTM_treino.shape[1]))
 
    
        #model.compile(loss=my_loss_fn, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])
        model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])

        return model
    
    kfold = KFold(n_splits=n_splits, shuffle=True)

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (1)])
        return pred_stack
    
    Prediction = []
    j = 1

    for train, valid in kfold.split(X_LSTM_treino,Y_LSTM_treino):
        X_treino, X_valid = X_LSTM_treino[train], X_LSTM_treino[valid]
        Y_treino, Y_valid = Y_LSTM_treino[train], Y_LSTM_treino[valid]
        lstm_modelo = create_model()

        print("split",j)
        for i in range(epochs):  
            lstm_modelo.train_on_batch(X_LSTM_treino, Y_LSTM_treino)
            if verbose == 1:
                print("Epoch ",i)
                
        lstm_modelo.save("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Modelos_Treinados/LSTM_completa_"+submercado+"_"+str(j)+"_"+str(horizonte)+"_dias")
        j= j+1
        
        prediction_valid = predict(lstm_modelo,X_valid)
        prediction_valid = prediction_valid*c_std+c_min
        Y_valid = Y_valid*c_std+c_min

        media_valid = np.mean(prediction_valid, axis = 0)
        MAPE_LSTM_valid=(np.abs((media_valid-Y_valid)/Y_valid))*100
        MAPE_AVG_LSTM_valid=np.average(MAPE_LSTM_valid,axis=1)
        media_mape_valid = np.average(MAPE_AVG_LSTM_valid,axis=0)

        print("MAPE medio validacao kfold: " ,media_mape_valid)

        prediction=predict(lstm_modelo,X_LSTM_teste)
        Prediction.append(prediction)

    toc=timeit.default_timer()

    Prediction = np.asarray(Prediction)
    y_pred_Stack=Prediction.mean(axis=0)
    y_pred_stack=y_pred_Stack*c_std+c_min
    Y_LSTM_Teste = Y_LSTM_teste*c_std+c_min

    media = np.mean(y_pred_stack, axis = 0)

    MAPE_LSTM=np.abs((media-Y_LSTM_Teste)/Y_LSTM_Teste)
    MAPE_AVG_LSTM=np.average(MAPE_LSTM,axis=1)*100
    media_MAPE_LSTM = np.average(MAPE_AVG_LSTM)
       
    Y_LSTM_test_mean = np.mean(Y_LSTM_Teste,axis = 1)
    
    MSE = []
    nash_LSTM = []
    DM_LSTM = []
    for i in range(Y_LSTM_Teste.shape[0]):
        nash_LSTM_i=1-np.sum((Y_LSTM_Teste[i]-media[i])**2)/np.sum((Y_LSTM_Teste[i]-Y_LSTM_test_mean[i])**2)
        nash_LSTM.append(nash_LSTM_i)

        DM_LSTM_i = np.sqrt(((MAPE_AVG_LSTM[i]/100)**2)+(1-nash_LSTM[i])**2)
        DM_LSTM.append(DM_LSTM_i)

        MSE_i = np.sum((y_pred_Stack[0][i]-Y_LSTM_teste[i])**2)/Y_LSTM_teste.shape[1]
        MSE.append(MSE_i) 

    print("\nTempo de simulação:",toc-tic)
    print("MAPE médio:",media_MAPE_LSTM)

    data = data_teste_lstm    
    if graficos ==1:
        for i in range(8):
            plt.figure(figsize=(8,8))
            plt.plot(Y_LSTM_Teste[i],label='Y_test')
            plt.plot(media[i],label='Prediction')
            plt.title("Data: %s \n LSTM \nMAPE=%.2f    NASH= %.3f   DM= %.3f "%(data[i].strftime("%d/%m/%Y"),MAPE_AVG_LSTM[i] ,nash_LSTM[i], DM_LSTM[i]))
            plt.legend()
    
    resultados = {"mape":MAPE_LSTM,"nash": nash_LSTM, "mse":MSE}

    return media,Y_LSTM_Teste,data_teste_lstm,resultados

def Run_Uni_LSTM_Kfold(submercado,inp, data_prev, var_norm,graficos,epochs,n_splits,verbose,horizonte):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    data_treino_lstm = data_prev["data_treino_lstm"]
    data_teste_lstm = data_prev["data_teste_lstm"]
        
    X_LSTM_treino = inp["X_LSTM_treino"]
    Y_LSTM_treino = inp["Y_LSTM_treino"]
    X_LSTM_teste = inp["X_LSTM_teste"]
    Y_LSTM_teste = inp["Y_LSTM_teste"]

    tic=timeit.default_timer()
    
    NumHiddenLayers=1
    NumNeurons=100
    beta=0.1
    VALIDATION_SPLIT=0.25
    learning_rate=1e-3
    
    class MCDropout(Dropout):
        def call(self, inputs):
            return super().call(inputs, training = True)
    
    def my_loss_fn(y_true, y_pred):
        return tf.reduce_mean(((y_pred*c_std+c_min)/(y_true*c_std+c_min)-1)**2)

    def create_model():
        model = Sequential()
        for i in range(NumHiddenLayers):
            model.add(LSTM(100, input_shape = (X_LSTM_treino.shape[1], X_LSTM_treino.shape[2])))
            model.add(Dense(Y_LSTM_treino.shape[1]))
 
    
        #model.compile(loss=my_loss_fn, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])
        model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])

        return model
    
    kfold = KFold(n_splits=n_splits, shuffle=True)

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (1)])
        return pred_stack
    
    Prediction = []
    j = 1

    for train, valid in kfold.split(X_LSTM_treino,Y_LSTM_treino):
        X_treino, X_valid = X_LSTM_treino[train], X_LSTM_treino[valid]
        Y_treino, Y_valid = Y_LSTM_treino[train], Y_LSTM_treino[valid]
        lstm_modelo = create_model()

        print("split",j)
        for i in range(epochs):  
            lstm_modelo.train_on_batch(X_LSTM_treino, Y_LSTM_treino)
            if verbose == 1:
                print("Epoch ",i)
                
        lstm_modelo.save("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Modelos_Treinados/Uni_LSTM_completa_"+submercado+"_"+str(j)+"_"+str(horizonte)+"_dias")
        j= j+1
        
        prediction_valid = predict(lstm_modelo,X_valid)
        prediction_valid = prediction_valid*c_std+c_min
        Y_valid = Y_valid*c_std+c_min

        media_valid = np.mean(prediction_valid, axis = 0)
        MAPE_LSTM_valid=(np.abs((media_valid-Y_valid)/Y_valid))*100
        MAPE_AVG_LSTM_valid=np.average(MAPE_LSTM_valid,axis=1)
        media_mape_valid = np.average(MAPE_AVG_LSTM_valid,axis=0)

        print("MAPE medio validacao kfold: " ,media_mape_valid)

        prediction=predict(lstm_modelo,X_LSTM_teste)
        Prediction.append(prediction)

    toc=timeit.default_timer()

    Prediction = np.asarray(Prediction)
    y_pred_Stack=Prediction.mean(axis=0)
    y_pred_stack=y_pred_Stack*c_std+c_min
    Y_LSTM_Teste = Y_LSTM_teste*c_std+c_min

    media = np.mean(y_pred_stack, axis = 0)

    MAPE_LSTM=np.abs((media-Y_LSTM_Teste)/Y_LSTM_Teste)
    MAPE_AVG_LSTM=np.average(MAPE_LSTM,axis=1)*100
    media_MAPE_LSTM = np.average(MAPE_AVG_LSTM)
       

    Y_LSTM_test_mean = np.mean(Y_LSTM_Teste,axis = 1)

    MSE = []
    nash_LSTM = []
    DM_LSTM = []
    for i in range(Y_LSTM_Teste.shape[0]):
        nash_LSTM_i=1-np.sum((Y_LSTM_Teste[i]-media[i])**2)/np.sum((Y_LSTM_Teste[i]-Y_LSTM_test_mean[i])**2)
        nash_LSTM.append(nash_LSTM_i)

        DM_LSTM_i = np.sqrt(((MAPE_AVG_LSTM[i]/100)**2)+(1-nash_LSTM[i])**2)
        DM_LSTM.append(DM_LSTM_i)

        MSE_i = np.sum((y_pred_Stack[0][i]-Y_LSTM_teste[i])**2)/Y_LSTM_teste.shape[1]
        MSE.append(MSE_i) 

    print("\nTempo de simulação:",toc-tic)
    print("MAPE médio:",media_MAPE_LSTM)

    data = data_teste_lstm    
    if graficos ==1:
        for i in range(8):
            plt.figure(figsize=(8,8))
            plt.plot(Y_LSTM_Teste[i],label='Y_test')
            plt.plot(media[i],label='Prediction')
            plt.title("Data: %s \n LSTM \nMAPE=%.2f    NASH= %.3f   DM= %.3f "%(data[i].strftime("%d/%m/%Y"),MAPE_AVG_LSTM[i] ,nash_LSTM[i], DM_LSTM[i]))
            plt.legend()
    
    resultados = {"mape":MAPE_LSTM,"nash": nash_LSTM, "mse":MSE}

    return media,Y_LSTM_Teste,data_teste_lstm,resultados

def Run_GRU_Kfold(submercado,inp, data_prev, var_norm,graficos,epochs,n_splits,verbose,horizonte):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    data_treino_lstm = data_prev["data_treino_lstm"]
    data_teste_lstm = data_prev["data_teste_lstm"]
        
    X_LSTM_treino = inp["X_LSTM_treino"]
    Y_LSTM_treino = inp["Y_LSTM_treino"]
    X_LSTM_teste = inp["X_LSTM_teste"]
    Y_LSTM_teste = inp["Y_LSTM_teste"]

    tic=timeit.default_timer()
    
    NumHiddenLayers=1
    NumNeurons=100
    beta=0.1
    VALIDATION_SPLIT=0.25
    learning_rate=1e-3
    
    class MCDropout(Dropout):
        def call(self, inputs):
            return super().call(inputs, training = True)
    
    def my_loss_fn(y_true, y_pred):
        return tf.reduce_mean(((y_pred*c_std+c_min)/(y_true*c_std+c_min)-1)**2)

    def create_model():
        model = Sequential()
        for i in range(NumHiddenLayers):
            model.add(Bidirectional(GRU(100), input_shape = (X_LSTM_treino.shape[1], X_LSTM_treino.shape[2])))
            model.add(Dense(Y_LSTM_treino.shape[1]))
 
    
        #model.compile(loss=my_loss_fn, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])
        model.compile(loss='MSE', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), metrics=["accuracy"])

        return model
    
    kfold = KFold(n_splits=n_splits, shuffle=True)

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (1000)])
        return pred_stack
    
    Prediction = []
    j = 1

    for train, valid in kfold.split(X_LSTM_treino,Y_LSTM_treino):
        X_treino, X_valid = X_LSTM_treino[train], X_LSTM_treino[valid]
        Y_treino, Y_valid = Y_LSTM_treino[train], Y_LSTM_treino[valid]
        lstm_modelo = create_model()

        print("split",j)
        for i in range(epochs):  
            lstm_modelo.train_on_batch(X_LSTM_treino, Y_LSTM_treino)
            if verbose == 1:
                print("Epoch ",i)
                
        lstm_modelo.save("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Modelos_Treinados_teste/GRU_completa_"+submercado+"_"+str(j)+"_"+str(horizonte)+"_dias")
        j= j+1

        prediction=predict(lstm_modelo,X_LSTM_teste)
        Prediction.append(prediction)

    toc=timeit.default_timer()

    Prediction = np.asarray(Prediction)
    y_pred_Stack=Prediction.mean(axis=0)
    y_pred_stack=y_pred_Stack*c_std+c_min
    Y_LSTM_Teste = Y_LSTM_teste*c_std+c_min

    media = np.mean(y_pred_stack, axis = 0)

    MAPE_LSTM=np.abs((media-Y_LSTM_Teste)/Y_LSTM_Teste)
    MAPE_AVG_LSTM=np.average(MAPE_LSTM,axis=1)*100
    media_MAPE_LSTM = np.average(MAPE_AVG_LSTM)
       

    Y_LSTM_test_mean = np.mean(Y_LSTM_Teste,axis = 1)

    MSE = []
    nash_LSTM = []
    DM_LSTM = []
    for i in range(Y_LSTM_Teste.shape[0]):
        nash_LSTM_i=1-np.sum((Y_LSTM_Teste[i]-media[i])**2)/np.sum((Y_LSTM_Teste[i]-Y_LSTM_test_mean[i])**2)
        nash_LSTM.append(nash_LSTM_i)

        DM_LSTM_i = np.sqrt(((MAPE_AVG_LSTM[i]/100)**2)+(1-nash_LSTM[i])**2)
        DM_LSTM.append(DM_LSTM_i)

        MSE_i = np.sum((y_pred_Stack[0][i]-Y_LSTM_teste[i])**2)/Y_LSTM_teste.shape[1]
        MSE.append(MSE_i) 

    print("\nTempo de simulação:",toc-tic)
    print("MAPE médio:",media_MAPE_LSTM)

    data = data_teste_lstm    
    if graficos ==1:
        for i in range(8):
            plt.figure(figsize=(8,8))
            plt.plot(Y_LSTM_Teste[i],label='Y_test')
            plt.plot(media[i],label='Prediction')
            plt.title("Data: %s \n LSTM \nMAPE=%.2f    NASH= %.3f   DM= %.3f "%(data[i].strftime("%d/%m/%Y"),MAPE_AVG_LSTM[i] ,nash_LSTM[i], DM_LSTM[i]))
            plt.legend()
    
    resultados = {"mape":MAPE_LSTM,"nash": nash_LSTM, "mse":MSE}

    return media,Y_LSTM_Teste,data_teste_lstm,resultados



def Run_MLP_Completa_diaria(submercado,inp, data_prev, var_norm,graficos,epochs,n_splits,NumHiddenLayers,verbose,horizonte):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    data_treino_MLP = data_prev["data_treino_MLP"]
    data_teste_MLP = data_prev["data_teste_MLP"]
        
    X_input_MLP_treino = inp["X_input_MLP_treino"]
    Y_input_MLP_treino = inp["Y_input_MLP_treino"]
    X_input_MLP_teste = inp["X_input_MLP_teste"]
    Y_input_MLP_teste = inp["Y_input_MLP_teste"]

    tic=timeit.default_timer()  
    NumNeurons=int(0.5*X_input_MLP_treino.shape[1])
    beta=0.1
    VALIDATION_SPLIT=0.25
    learning_rate=1e-3

    class MCDropout(Dropout):
        def call(self, inputs):
            return super().call(inputs, training = True)

    def my_loss_fn(y_true, y_pred):
        return tf.reduce_mean(((y_pred*c_std+c_min)/(y_true*c_std+c_min)-1)**2)


    def create_model():
        model = tf.keras.Sequential()
        for i in range(NumHiddenLayers):
            Layer=Dense(NumNeurons, kernel_initializer=keras.initializers.he_normal(seed=1), activation='relu')
            model.add(Layer)
            model.add(MCDropout(beta))


        OutputLayer=keras.layers.Dense(Y_input_MLP_treino.shape[1], kernel_initializer=keras.initializers.he_normal(seed=1), activation='relu')
        model.add(OutputLayer)

        loss_object = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad = True ), loss=my_loss_fn, metrics=["accuracy"])

        return model

    kfold = KFold(n_splits=n_splits, shuffle=True)

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (1000)])
        return pred_stack 

    Prediction = []
    j = 1
    for train, valid in kfold.split(X_input_MLP_treino,Y_input_MLP_treino):
        X_treino, X_valid = X_input_MLP_treino[train], X_input_MLP_treino[valid]
        Y_treino, Y_valid = Y_input_MLP_treino[train], Y_input_MLP_treino[valid]
        MLP_completa = create_model()
        
        #print('split',j)
        for i in range(epochs):  
            MLP_completa.train_on_batch(X_treino, Y_treino)
            if verbose == 1:
                print("Epoch ",i)
        
        #MLP_completa.save("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Modelos_Treinados/MLP_completa_"+submercado+"_"+str(j)+"_"+str(horizonte)+"_dias")
        
        j= j+1

        prediction=predict(MLP_completa,X_input_MLP_teste)
        Prediction.append(prediction)


    toc=timeit.default_timer()  

    Prediction = np.asarray(Prediction)
    y_pred_Stack=Prediction.mean(axis=0)
    y_pred_stack=y_pred_Stack*c_std+c_min
    Y_MLP_test = Y_input_MLP_teste*c_std+c_min
    
    media = np.mean(y_pred_stack, axis = 0)
    
    MAPE_MLP=np.abs((media-Y_MLP_test)/Y_MLP_test)
    MAPE_AVG_MLP=np.average(MAPE_MLP,axis=1)*100
    media_mape = np.average(MAPE_AVG_MLP,axis=0)
    
    print("Tempo de Simulação:",toc - tic)
    print("MAPE médio:",media_mape)
    
    data = data_teste_MLP
    
    resultados = {"mape":MAPE_MLP}
    return media,Y_MLP_test,data_teste_MLP,resultados