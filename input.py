import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU, RepeatVector, Dropout
import tensorflow as tf
from datetime import date
from datetime import datetime
from datetime import timedelta
import nbimporter
from sklearn.model_selection import KFold
import timeit
pd.set_option('mode.chained_assignment', None)

def input_MLP(tipo_input, dia_previsao,parametros_prev,var_norm, dados_enc,one_hot_enc,datas):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    tamanho_janela_MLP = parametros_prev["tamanho_janela_MLP"]
    horizonte_previsao_MLP = parametros_prev["horizonte_previsao_MLP"]
    horizonte_temp_MLP = parametros_prev["horizonte_temp_MLP"]

    NumExemplos_MLP= int(len(dados_enc)/24) - tamanho_janela_MLP - horizonte_previsao_MLP
    
    data_prev_MLP = []
    
    dados_Features_MLP = np.ones((int(len(dados_enc)/24),one_hot_enc.shape[1]),dtype = float)*-1
    for i in range(int(len(dados_enc)/24)):
        dados_Features_MLP[i] = one_hot_enc.loc[[i*24]]
        data_prev_MLP.append(datas.loc[i*24])

    NumFeatures_MLP = 50+24 +horizonte_temp_MLP*24 + 2*tamanho_janela_MLP*24
    x_MLP= np.ones((NumFeatures_MLP,NumExemplos_MLP) ,dtype=float)*-1
    y_MLP=np.ones((horizonte_previsao_MLP*24,NumExemplos_MLP) ,dtype=float)*-1
    data_previsao_MLP = []

    for i in range(NumExemplos_MLP):
        x_MLP[0:50, i] = dados_Features_MLP[i+1+tamanho_janela_MLP] #dados temporais codificados 

        x_MLP[50:(50+tamanho_janela_MLP*24),i] = dados_enc["carga"][(24*i):(24*(i+tamanho_janela_MLP))] #carga histórica (x dias atrás)
        x_MLP[(50+tamanho_janela_MLP*24):(50+2*tamanho_janela_MLP*24),i] = dados_enc["temp"][(24*i):(24*(i+tamanho_janela_MLP))] #temp histórica (x dias atrás)

        x_MLP[(50+2*tamanho_janela_MLP*24):(50+2*tamanho_janela_MLP*24+24+horizonte_previsao_MLP*24),i] = dados_enc["temp"][(24*(tamanho_janela_MLP+i)):(24*(i+tamanho_janela_MLP+horizonte_previsao_MLP)+24)]
        
        y_MLP[:,i] = dados_enc["carga"][(24*(tamanho_janela_MLP+i+1)):(24*(i+1+tamanho_janela_MLP+horizonte_previsao_MLP))]
        data_previsao_MLP.append(data_prev_MLP[tamanho_janela_MLP+i+1]) 
        
    XY_MLP=np.concatenate((x_MLP, y_MLP), axis=0)
    d=np.asarray(data_previsao_MLP)
    d = np.reshape(d,(len(d),1))
    XY_Data = np.concatenate((XY_MLP.T,d),axis = 1)
    
    tamanho_treino_MLP = NumExemplos_MLP - 10
    tamanho_teste_MLP = NumExemplos_MLP - tamanho_treino_MLP 
    treino,teste = XY_Data[0:tamanho_treino_MLP], XY_Data[tamanho_treino_MLP:NumExemplos_MLP]
    
    np.random.shuffle(treino)
    #print(treino.shape,teste.shape)
    
    X_treino= treino[:,0:NumFeatures_MLP]
    Y_treino= treino[:,NumFeatures_MLP:(NumFeatures_MLP + horizonte_previsao_MLP*24)]
    data_treino_MLP = treino[:,(NumFeatures_MLP + horizonte_previsao_MLP*24):]
     
    X_teste= teste[:,0:NumFeatures_MLP]
    Y_teste= teste[:,NumFeatures_MLP:(NumFeatures_MLP + horizonte_previsao_MLP*24)]
    data_teste_MLP = teste[:,(NumFeatures_MLP + horizonte_previsao_MLP*24):]
    
    
    aux_X_treino = X_treino.tolist()
    aux_Y_treino = Y_treino.tolist()
    aux_X_teste = X_teste.tolist()
    aux_Y_teste = Y_teste.tolist()
    
    X_MLP_treino= np.ones((tamanho_treino_MLP,NumFeatures_MLP) ,dtype=float)*-1
    Y_MLP_treino=np.ones((tamanho_treino_MLP,horizonte_previsao_MLP*24) ,dtype=float)*-1
    
    X_MLP_teste= np.ones((tamanho_teste_MLP,NumFeatures_MLP) ,dtype=float)*-1
    Y_MLP_teste=np.ones((tamanho_teste_MLP,horizonte_previsao_MLP*24) ,dtype=float)*-1
    
    for i in range(tamanho_treino_MLP):
        X_MLP_treino[i] = aux_X_treino[i]
        Y_MLP_treino[i]= aux_Y_treino[i]
    
    for i in range(tamanho_teste_MLP):
        X_MLP_teste[i] = aux_X_teste[i]
        Y_MLP_teste[i] = aux_Y_teste[i]
    
    datas_prev = {"data_treino_MLP":data_treino_MLP,"data_teste_MLP":data_teste_MLP}
      
    if tipo_input == "semanal":
        X_input_MLP_treino = X_MLP_treino
        Y_input_MLP_treino = Y_MLP_treino
        X_input_MLP_teste = X_MLP_teste
        Y_input_MLP_teste = Y_MLP_teste

        
    input_MLP = {"X_input_MLP_treino":X_input_MLP_treino,"Y_input_MLP_treino":Y_input_MLP_treino,"X_input_MLP_teste":X_input_MLP_teste,
                "Y_input_MLP_teste":Y_input_MLP_teste}   
    
   # print(X_input_MLP_treino.shape,Y_input_MLP_treino.shape,X_input_MLP_teste.shape,Y_input_MLP_teste.shape)
        
    return input_MLP, datas_prev

def input_LSTM(parametros_prev,var_norm, dados_enc,one_hot_enc,datas):

    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]

    tamanho_janela_LSTM = parametros_prev["tamanho_janela_LSTM"]
    horizonte_previsao_LSTM = parametros_prev["horizonte_previsao_LSTM"]
    horizonte_temp_LSTM = parametros_prev["horizonte_temp_LSTM"]
    time_steps = parametros_prev["time_steps"]
    
    NumExemplos_LSTM= int(len(dados_enc)/24) - tamanho_janela_LSTM -horizonte_previsao_LSTM
    NumDias_LSTM= int(len(dados_enc)/24)
    
    data_prev_LSTM = []
    dados_Features_LSTM = np.ones((int(len(dados_enc)/24),one_hot_enc.shape[1]),dtype = float)*-1
    for i in range(int(len(dados_enc)/24)):
        dados_Features_LSTM[i] = one_hot_enc.loc[[i*24]]
        data_prev_LSTM.append(datas.loc[i*24])
        
    NumFeatures_LSTM = 50+24+horizonte_temp_LSTM*24 + 2*tamanho_janela_LSTM*24
    x_LSTM= np.ones((NumFeatures_LSTM,NumExemplos_LSTM) ,dtype=float)*-1
    y_LSTM=np.ones((horizonte_previsao_LSTM*24,NumExemplos_LSTM) ,dtype=float)*-1
    data_previsao_LSTM = []
    
    for i in range(NumExemplos_LSTM):
            
        x_LSTM[0:50, i] = dados_Features_LSTM[i+1+tamanho_janela_LSTM] #dados temporais codificados 

        x_LSTM[50:(50+tamanho_janela_LSTM*24),i] = dados_enc["carga"][(24*i):(24*(i+tamanho_janela_LSTM))] #carga histórica (x dias atrás)
        x_LSTM[(50+tamanho_janela_LSTM*24):(50+2*tamanho_janela_LSTM*24),i] = dados_enc["temp"][(24*i):(24*(i+tamanho_janela_LSTM))] #temp histórica (x dias atrás)

        x_LSTM[(50+2*tamanho_janela_LSTM*24):(50+2*tamanho_janela_LSTM*24+24+horizonte_previsao_LSTM*24),i] = dados_enc["temp"][(24*(tamanho_janela_LSTM+i)):(24*(i+tamanho_janela_LSTM+horizonte_previsao_LSTM)+24)]
    
        y_LSTM[:,i] = dados_enc["carga"][(24*(tamanho_janela_LSTM+i+1)):(24*(i+1+tamanho_janela_LSTM+horizonte_previsao_LSTM))]
        data_previsao_LSTM.append(data_prev_LSTM[tamanho_janela_LSTM+i+1]) 

    xy_LSTM = np.concatenate((x_LSTM,y_LSTM),axis = 0)
    lista = []
    for i in range(xy_LSTM.shape[0]):
        coluna = "col"+str(i)
        lista.append(coluna)
    frame = pd.DataFrame(xy_LSTM.T,columns= lista)

    tamanho_treino_LSTM = NumExemplos_LSTM - 10

    treino, teste = frame[0:tamanho_treino_LSTM], frame[tamanho_treino_LSTM:NumExemplos_LSTM]
    treino = treino.reset_index(drop=True)
    teste = teste.reset_index(drop=True)

    data_treino_LSTM = []
    data_teste_LSTM = []
    data_treino_LSTM,data_teste_LSTM = data_previsao_LSTM[0:tamanho_treino_LSTM],data_previsao_LSTM[tamanho_treino_LSTM:NumExemplos_LSTM]

    def gera_dataset(dataset, time_steps,horizonte_previsao):
        dataA, dataB = [], []
        for i in range(len(dataset)-time_steps+1):
            a = dataset.loc[i:(i+time_steps-1),:"col"+str(NumFeatures_LSTM-1)]
            dataA.append(a)
            b = dataset.loc[i+time_steps-1:i+time_steps-1,"col"+str(NumFeatures_LSTM):]
            dataB.append(b)
        return np.asarray(dataA, dtype=np.float32), np.asarray(dataB, dtype=np.float32)

    X_LSTM_treino, Y_LSTM_treino = gera_dataset(treino, time_steps=time_steps,horizonte_previsao=horizonte_previsao_LSTM)
    Y_LSTM_treino = Y_LSTM_treino.reshape((Y_LSTM_treino.shape[0],Y_LSTM_treino.shape[2]))
    X_LSTM_teste, Y_LSTM_teste = gera_dataset(teste, time_steps=time_steps,horizonte_previsao=horizonte_previsao_LSTM)
    Y_LSTM_teste = Y_LSTM_teste.reshape((Y_LSTM_teste.shape[0],Y_LSTM_teste.shape[2]))

# print(X_LSTM_treino.shape,Y_LSTM_treino.shape,X_LSTM_teste.shape,Y_LSTM_teste.shape)
    
    input_LSTM = {"X_LSTM_treino":X_LSTM_treino,"Y_LSTM_treino":Y_LSTM_treino,"X_LSTM_teste":X_LSTM_teste,"Y_LSTM_teste":Y_LSTM_teste}
    
    data_treino_lstm= []
    data_teste_lstm= []
    for i in range(X_LSTM_treino.shape[0]):
        data_treino_lstm_i = data_treino_LSTM[i+time_steps-1]
        data_treino_lstm.append(data_treino_lstm_i)

    for i in range(X_LSTM_teste.shape[0]):
        data_teste_lstm_i = data_teste_LSTM[i+time_steps-1]
        data_teste_lstm.append(data_teste_lstm_i)
    
    datas_prev = {"data_treino_lstm":data_treino_lstm,"data_teste_lstm":data_teste_lstm}
    
    return input_LSTM, datas_prev

def input_MLP_diario(variavel,tipo_input, dias_teste,parametros_prev,var_norm, dados_enc,one_hot_enc,datas):
    
    c_min=var_norm["c_min"]
    c_std = var_norm["c_std"]
    t_min=var_norm["t_min"]
    t_std = var_norm["t_std"]
    
    tamanho_janela_MLP = parametros_prev["tamanho_janela_MLP"]
    horizonte_previsao_MLP = parametros_prev["horizonte_previsao_MLP"]
    horizonte_temp_MLP = parametros_prev["horizonte_temp_MLP"]

    NumExemplos_MLP= int(len(dados_enc)) - tamanho_janela_MLP
    
    data_prev_MLP = []
    
    dados_Features_MLP = np.ones((int(len(dados_enc)),one_hot_enc.shape[1]),dtype = float)*-1
    for i in range(int(len(dados_enc))):
        dados_Features_MLP[i] = one_hot_enc.loc[[i]]
        data_prev_MLP.append(datas.loc[i])

    NumFeatures_MLP = 50 + 3*horizonte_temp_MLP + 6*tamanho_janela_MLP
    x_MLP= np.ones((NumFeatures_MLP,NumExemplos_MLP) ,dtype=float)*-1
    y_MLP=np.ones((horizonte_previsao_MLP,NumExemplos_MLP) ,dtype=float)*-1
    data_previsao_MLP = []

    for i in range(NumExemplos_MLP):
        x_MLP[0:50, i] = dados_Features_MLP[i+tamanho_janela_MLP] #dados temporais codificados 

        x_MLP[50:(50+tamanho_janela_MLP),i] = dados_enc["carga"][(i):((i+tamanho_janela_MLP))] #carga histórica (x dias atrás)
        x_MLP[(50+1*tamanho_janela_MLP):(50+2*tamanho_janela_MLP),i] = dados_enc["carga_max"][(i):((i+tamanho_janela_MLP))] #temp histórica (x dias atrás)
        x_MLP[(50+2*tamanho_janela_MLP):(50+3*tamanho_janela_MLP),i] = dados_enc["carga_min"][(i):((i+tamanho_janela_MLP))] 

        x_MLP[(50+3*tamanho_janela_MLP):(50+4*tamanho_janela_MLP),i] = dados_enc["temp"][(i):((i+tamanho_janela_MLP))] #temp histórica (x dias atrás)
        x_MLP[(50+4*tamanho_janela_MLP):(50+5*tamanho_janela_MLP),i] = dados_enc["temp_max"][(i):((i+tamanho_janela_MLP))] #temp max histórica (x dias atrás)
        x_MLP[(50+5*tamanho_janela_MLP):(50+6*tamanho_janela_MLP),i] = dados_enc["temp_min"][(i):((i+tamanho_janela_MLP))] #temp max histórica (x dias atrás)
        
        x_MLP[(50+6*tamanho_janela_MLP):(50+6*tamanho_janela_MLP+horizonte_previsao_MLP),i] = dados_enc["temp"][(tamanho_janela_MLP+i):(i+tamanho_janela_MLP+horizonte_previsao_MLP)]
        x_MLP[(50+6*tamanho_janela_MLP+horizonte_previsao_MLP):(50+6*tamanho_janela_MLP+2*horizonte_previsao_MLP),i] = dados_enc["temp_max"][(tamanho_janela_MLP+i):(i+tamanho_janela_MLP+horizonte_previsao_MLP)]
        x_MLP[(50+6*tamanho_janela_MLP+2*horizonte_previsao_MLP):(50+6*tamanho_janela_MLP+3*horizonte_previsao_MLP),i] = dados_enc["temp_min"][(tamanho_janela_MLP+i):(i+tamanho_janela_MLP+horizonte_previsao_MLP)]

        y_MLP[:,i] = dados_enc[variavel][((tamanho_janela_MLP+i)):((i+tamanho_janela_MLP+horizonte_previsao_MLP))]
        data_previsao_MLP.append(data_prev_MLP[tamanho_janela_MLP+i]) 
    
  
    XY_MLP=np.concatenate((x_MLP, y_MLP), axis=0)
    d=np.asarray(data_previsao_MLP)
    d = np.reshape(d,(len(d),1))
    XY_Data = np.concatenate((XY_MLP.T,d),axis = 1)
    
    tamanho_treino_MLP = NumExemplos_MLP - dias_teste
    tamanho_teste_MLP = NumExemplos_MLP - tamanho_treino_MLP 
    treino,teste = XY_Data[0:tamanho_treino_MLP], XY_Data[tamanho_treino_MLP:NumExemplos_MLP]
    
    np.random.shuffle(treino)
    #print(treino.shape,teste.shape)
    
    X_treino= treino[:,0:NumFeatures_MLP]
    Y_treino= treino[:,NumFeatures_MLP:(NumFeatures_MLP + horizonte_previsao_MLP)]
    data_treino_MLP = treino[:,(NumFeatures_MLP + horizonte_previsao_MLP):]
    
    X_teste= teste[:,0:NumFeatures_MLP]
    Y_teste= teste[:,NumFeatures_MLP:(NumFeatures_MLP + horizonte_previsao_MLP)]
    data_teste_MLP = teste[:,(NumFeatures_MLP + horizonte_previsao_MLP):]
    
    
    aux_X_treino = X_treino.tolist()
    aux_Y_treino = Y_treino.tolist()
    aux_X_teste = X_teste.tolist()
    aux_Y_teste = Y_teste.tolist()

    X_MLP_treino= np.ones((tamanho_treino_MLP,NumFeatures_MLP) ,dtype=float)*-1
    Y_MLP_treino=np.ones((tamanho_treino_MLP,horizonte_previsao_MLP) ,dtype=float)*-1
    
    X_MLP_teste= np.ones((tamanho_teste_MLP,NumFeatures_MLP) ,dtype=float)*-1
    Y_MLP_teste=np.ones((tamanho_teste_MLP,horizonte_previsao_MLP) ,dtype=float)*-1
    
    for i in range(tamanho_treino_MLP):
        X_MLP_treino[i] = aux_X_treino[i]
        Y_MLP_treino[i]= aux_Y_treino[i]
    
    for i in range(tamanho_teste_MLP):
        X_MLP_teste[i] = aux_X_teste[i]
        Y_MLP_teste[i] = aux_Y_teste[i]
    
    datas_prev = {"data_treino_MLP":data_treino_MLP,"data_teste_MLP":data_teste_MLP}
    
    
    if tipo_input == "semanal":
        X_input_MLP_treino = X_MLP_treino
        Y_input_MLP_treino = Y_MLP_treino
        X_input_MLP_teste = X_MLP_teste
        Y_input_MLP_teste = Y_MLP_teste

         
    input_MLP = {"X_input_MLP_treino":X_input_MLP_treino,"Y_input_MLP_treino":Y_input_MLP_treino,"X_input_MLP_teste":X_input_MLP_teste,
                "Y_input_MLP_teste":Y_input_MLP_teste}   
    
   # print(X_input_MLP_treino.shape,Y_input_MLP_treino.shape,X_input_MLP_teste.shape,Y_input_MLP_teste.shape)
        
    return input_MLP, datas_prev