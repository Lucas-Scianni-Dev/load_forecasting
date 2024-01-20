import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM,GRU, RepeatVector, Dropout
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from datetime import date
from datetime import datetime
from datetime import timedelta
import nbimporter
from sklearn.model_selection import KFold
import timeit
import os
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from Tratamento_dos_dados import Tratamento_dos_dados
from input import input_MLP, input_LSTM
from REDES import Run_MLP_Completa, Run_LSTM_Kfold, Run_Uni_LSTM_Kfold, Run_GRU_Kfold

parametros = sys.argv[1:]

submercado = parametros[0]
caminho_pasta = parametros[1]  

caminho_arquivos = caminho_pasta +'/DeckCorrigido/'
caminho_arquivos_antigos = caminho_pasta +'/DeckCorrigido_storage'
caminho_previsoes = caminho_pasta + '/Previsoes/'
caminho_verifica_data = caminho_pasta +'/DeckCorrigido/SECO'

arqs = os.listdir(caminho_verifica_data)
arq = arqs[0]
for i in range(len(arq)):
    if arq[i] == '2':
        n_i = i
        break
data = arq[n_i:n_i + 10]
ano_arquivo_existente, mes_arquivo_existente, dia_arquivo_existente = int(data[0:4]),int(data[5:7]), int(data[8:10])

data_ultima_atualizacao = date(ano_arquivo_existente,mes_arquivo_existente,dia_arquivo_existente)

arquivo_carga = pd.read_csv(caminho_arquivos + submercado+'/'+submercado.lower()+'_carga_deck_'+str(data_ultima_atualizacao)+'.csv',delimiter = ',',decimal = ',')
arquivo_temp_hist = pd.read_csv(caminho_arquivos + submercado+'/'+submercado.lower()+'_temp_deck_'+str(data_ultima_atualizacao)+'.csv', delimiter = ',',decimal = ',')
feriados = pd.read_csv(caminho_arquivos + submercado+'/'+submercado+'_'+str(data_ultima_atualizacao)+'_FERIADOS.csv', delimiter = ',',decimal = ',')

#montagem do input da rede MLP
tamanho_janela_MLP = 2 #dias
horizonte_previsao_MLP = 1 #dia
horizonte_temp_MLP= horizonte_previsao_MLP #dia

#montagem do input da rede LSTM
tamanho_janela_LSTM = tamanho_janela_MLP #dias
horizonte_previsao_LSTM = horizonte_previsao_MLP #dia
horizonte_temp_LSTM= horizonte_previsao_LSTM #dia

if (submercado == 'SECO') | (submercado == 'S'):
    time_steps = 5
else:
    time_steps = 2

dados_enc,one_hot_enc, datas,var_norm= Tratamento_dos_dados(arquivo_carga,arquivo_temp_hist,feriados,horizonte_previsao_MLP)

parametros_prev_LSTM = {"tamanho_janela_LSTM":tamanho_janela_LSTM,"horizonte_previsao_LSTM" :horizonte_previsao_LSTM,"horizonte_temp_LSTM" :horizonte_temp_LSTM,"time_steps":time_steps}
parametros_prev_MLP = {"tamanho_janela_MLP":tamanho_janela_MLP,"horizonte_previsao_MLP" :horizonte_previsao_MLP,"horizonte_temp_MLP" :horizonte_temp_MLP}
tipo_input = "semanal" #diario ou semanal 
dia_previsao = "terca"  #dia da semana a ser previsto (apenas para tipo diário)

inp_MLP,data_prev_MLP = input_MLP (tipo_input, dia_previsao,parametros_prev_MLP,var_norm, dados_enc,one_hot_enc,datas)
inp_LSTM,data_prev_LSTM = input_LSTM (parametros_prev_LSTM,var_norm, dados_enc,one_hot_enc,datas)

print('\ntreinando MLP...')
p_MLP_completa, y_MLP_completa, d_MLP_completa,resultados_MLP_completa = Run_MLP_Completa(submercado,inp_MLP, data_prev_MLP, var_norm,graficos = 0,epochs = 10000, n_splits = 10,NumHiddenLayers = 1,verbose = 0,horizonte = horizonte_previsao_MLP)
print('\ntreinando LSTM...')
p_LSTM_completa, y_LSTM_completa, d_LSTM_completa,resultados_LSTM_completa = Run_LSTM_Kfold(submercado,inp_LSTM, data_prev_LSTM, var_norm,graficos = 0,epochs = 7000, n_splits = 20,verbose = 0,horizonte = horizonte_previsao_LSTM)
print('\ntreinando GRU...')
p_GRU_completa, y_GRU_completa, d_GRU_completa,resultados_GRU_completa = Run_GRU_Kfold(submercado,inp_LSTM, data_prev_LSTM, var_norm,graficos = 0,epochs = 7000, n_splits = 20,verbose = 0,horizonte = horizonte_previsao_LSTM)