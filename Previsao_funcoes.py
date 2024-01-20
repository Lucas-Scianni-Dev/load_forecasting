import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime as dt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from datetime import date
from datetime import datetime
from datetime import timedelta
import tensorflow as tf
import timeit
import os
import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import plotly.graph_objects as go
import plotly.express as px

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
from Tratamento_dos_dados import Tratamento_dos_dados
import functools
print = functools.partial(print, flush=True)

def montar_conjuntos_previsao(dias_a_frente,atraso,feriado_futuros,caminho_pasta):
    hoje = (date.today() -timedelta(days=atraso)).isoformat()
    hoje_lstm = date.today() -timedelta(days=atraso)
    ontem = (date.today() -timedelta(days=atraso +1)).isoformat()
    data_previsao = ((date.today() -timedelta(days=atraso))+timedelta(days=dias_a_frente))

    caminho_arquivos = caminho_pasta +'DeckCorrigido/'
    caminho_previsao = caminho_pasta + 'Previsoes/'

    #2 DIAS DE JANELA / 5 TIME STEPS - S e SECO           #3 DIAS DE JANELA / 5 TIME STEPS - S e SECO
    #2 DIAS DE JANELA / 2 TIME STEPS - NE e N             #2 DIAS DE JANELA / 2 TIME STEPS - NE         3 DIAS DE JANELA / 2 TIME STEPS -N

    #montagem do input da rede MLP
    tamanho_janela_MLP_seco = 2 #dias
    tamanho_janela_MLP_s = 2 #dias
    tamanho_janela_MLP_ne = 2 #dias
    tamanho_janela_MLP_n = 2 #dias

    horizonte_previsao_MLP = 1 #dia
    horizonte_temp_MLP= horizonte_previsao_MLP #dia

    #montagem do input da rede LSTM
    tamanho_janela_LSTM_seco = tamanho_janela_MLP_seco #dias
    tamanho_janela_LSTM_s = tamanho_janela_MLP_s #dias
    tamanho_janela_LSTM_ne = tamanho_janela_MLP_ne #dias
    tamanho_janela_LSTM_n = tamanho_janela_MLP_n #dias

    horizonte_previsao_LSTM = horizonte_previsao_MLP #dia
    horizonte_temp_LSTM= horizonte_previsao_LSTM #dia

    time_steps_seco = 5
    time_steps_s = 5
    time_steps_ne = 2
    time_steps_n = 2

    modelo_previsao = 'LSTM' #MLP, LSTM OU GRU
    data_prevista = (date.today() -timedelta(days=atraso +horizonte_previsao_MLP)).isoformat()

    #####################################################################################################################################################################
    #Lê arquivos e trata os dados
    SECO_arquivo_carga = pd.read_csv(caminho_arquivos + 'SECO/seco_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    SECO_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'SECO/seco_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    SECO_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'SECO/seco_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    SECO_feriados = pd.read_csv(caminho_arquivos + 'SECO/SECO_'+hoje+'_FERIADOS.csv', delimiter = ',',decimal = ',')

    S_arquivo_carga = pd.read_csv(caminho_arquivos + 'S/s_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    S_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'S/s_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    S_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'S/s_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    S_feriados = pd.read_csv(caminho_arquivos + 'S/S_'+hoje+'_FERIADOS.csv', delimiter = ',')

    NE_arquivo_carga = pd.read_csv(caminho_arquivos + 'NE/ne_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    NE_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'NE/ne_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    NE_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'NE/ne_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    NE_feriados = pd.read_csv(caminho_arquivos + 'NE/NE_'+hoje+'_FERIADOS.csv', delimiter = ',')

    N_arquivo_carga = pd.read_csv(caminho_arquivos + 'N/n_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    N_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'N/n_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    N_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'N/n_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    N_feriados = pd.read_csv(caminho_arquivos + 'N/N_'+hoje+'_FERIADOS.csv', delimiter = ',',decimal = ',')

    seco_dados_enc,seco_one_hot_enc, seco_datas,seco_var_norm = Tratamento_dos_dados(SECO_arquivo_carga,SECO_arquivo_temp_hist,SECO_feriados,horizonte_previsao_MLP)
    s_dados_enc,s_one_hot_enc, s_datas,s_var_norm = Tratamento_dos_dados(S_arquivo_carga,S_arquivo_temp_hist,S_feriados,horizonte_previsao_MLP)
    ne_dados_enc,ne_one_hot_enc, ne_datas,ne_var_norm = Tratamento_dos_dados(NE_arquivo_carga,NE_arquivo_temp_hist,NE_feriados,horizonte_previsao_MLP)
    n_dados_enc,n_one_hot_enc, n_datas,n_var_norm = Tratamento_dos_dados(N_arquivo_carga,N_arquivo_temp_hist,N_feriados,horizonte_previsao_MLP)

    seco_c_min,seco_c_std,seco_t_min,seco_t_std =seco_var_norm["c_min"],seco_var_norm["c_std"],seco_var_norm["t_min"],seco_var_norm["t_std"]
    s_c_min,s_c_std,s_t_min,s_t_std =s_var_norm["c_min"],s_var_norm["c_std"],s_var_norm["t_min"],s_var_norm["t_std"]
    ne_c_min,ne_c_std,ne_t_min,ne_t_std =ne_var_norm["c_min"],ne_var_norm["c_std"],ne_var_norm["t_min"],ne_var_norm["t_std"]
    n_c_min,n_c_std,n_t_min,n_t_std =n_var_norm["c_min"],n_var_norm["c_std"],n_var_norm["t_min"],n_var_norm["t_std"]

    seco_arquivo_temp_prevista = (SECO_arquivo_temp_prevista["Temperatura"]-seco_t_min)/seco_t_std
    s_arquivo_temp_prevista = (S_arquivo_temp_prevista["Temperatura"]-s_t_min)/s_t_std
    ne_arquivo_temp_prevista = (NE_arquivo_temp_prevista["Temperatura"]-ne_t_min)/ne_t_std
    n_arquivo_temp_prevista = (N_arquivo_temp_prevista["Temperatura"]-n_t_min)/n_t_std

    seco_carga_enc,s_carga_enc,ne_carga_enc,n_carga_enc = seco_dados_enc["carga"],s_dados_enc["carga"],ne_dados_enc["carga"],n_dados_enc["carga"]
    seco_temp_enc,s_temp_enc,ne_temp_enc,n_temp_enc = seco_dados_enc["temp"],s_dados_enc["temp"],ne_dados_enc["temp"],n_dados_enc["temp"]

    ###############################################################################################################################################################    
    if dias_a_frente > 1:
        prevs_seco = []
        prevs_s = []
        prevs_ne = []
        prevs_n = []
        for i in range(dias_a_frente - 1):
            prev_arq_data = ((date.today() -timedelta(days=atraso))+timedelta(days=i)).isoformat()
            if i == 0:
                prev_arq_data_anterior = (date.today() -timedelta(days=atraso+1))
            else:
                prev_arq_data_anterior = (date.today() -timedelta(days=atraso))

            try:
                previsao_passada = pd.read_csv(caminho_pasta + "Previsoes/"+str(prev_arq_data_anterior.year)+'/'+ '%02d' % prev_arq_data_anterior.month+'/'+str(prev_arq_data_anterior.isoformat())+'/daily/'+prev_arq_data+'.csv',delimiter = ';',decimal = ',')
            except:
                previsao_passada = pd.read_csv(caminho_pasta + "Previsoes/"+str((prev_arq_data_anterior - timedelta(1)).year)+'/'+'%02d' %((prev_arq_data_anterior - timedelta(1)).month)+'/'+str((prev_arq_data_anterior - timedelta(1)).isoformat())+'/daily/'+prev_arq_data+'.csv',delimiter = ';',decimal = ',')

            prev_seco = (previsao_passada[modelo_previsao+'_Seco']-seco_c_min)/seco_c_std
            prev_s = (previsao_passada[modelo_previsao+'_S']-s_c_min)/s_c_std
            prev_ne = (previsao_passada[modelo_previsao+'_NE']-ne_c_min)/ne_c_std
            prev_n = (previsao_passada[modelo_previsao+'_N']-n_c_min)/n_c_std

            for i in range(len(prev_seco)):
                prevs_seco.append(prev_seco[i])
                prevs_s.append(prev_s[i])
                prevs_ne.append(prev_ne[i])
                prevs_n.append(prev_n[i])
            #print(prev_aux)

        previsoes_seco = []
        previsoes_s = []
        previsoes_ne = []
        previsoes_n = []
        j = 0
        for i in range(int(len(prevs_seco)/2)):
            previsoes_seco.append(prevs_seco[j])
            previsoes_s.append(prevs_s[j])
            previsoes_ne.append(prevs_ne[j])
            previsoes_n.append(prevs_n[j])
            j = j+2
            
        previsoes_seco = pd.Series(previsoes_seco)
        previsoes_s = pd.Series(previsoes_s)
        previsoes_ne = pd.Series(previsoes_ne)
        previsoes_n = pd.Series(previsoes_n)

        seco_temp_enc = seco_temp_enc.append(seco_arquivo_temp_prevista[:(dias_a_frente-1)*24],ignore_index=True)
        s_temp_enc = s_temp_enc.append(s_arquivo_temp_prevista[:(dias_a_frente-1)*24],ignore_index=True)
        ne_temp_enc = ne_temp_enc.append(ne_arquivo_temp_prevista[:(dias_a_frente-1)*24],ignore_index=True)
        n_temp_enc = n_temp_enc.append(n_arquivo_temp_prevista[:(dias_a_frente-1)*24],ignore_index=True)

        seco_arquivo_temp_prevista = seco_arquivo_temp_prevista.loc[((dias_a_frente-1)*24):((dias_a_frente-1)*24)+47].reset_index(drop = True)
        s_arquivo_temp_prevista = s_arquivo_temp_prevista.loc[((dias_a_frente-1)*24):((dias_a_frente-1)*24)+47].reset_index(drop = True)
        ne_arquivo_temp_prevista = ne_arquivo_temp_prevista.loc[((dias_a_frente-1)*24):((dias_a_frente-1)*24)+47].reset_index(drop = True)
        n_arquivo_temp_prevista = n_arquivo_temp_prevista.loc[((dias_a_frente-1)*24):((dias_a_frente-1)*24)+47].reset_index(drop = True)

        seco_carga_enc = seco_carga_enc.append(previsoes_seco,ignore_index=True)
        s_carga_enc = s_carga_enc.append(previsoes_s,ignore_index=True)
        ne_carga_enc = ne_carga_enc.append(previsoes_ne,ignore_index=True)
        n_carga_enc = n_carga_enc.append(previsoes_n,ignore_index=True)

    #########################################################################################################################################################
    seco_carga_hist_MLP = seco_carga_enc[((len(seco_carga_enc)-tamanho_janela_MLP_seco*24)):]
    seco_temp_hist_MLP = seco_temp_enc[(len(seco_temp_enc)-tamanho_janela_MLP_seco*24):]
    seco_temp_prevista_MLP= seco_arquivo_temp_prevista[:48]

    seco_carga_hist_LSTM = seco_carga_enc[((len(seco_carga_enc)-(time_steps_seco+tamanho_janela_LSTM_seco)*24+24)):]
    seco_temp_hist_LSTM = seco_temp_enc[(len(seco_temp_enc)-(time_steps_seco+tamanho_janela_LSTM_seco)*24+24):]
    seco_temp_prevista_LSTM = seco_temp_hist_LSTM[len(seco_temp_hist_LSTM)-(time_steps_seco-1)*24:]
    seco_temp_prevista_LSTM = seco_temp_prevista_LSTM.append(seco_arquivo_temp_prevista[:48],ignore_index = True)
    #########################################################################################################################################################
    s_carga_hist_MLP = s_carga_enc[(len(s_carga_enc)-tamanho_janela_MLP_s*24):]
    s_temp_hist_MLP = s_temp_enc[(len(s_temp_enc)-tamanho_janela_MLP_s*24):]
    s_temp_prevista_MLP= s_arquivo_temp_prevista[:48]

    s_carga_hist_LSTM = s_carga_enc[((len(s_datas)-(time_steps_s+tamanho_janela_LSTM_s)*24+24)):]
    s_temp_hist_LSTM = s_temp_enc[(len(s_datas)-(time_steps_s+tamanho_janela_LSTM_s)*24+24):]
    s_temp_prevista_LSTM = s_temp_hist_LSTM[len(s_temp_hist_LSTM)-(time_steps_s-1)*24:]
    s_temp_prevista_LSTM = s_temp_prevista_LSTM.append(s_arquivo_temp_prevista[:48],ignore_index = True)
    ##########################################################################################################################################################
    ne_carga_hist_MLP = ne_carga_enc[(len(ne_carga_enc)-tamanho_janela_MLP_ne*24):]
    ne_temp_hist_MLP = ne_temp_enc[(len(ne_temp_enc)-tamanho_janela_MLP_ne*24):]
    ne_temp_prevista_MLP= ne_arquivo_temp_prevista[:48]

    ne_carga_hist_LSTM = ne_carga_enc[((len(ne_datas)-(time_steps_ne+tamanho_janela_LSTM_ne)*24+24)):]
    ne_temp_hist_LSTM = ne_temp_enc[(len(ne_datas)-(time_steps_ne+tamanho_janela_LSTM_ne)*24+24):]
    ne_temp_prevista_LSTM = ne_temp_hist_LSTM[len(ne_temp_hist_LSTM)-(time_steps_ne-1)*24:]
    ne_temp_prevista_LSTM = ne_temp_prevista_LSTM.append(ne_arquivo_temp_prevista[:48],ignore_index = True)
    ########################################################################################################################################################
    n_carga_hist_MLP = n_carga_enc[(len(n_carga_enc)-tamanho_janela_MLP_n*24):]
    n_temp_hist_MLP = n_temp_enc[(len(n_temp_enc)-tamanho_janela_MLP_n*24):]
    n_temp_prevista_MLP= n_arquivo_temp_prevista[:48]

    n_carga_hist_LSTM = n_carga_enc[((len(n_datas)-(time_steps_n+tamanho_janela_LSTM_n)*24+24)):]
    n_temp_hist_LSTM = n_temp_enc[(len(n_datas)-(time_steps_n+tamanho_janela_LSTM_n)*24+24):]
    n_temp_prevista_LSTM = n_temp_hist_LSTM[len(n_temp_hist_LSTM)-(time_steps_n-1)*24:]
    n_temp_prevista_LSTM = n_temp_prevista_LSTM.append(n_arquivo_temp_prevista[:48],ignore_index = True)

    data_feriados_SECO, data_feriados_S, data_feriados_NE, data_feriados_N = [],[],[],[]
    for i in range(len(SECO_feriados["Dia"])):
        Data_i=dt.date(int(SECO_feriados["Ano"][i]), int(SECO_feriados["Mes"][i]), int(SECO_feriados["Dia"][i]))
        data_feriados_SECO.append(Data_i)
    for i in range(len(S_feriados["Dia"])):
        Data_i=dt.date(int(S_feriados["Ano"][i]), int(S_feriados["Mes"][i]), int(S_feriados["Dia"][i]))
        data_feriados_S.append(Data_i)
    for i in range(len(NE_feriados["Dia"])):
        Data_i=dt.date(int(NE_feriados["Ano"][i]), int(NE_feriados["Mes"][i]), int(NE_feriados["Dia"][i]))
        data_feriados_NE.append(Data_i)
    for i in range(len(N_feriados["Dia"])):
        Data_i=dt.date(int(N_feriados["Ano"][i]), int(N_feriados["Mes"][i]), int(N_feriados["Dia"][i]))
        data_feriados_N.append(Data_i)

    var_feriado_SECO, var_feriado_S, var_feriado_NE, var_feriado_N = 0,0,0,0
    for i in range(len(SECO_feriados["Dia"])):
        if data_previsao == data_feriados_SECO[i]:
            var_feriado_SECO = 1
    for i in range(len(S_feriados["Dia"])):
        if data_previsao == data_feriados_S[i]:
            var_feriado_S = 1
    for i in range(len(NE_feriados["Dia"])):
        if data_previsao == data_feriados_NE[i]:
            var_feriado_NE = 1
    for i in range(len(N_feriados["Dia"])):
        if data_previsao == data_feriados_N[i]:
            var_feriado_N = 1

    if feriado_futuros[dias_a_frente] == 1:
        var_feriado_SECO = 1
        var_feriado_S = 1 
        var_feriado_NE = 1  
        var_feriado_N = 1
    
    def One_Hot(mes, dia_semana,dia_mes,feriado):

            dia_semana_enc=np.zeros(7,dtype=float)

            if feriado == 1:
                dia_semana_enc[6] = 1
            else:
                dia_semana_enc[dia_semana]=1

            dia_mes_enc=np.zeros(31,dtype=float)
            dia_mes_enc[dia_mes-1]=1

            mes_enc = np.zeros(12,dtype = float)
            mes_enc[mes-1] =1 

            OneHot_Encoded=np.concatenate((dia_semana_enc,mes_enc,dia_mes_enc))

            return OneHot_Encoded

    one_hot_enc_previsao_SECO = []
    one_hot_enc_previsao_S = []
    one_hot_enc_previsao_NE = []
    one_hot_enc_previsao_N = []

    mes_previsao = data_previsao.month 
    dia_semana_previsao = data_previsao.weekday()
    dia_mes_previsao = data_previsao.day

    oneHot_SECO = One_Hot(mes_previsao,dia_semana_previsao,dia_mes_previsao,var_feriado_SECO)
    oneHot_S = One_Hot(mes_previsao,dia_semana_previsao,dia_mes_previsao,var_feriado_S)
    oneHot_NE = One_Hot(mes_previsao,dia_semana_previsao,dia_mes_previsao,var_feriado_NE)
    oneHot_N = One_Hot(mes_previsao,dia_semana_previsao,dia_mes_previsao,var_feriado_N)

    encoded_datas_passadas_seco = np.ones((int(len(seco_dados_enc)/24),50),dtype = float)*-1
    encoded_datas_passadas_s = np.ones((int(len(s_dados_enc)/24),50),dtype = float)*-1
    encoded_datas_passadas_ne = np.ones((int(len(ne_dados_enc)/24),50),dtype = float)*-1
    encoded_datas_passadas_n = np.ones((int(len(n_dados_enc)/24),50),dtype = float)*-1

    for i in range(int(len(seco_dados_enc)/24)):
        encoded_datas_passadas_seco[i] = seco_one_hot_enc.loc[[i*24]]
    for i in range(int(len(s_dados_enc)/24)):
        encoded_datas_passadas_s[i] = s_one_hot_enc.loc[[i*24]]
    for i in range(int(len(ne_dados_enc)/24)):
        encoded_datas_passadas_ne[i] = ne_one_hot_enc.loc[[i*24]]
    for i in range(int(len(n_dados_enc)/24)):
        encoded_datas_passadas_n[i] = n_one_hot_enc.loc[[i*24]]

    one_hot_futuro = []
    for i in range(dias_a_frente):
        data_one_hot = ((date.today() -timedelta(days=atraso))+timedelta(days=i))
        mes_one_hot = data_one_hot.month 
        dia_semana_one_hot = data_one_hot.weekday()
        dia_mes_one_hot = data_one_hot.day

        feriado = 0
        if feriado_futuros[i] == 1:
            feriado = 1
        one_hot_data_i = One_Hot(mes_one_hot,dia_semana_one_hot,dia_mes_one_hot,feriado)     
        one_hot_futuro.append(one_hot_data_i)

    one_hot_futuro.append(oneHot_SECO)

    encoded_datas_passadas_seco = pd.DataFrame(encoded_datas_passadas_seco)
    encoded_datas_passadas_s = pd.DataFrame(encoded_datas_passadas_s)
    encoded_datas_passadas_ne = pd.DataFrame(encoded_datas_passadas_ne)
    encoded_datas_passadas_n = pd.DataFrame(encoded_datas_passadas_n)

    encoded_datas_seco = encoded_datas_passadas_seco.append(one_hot_futuro).reset_index(drop = True)
    encoded_datas_s = encoded_datas_passadas_s.append(one_hot_futuro).reset_index(drop = True)
    encoded_datas_ne = encoded_datas_passadas_ne.append(one_hot_futuro).reset_index(drop = True)
    encoded_datas_n = encoded_datas_passadas_n.append(one_hot_futuro).reset_index(drop = True)
    ########################################################################################################################################################

    data_feature_LSTM_seco = encoded_datas_seco[len(encoded_datas_seco)-time_steps_seco:].reset_index(drop = True)
    data_feature_LSTM_s = encoded_datas_s[len(encoded_datas_s)-time_steps_s:].reset_index(drop = True)
    data_feature_LSTM_ne = encoded_datas_ne[len(encoded_datas_ne)-time_steps_ne:].reset_index(drop = True)
    data_feature_LSTM_n = encoded_datas_n[len(encoded_datas_n)-time_steps_n:] .reset_index(drop = True)

    NumFeatures_MLP_seco = 50+24 + horizonte_temp_MLP*24 + 2*tamanho_janela_MLP_seco*24
    NumFeatures_MLP_s = 50+24 + horizonte_temp_MLP*24 + 2*tamanho_janela_MLP_s*24
    NumFeatures_MLP_ne = 50+24 + horizonte_temp_MLP*24 + 2*tamanho_janela_MLP_ne*24
    NumFeatures_MLP_n = 50+24 + horizonte_temp_MLP*24 + 2*tamanho_janela_MLP_n*24

    seco_conjunto_Previsao = np.ones((1,NumFeatures_MLP_seco) ,dtype=float)*-1
    seco_conjunto_Previsao[0,0:50] = encoded_datas_seco.loc[len(encoded_datas_seco)-1]
    seco_conjunto_Previsao[0,50:(50+tamanho_janela_MLP_seco*24)] = seco_carga_hist_MLP.T
    seco_conjunto_Previsao[0,(50+tamanho_janela_MLP_seco*24):50+(2*tamanho_janela_MLP_seco*24)] = seco_temp_hist_MLP.T
    seco_conjunto_Previsao[0,(50+2*tamanho_janela_MLP_seco*24):] = seco_temp_prevista_MLP.T

    s_conjunto_Previsao = np.ones((1,NumFeatures_MLP_s) ,dtype=float)*-1
    s_conjunto_Previsao[0,0:50] = encoded_datas_s.loc[len(encoded_datas_s)-1]
    s_conjunto_Previsao[0,50:(50+tamanho_janela_MLP_s*24)] = s_carga_hist_MLP.T
    s_conjunto_Previsao[0,(50+tamanho_janela_MLP_s*24):50+(2*tamanho_janela_MLP_s*24)] = s_temp_hist_MLP.T
    s_conjunto_Previsao[0,(50+2*tamanho_janela_MLP_s*24):] = s_temp_prevista_MLP.T

    ne_conjunto_Previsao = np.ones((1,NumFeatures_MLP_ne) ,dtype=float)*-1
    ne_conjunto_Previsao[0,0:50] = encoded_datas_ne.loc[len(encoded_datas_ne)-1]
    ne_conjunto_Previsao[0,50:(50+tamanho_janela_MLP_ne*24)] = ne_carga_hist_MLP.T
    ne_conjunto_Previsao[0,(50+tamanho_janela_MLP_ne*24):50+(2*tamanho_janela_MLP_ne*24)] = ne_temp_hist_MLP.T
    ne_conjunto_Previsao[0,(50+2*tamanho_janela_MLP_ne*24):] = ne_temp_prevista_MLP.T

    n_conjunto_Previsao = np.ones((1,NumFeatures_MLP_n) ,dtype=float)*-1
    n_conjunto_Previsao[0,0:50] = encoded_datas_n.loc[len(encoded_datas_n)-1]
    n_conjunto_Previsao[0,50:(50+tamanho_janela_MLP_n*24)] = n_carga_hist_MLP.T
    n_conjunto_Previsao[0,(50+tamanho_janela_MLP_n*24):50+(2*tamanho_janela_MLP_n*24)] = n_temp_hist_MLP.T
    n_conjunto_Previsao[0,(50+2*tamanho_janela_MLP_n*24):] = n_temp_prevista_MLP.T

    NumFeatures_LSTM_seco = 50+24 +horizonte_temp_LSTM*24 + 2*tamanho_janela_LSTM_seco*24
    NumFeatures_LSTM_s = 50+24 +horizonte_temp_LSTM*24 + 2*tamanho_janela_LSTM_s*24
    NumFeatures_LSTM_ne = 50+24 +horizonte_temp_LSTM*24 + 2*tamanho_janela_LSTM_ne*24
    NumFeatures_LSTM_n = 50+24 +horizonte_temp_LSTM*24 + 2*tamanho_janela_LSTM_n*24

    seco_conjunto_Previsao_LSTM= np.ones((time_steps_seco,NumFeatures_LSTM_seco) ,dtype=float)*-1
    s_conjunto_Previsao_LSTM= np.ones((time_steps_s,NumFeatures_LSTM_s) ,dtype=float)*-1
    ne_conjunto_Previsao_LSTM= np.ones((time_steps_ne,NumFeatures_LSTM_ne) ,dtype=float)*-1
    n_conjunto_Previsao_LSTM= np.ones((time_steps_n,NumFeatures_LSTM_n) ,dtype=float)*-1

    for i in range(time_steps_seco):
        seco_conjunto_Previsao_LSTM[i,0:50] = data_feature_LSTM_seco.loc[i]
        seco_conjunto_Previsao_LSTM[i,50:50+tamanho_janela_LSTM_seco*24] = seco_carga_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_seco))]
        seco_conjunto_Previsao_LSTM[i,50+tamanho_janela_LSTM_seco*24:50+(2*tamanho_janela_LSTM_seco*24)] = seco_temp_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_seco))]
        seco_conjunto_Previsao_LSTM[i,(50+2*tamanho_janela_LSTM_seco*24):(50+2*tamanho_janela_LSTM_seco*24+24*horizonte_temp_LSTM+24)] = seco_temp_prevista_LSTM[(24*i):(24*(i+1+horizonte_temp_LSTM))]
    for i in range(time_steps_s):
        s_conjunto_Previsao_LSTM[i,0:50] = data_feature_LSTM_s.loc[i]
        s_conjunto_Previsao_LSTM[i,50:50+tamanho_janela_LSTM_s*24] = s_carga_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_s))]
        s_conjunto_Previsao_LSTM[i,50+tamanho_janela_LSTM_s*24:50+(2*tamanho_janela_LSTM_s*24)] = s_temp_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_s))]
        s_conjunto_Previsao_LSTM[i,(50+2*tamanho_janela_LSTM_s*24):(50+2*tamanho_janela_LSTM_s*24+24*horizonte_temp_LSTM+24)] = s_temp_prevista_LSTM[(24*i):(24*(i+1+horizonte_temp_LSTM))]
    for i in range(time_steps_ne):
        ne_conjunto_Previsao_LSTM[i,0:50] = data_feature_LSTM_ne.loc[i]
        ne_conjunto_Previsao_LSTM[i,50:50+tamanho_janela_LSTM_ne*24] = ne_carga_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_ne))]
        ne_conjunto_Previsao_LSTM[i,50+tamanho_janela_LSTM_ne*24:50+(2*tamanho_janela_LSTM_ne*24)] = ne_temp_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_ne))]
        ne_conjunto_Previsao_LSTM[i,(50+2*tamanho_janela_LSTM_ne*24):(50+2*tamanho_janela_LSTM_ne*24+24*horizonte_temp_LSTM+24)] = ne_temp_prevista_LSTM[(24*i):(24*(i+1+horizonte_temp_LSTM))]
    for i in range(time_steps_n):
        n_conjunto_Previsao_LSTM[i,0:50] = data_feature_LSTM_n.loc[i]
        n_conjunto_Previsao_LSTM[i,50:50+tamanho_janela_LSTM_n*24] = n_carga_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_n))]
        n_conjunto_Previsao_LSTM[i,50+tamanho_janela_LSTM_n*24:50+(2*tamanho_janela_LSTM_n*24)] = n_temp_hist_LSTM[(24*i):(24*(i+tamanho_janela_LSTM_n))]
        n_conjunto_Previsao_LSTM[i,(50+2*tamanho_janela_LSTM_n*24):(50+2*tamanho_janela_LSTM_n*24+24*horizonte_temp_LSTM+24)] = n_temp_prevista_LSTM[(24*i):(24*(i+1+horizonte_temp_LSTM))]

        #print("steptime",i+1,"montado!")

    #seco_conjunto_Previsao_LSTM = seco_conjunto_Previsao_LSTM[::-1]
    #s_conjunto_Previsao_LSTM = s_conjunto_Previsao_LSTM[::-1]
    #ne_conjunto_Previsao_LSTM = ne_conjunto_Previsao_LSTM[::-1]
    #n_conjunto_Previsao_LSTM = n_conjunto_Previsao_LSTM[::-1]

    seco_conjunto_Previsao_LSTM = np.reshape(seco_conjunto_Previsao_LSTM,(1,seco_conjunto_Previsao_LSTM.shape[0],seco_conjunto_Previsao_LSTM.shape[1]))
    s_conjunto_Previsao_LSTM = np.reshape(s_conjunto_Previsao_LSTM,(1,s_conjunto_Previsao_LSTM.shape[0],s_conjunto_Previsao_LSTM.shape[1]))
    ne_conjunto_Previsao_LSTM = np.reshape(ne_conjunto_Previsao_LSTM,(1,ne_conjunto_Previsao_LSTM.shape[0],ne_conjunto_Previsao_LSTM.shape[1]))
    n_conjunto_Previsao_LSTM = np.reshape(n_conjunto_Previsao_LSTM,(1,n_conjunto_Previsao_LSTM.shape[0],n_conjunto_Previsao_LSTM.shape[1]))

    conjuntos_mlp = {'seco_conjunto_Previsao':seco_conjunto_Previsao, 's_conjunto_Previsao':s_conjunto_Previsao,'ne_conjunto_Previsao':ne_conjunto_Previsao,'n_conjunto_Previsao':n_conjunto_Previsao}
    conjuntos_lstm = {'seco_conjunto_Previsao_LSTM':seco_conjunto_Previsao_LSTM,'s_conjunto_Previsao_LSTM':s_conjunto_Previsao_LSTM,'ne_conjunto_Previsao_LSTM':ne_conjunto_Previsao_LSTM,'n_conjunto_Previsao_LSTM':n_conjunto_Previsao_LSTM}

    var_norm = {'seco_var_norm':seco_var_norm,'s_var_norm':s_var_norm,'ne_var_norm':ne_var_norm,'n_var_norm':n_var_norm}
    return conjuntos_mlp,conjuntos_lstm,var_norm

def prever(atraso,dias_a_frente,conjuntos_mlp,conjuntos_lstm,var_norm,caminho_pasta):
    
    hoje = (date.today() -timedelta(days=atraso)).isoformat()
    hoje_lstm = date.today() -timedelta(days=atraso)
    ontem = (date.today() -timedelta(days=atraso +1)).isoformat()
    data_previsao = ((date.today() -timedelta(days=atraso))+timedelta(days=dias_a_frente))

    caminho_arquivos = caminho_pasta +'DeckCorrigido/'
    caminho_previsao = caminho_pasta + 'Previsoes/'

    #2 DIAS DE JANELA / 5 TIME STEPS - S e SECO           #3 DIAS DE JANELA / 5 TIME STEPS - S e SECO
    #2 DIAS DE JANELA / 2 TIME STEPS - NE e N             #2 DIAS DE JANELA / 2 TIME STEPS - NE         3 DIAS DE JANELA / 2 TIME STEPS -N

    #montagem do input da rede MLP
    tamanho_janela_MLP_seco = 2 #dias
    tamanho_janela_MLP_s = 2 #dias
    tamanho_janela_MLP_ne = 2 #dias
    tamanho_janela_MLP_n = 2 #dias

    horizonte_previsao_MLP = 1 #dia
    horizonte_temp_MLP= horizonte_previsao_MLP #dia

    #montagem do input da rede LSTM
    tamanho_janela_LSTM_seco = tamanho_janela_MLP_seco #dias
    tamanho_janela_LSTM_s = tamanho_janela_MLP_s #dias
    tamanho_janela_LSTM_ne = tamanho_janela_MLP_ne #dias
    tamanho_janela_LSTM_n = tamanho_janela_MLP_n #dias

    horizonte_previsao_LSTM = horizonte_previsao_MLP #dia
    horizonte_temp_LSTM= horizonte_previsao_LSTM #dia

    time_steps_seco = 5
    time_steps_s = 5
    time_steps_ne = 2
    time_steps_n = 2

    modelo_previsao = 'LSTM' #MLP, LSTM OU GRU
    data_prevista = (date.today() -timedelta(days=atraso +horizonte_previsao_MLP)).isoformat()
    
    #####################################################################################################################################################################
    seco_conjunto_Previsao,s_conjunto_Previsao,ne_conjunto_Previsao,n_conjunto_Previsao = conjuntos_mlp['seco_conjunto_Previsao'],conjuntos_mlp['s_conjunto_Previsao'],conjuntos_mlp['ne_conjunto_Previsao'],conjuntos_mlp['n_conjunto_Previsao']
    seco_conjunto_Previsao_LSTM,s_conjunto_Previsao_LSTM,ne_conjunto_Previsao_LSTM,n_conjunto_Previsao_LSTM = conjuntos_lstm['seco_conjunto_Previsao_LSTM'],conjuntos_lstm['s_conjunto_Previsao_LSTM'],conjuntos_lstm['ne_conjunto_Previsao_LSTM'],conjuntos_lstm['n_conjunto_Previsao_LSTM']

    seco_c_std,seco_c_min, = var_norm['seco_var_norm']['c_std'],var_norm['seco_var_norm']['c_min']
    s_c_std,s_c_min, = var_norm['s_var_norm']['c_std'],var_norm['s_var_norm']['c_min']
    ne_c_std,ne_c_min, = var_norm['ne_var_norm']['c_std'],var_norm['ne_var_norm']['c_min']
    n_c_std,n_c_min, = var_norm['n_var_norm']['c_std'],var_norm['n_var_norm']['c_min']

    #####################################################################################################################################################################

    def my_loss_fn_seco(y_true, y_pred):
        return tf.reduce_mean(((y_pred*seco_c_std+seco_c_min)/(y_true*seco_c_std+seco_c_min)-1)**2)

    def my_loss_fn_s(y_true, y_pred):
        return tf.reduce_mean(((y_pred*s_c_std+s_c_min)/(y_true*s_c_std+s_c_min)-1)**2)

    def my_loss_fn_ne(y_true, y_pred):
        return tf.reduce_mean(((y_pred*ne_c_std+ne_c_min)/(y_true*ne_c_std+ne_c_min)-1)**2)

    def my_loss_fn_n(y_true, y_pred):
        return tf.reduce_mean(((y_pred*n_c_std+n_c_min)/(y_true*n_c_std+n_c_min)-1)**2)

    tic=timeit.default_timer()
    print("Prevendo carga...")
    
    ##############################  MLP COMPLETA ##################################
    n_splits = 10
    n_prevs = 100
    Previsao_SECO = []
    Previsao_S = []
    Previsao_NE = []
    Previsao_N = []

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (n_prevs)])
        return pred_stack 

    j = 0
    for i in range(n_splits):
        j = j+1
        #print("split ",j)
        MLP_SECO = tf.keras.models.load_model("Modelos_Treinados/MLP_completa_SECO_"+str(j)+"_"+str(horizonte_previsao_MLP)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_seco})
        MLP_S = tf.keras.models.load_model("Modelos_Treinados/MLP_completa_S_"+str(j)+"_"+str(horizonte_previsao_MLP)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_s})
        MLP_NE = tf.keras.models.load_model("Modelos_Treinados/MLP_completa_NE_"+str(j)+"_"+str(horizonte_previsao_MLP)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_ne})
        MLP_N = tf.keras.models.load_model("Modelos_Treinados/MLP_completa_N_"+str(j)+"_"+str(horizonte_previsao_MLP)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_n})

        previsao_SECO=predict(MLP_SECO,seco_conjunto_Previsao)
        Previsao_SECO.append(previsao_SECO)
        previsao_S=predict(MLP_S,s_conjunto_Previsao)
        Previsao_S.append(previsao_S)
        previsao_NE=predict(MLP_NE,ne_conjunto_Previsao)
        Previsao_NE.append(previsao_NE)
        previsao_N=predict(MLP_N,n_conjunto_Previsao)
        Previsao_N.append(previsao_N)
        

    Previsao_SECO = np.asarray(Previsao_SECO)
    prev_stack_SECO=Previsao_SECO.mean(axis=0)
    prev_stack_SECO=prev_stack_SECO*seco_c_std+seco_c_min
    previsao_media_SECO = np.mean(prev_stack_SECO, axis = 0)

    Previsao_S = np.asarray(Previsao_S)
    prev_stack_S=Previsao_S.mean(axis=0)
    prev_stack_S=prev_stack_S*s_c_std+s_c_min
    previsao_media_S = np.mean(prev_stack_S, axis = 0)

    Previsao_NE = np.asarray(Previsao_NE)
    prev_stack_NE=Previsao_NE.mean(axis=0)
    prev_stack_NE=prev_stack_NE*ne_c_std+ne_c_min
    previsao_media_NE = np.mean(prev_stack_NE, axis = 0)

    Previsao_N = np.asarray(Previsao_N)
    prev_stack_N=Previsao_N.mean(axis=0)
    prev_stack_N=prev_stack_N*n_c_std+n_c_min
    previsao_media_N = np.mean(prev_stack_N, axis = 0)

    ##############################  LSTM COMPLETA ##################################
    n_splits_LSTM = 20
    n_prevs_LSTM = 1
    Previsao_SECO_LSTM = []
    Previsao_S_LSTM = []
    Previsao_NE_LSTM = []
    Previsao_N_LSTM = []

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (n_prevs_LSTM)])
        return pred_stack 

    j = 0
    for i in range(n_splits_LSTM):
        j = j+1
        #print("split ",j)
        LSTM_SECO = tf.keras.models.load_model("Modelos_Treinados/LSTM_completa_SECO_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_seco})
        LSTM_S = tf.keras.models.load_model("Modelos_Treinados/LSTM_completa_S_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_s})
        LSTM_NE = tf.keras.models.load_model("Modelos_Treinados/LSTM_completa_NE_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_ne})
        LSTM_N = tf.keras.models.load_model("Modelos_Treinados/LSTM_completa_N_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_n})
        
        previsao_SECO_LSTM=predict(LSTM_SECO,seco_conjunto_Previsao_LSTM)
        Previsao_SECO_LSTM.append(previsao_SECO_LSTM)
        #print("seco")
        previsao_S_LSTM=predict(LSTM_S,s_conjunto_Previsao_LSTM)
        Previsao_S_LSTM.append(previsao_S_LSTM)
        #print("s")
        previsao_NE_LSTM=predict(LSTM_NE,ne_conjunto_Previsao_LSTM)
        Previsao_NE_LSTM.append(previsao_NE_LSTM)
        #print("ne")
        previsao_N_LSTM=predict(LSTM_N,n_conjunto_Previsao_LSTM)
        Previsao_N_LSTM.append(previsao_N_LSTM)
        #print("n")
        

    Previsao_SECO_LSTM = np.asarray(Previsao_SECO_LSTM)
    prev_stack_SECO_LSTM=Previsao_SECO_LSTM.mean(axis=0)
    prev_stack_SECO_LSTM=prev_stack_SECO_LSTM*seco_c_std+seco_c_min
    previsao_media_SECO_LSTM = np.mean(prev_stack_SECO_LSTM, axis = 0)

    Previsao_S_LSTM = np.asarray(Previsao_S_LSTM)
    prev_stack_S_LSTM=Previsao_S_LSTM.mean(axis=0)
    prev_stack_S_LSTM=prev_stack_S_LSTM*s_c_std+s_c_min
    previsao_media_S_LSTM = np.mean(prev_stack_S_LSTM, axis = 0)

    Previsao_NE_LSTM = np.asarray(Previsao_NE_LSTM)
    prev_stack_NE_LSTM=Previsao_NE_LSTM.mean(axis=0)
    prev_stack_NE_LSTM=prev_stack_NE_LSTM*ne_c_std+ne_c_min
    previsao_media_NE_LSTM = np.mean(prev_stack_NE_LSTM, axis = 0)

    Previsao_N_LSTM = np.asarray(Previsao_N_LSTM)
    prev_stack_N_LSTM=Previsao_N_LSTM.mean(axis=0)
    prev_stack_N_LSTM=prev_stack_N_LSTM*n_c_std+n_c_min
    previsao_media_N_LSTM = np.mean(prev_stack_N_LSTM, axis = 0)

    ##############################  GRU COMPLETA ##################################
    n_splits_GRU = 20
    n_prevs_GRU = 1
    Previsao_SECO_GRU = []
    Previsao_S_GRU = []
    Previsao_NE_GRU = []
    Previsao_N_GRU = []

    def predict(model,X_Test):
        pred_stack = np.stack([model.predict(X_Test) for sample in range (n_prevs_GRU)])
        return pred_stack 

    j = 0
    for i in range(n_splits_GRU):
        j = j+1
        #print("split ",j)
        GRU_SECO = tf.keras.models.load_model("Modelos_Treinados/GRU_completa_SECO_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_seco})
        GRU_S = tf.keras.models.load_model("Modelos_Treinados/GRU_completa_S_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_s})
        GRU_NE = tf.keras.models.load_model("Modelos_Treinados/GRU_completa_NE_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_ne})
        GRU_N = tf.keras.models.load_model("Modelos_Treinados/GRU_completa_N_"+str(j)+"_"+str(horizonte_previsao_LSTM)+"_dias",custom_objects={'my_loss_fn': my_loss_fn_n})
        
        previsao_SECO_GRU=predict(GRU_SECO,seco_conjunto_Previsao_LSTM)
        Previsao_SECO_GRU.append(previsao_SECO_GRU)
        previsao_S_GRU=predict(GRU_S,s_conjunto_Previsao_LSTM)
        Previsao_S_GRU.append(previsao_S_GRU)
        previsao_NE_GRU=predict(GRU_NE,ne_conjunto_Previsao_LSTM)
        Previsao_NE_GRU.append(previsao_NE_GRU)
        previsao_N_GRU=predict(GRU_N,n_conjunto_Previsao_LSTM)
        Previsao_N_GRU.append(previsao_N_GRU)
        
        
    Previsao_SECO_GRU = np.asarray(Previsao_SECO_GRU)
    prev_stack_SECO_GRU=Previsao_SECO_GRU.mean(axis=0)
    prev_stack_SECO_GRU=prev_stack_SECO_GRU*seco_c_std+seco_c_min
    previsao_media_SECO_GRU = np.mean(prev_stack_SECO_GRU, axis = 0)

    Previsao_S_GRU = np.asarray(Previsao_S_GRU)
    prev_stack_S_GRU=Previsao_S_GRU.mean(axis=0)
    prev_stack_S_GRU=prev_stack_S_GRU*s_c_std+s_c_min
    previsao_media_S_GRU = np.mean(prev_stack_S_GRU, axis = 0)

    Previsao_NE_GRU = np.asarray(Previsao_NE_GRU)
    prev_stack_NE_GRU=Previsao_NE_GRU.mean(axis=0)
    prev_stack_NE_GRU=prev_stack_NE_GRU*ne_c_std+ne_c_min
    previsao_media_NE_GRU = np.mean(prev_stack_NE_GRU, axis = 0)

    Previsao_N_GRU = np.asarray(Previsao_N_GRU)
    prev_stack_N_GRU=Previsao_N_GRU.mean(axis=0)
    prev_stack_N_GRU=prev_stack_N_GRU*n_c_std+n_c_min
    previsao_media_N_GRU = np.mean(prev_stack_N_GRU, axis = 0)

    toc=timeit.default_timer()
    print("Tempo de previsão: ", (toc - tic))
    ############################################################################################################################################################
    prev_ONS_SECO= pd.read_csv(caminho_arquivos + 'SECO/seco_carga_prev_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    ons_SECO = np.array(prev_ONS_SECO["val_previsaocarga"][(dias_a_frente-1)*48:dias_a_frente*48])

    prev_ONS_S= pd.read_csv(caminho_arquivos + 'S/s_carga_prev_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    ons_S = np.array(prev_ONS_S["val_previsaocarga"][(dias_a_frente-1)*48:dias_a_frente*48])

    prev_ONS_NE= pd.read_csv(caminho_arquivos + 'NE/ne_carga_prev_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    ons_NE = np.array(prev_ONS_NE["val_previsaocarga"][(dias_a_frente-1)*48:dias_a_frente*48])

    prev_ONS_N= pd.read_csv(caminho_arquivos + 'N/n_carga_prev_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    ons_N = np.array(prev_ONS_N["val_previsaocarga"][(dias_a_frente-1)*48:dias_a_frente*48])

    eixo_x = np.array(list(range(horizonte_previsao_MLP*24)))

    from scipy.interpolate import CubicSpline
    def interpola(y_prev_seco,y_prev_s,y_prev_ne,y_prev_n):

        SECO = np.array(y_prev_seco[0])
        S = np.array(y_prev_s[0])
        NE = np.array(y_prev_ne[0])
        N = np.array(y_prev_n[0])

        cs_SECO = CubicSpline(eixo_x,SECO,bc_type='natural')
        cs_S = CubicSpline(eixo_x,S,bc_type='natural')
        cs_NE = CubicSpline(eixo_x,NE,bc_type='natural')
        cs_N = CubicSpline(eixo_x,N,bc_type='natural')

        prev_SECO_aux = []
        prev_S_aux = []
        prev_NE_aux = []
        prev_N_aux = []

        prev_SECO_final = []
        prev_S_final = []
        prev_NE_final = []
        prev_N_final = []

        j = 0
        for i in range(horizonte_previsao_MLP*48):
            prev_SECO_aux.append(cs_SECO(j))
            prev_S_aux.append(cs_S(j))
            prev_NE_aux.append(cs_NE(j))
            prev_N_aux.append(cs_N(j))
            j = j+0.5

        prev_SECO_aux= np.array(prev_SECO_aux)
        prev_SECO_aux = prev_SECO_aux.tolist()
        prev_S_aux= np.array(prev_S_aux)
        prev_S_aux = prev_S_aux.tolist()
        prev_NE_aux= np.array(prev_NE_aux)
        prev_NE_aux = prev_NE_aux.tolist()
        prev_N_aux= np.array(prev_N_aux)
        prev_N_aux = prev_N_aux.tolist()

        for i in range(horizonte_previsao_MLP*48):
            f_SECO = prev_SECO_aux[i]
            prev_SECO_final.append(f_SECO)

            f_S = prev_S_aux[i]
            prev_S_final.append(f_S)

            f_NE = prev_NE_aux[i]
            prev_NE_final.append(f_NE)

            f_N = prev_N_aux[i]
            prev_N_final.append(f_N)
        
        return prev_SECO_final,prev_S_final,prev_NE_final,prev_N_final

    prev_mlp_completa_SECO_final,prev_mlp_completa_S_final,prev_mlp_completa_NE_final,prev_mlp_completa_N_final = interpola(previsao_media_SECO,previsao_media_S,previsao_media_NE,previsao_media_N)
    prev_lstm_completa_SECO_final,prev_lstm_completa_S_final,prev_lstm_completa_NE_final,prev_lstm_completa_N_final = interpola(previsao_media_SECO_LSTM,previsao_media_S_LSTM,previsao_media_NE_LSTM,previsao_media_N_LSTM)
    prev_gru_completa_SECO_final,prev_gru_completa_S_final,prev_gru_completa_NE_final,prev_gru_completa_N_final = interpola(previsao_media_SECO_GRU,previsao_media_S_GRU,previsao_media_NE_GRU,previsao_media_N_GRU)

    #####################################################################################################################################################
 
    Prev_fechada = pd.DataFrame()

    #Prev_fechada["MLP_Simples_Seco"] = prev_mlp_simples_SECO_final
    #Prev_fechada["MLP_Simples_S"] = prev_mlp_simples_S_final
    #Prev_fechada["MLP_Simples_NE"] = prev_mlp_simples_NE_final
    #Prev_fechada["MLP_Simples_N"] = prev_mlp_simples_N_final

    #Prev_fechada["LSTM_Simples_Seco"] = prev_lstm_simples_SECO_final
    #Prev_fechada["LSTM_Simples_S"] = prev_lstm_simples_S_final
    #Prev_fechada["LSTM_Simples_NE"] = prev_lstm_simples_NE_final
    #Prev_fechada["LSTM_Simples_N"] = prev_lstm_simples_N_final

    #Prev_fechada["GRU_Simples_Seco"] = prev_GRU_simples_SECO_final
    #Prev_fechada["GRU_Simples_S"] = prev_GRU_simples_S_final
    #Prev_fechada["GRU_Simples_NE"] = prev_GRU_simples_NE_final
    #Prev_fechada["GRU_Simples_N"] = prev_GRU_simples_N_final

    Prev_fechada["MLP_Seco"] = prev_mlp_completa_SECO_final
    Prev_fechada["MLP_S"] = prev_mlp_completa_S_final
    Prev_fechada["MLP_NE"] = prev_mlp_completa_NE_final
    Prev_fechada["MLP_N"] = prev_mlp_completa_N_final
    MLP_SIN = []
    for i in range(len(prev_mlp_completa_SECO_final)):
        MLP_SIN.append(prev_mlp_completa_SECO_final[i]+prev_mlp_completa_S_final[i]+prev_mlp_completa_NE_final[i]+prev_mlp_completa_N_final[i])
    Prev_fechada["MLP_SIN"] = MLP_SIN

    Prev_fechada["LSTM_Seco"] = prev_lstm_completa_SECO_final
    Prev_fechada["LSTM_S"] = prev_lstm_completa_S_final
    Prev_fechada["LSTM_NE"] = prev_lstm_completa_NE_final
    Prev_fechada["LSTM_N"] = prev_lstm_completa_N_final
    LSTM_SIN = []
    for i in range(len(prev_mlp_completa_SECO_final)):
        LSTM_SIN.append(prev_lstm_completa_SECO_final[i]+prev_lstm_completa_S_final[i]+prev_lstm_completa_NE_final[i]+prev_lstm_completa_N_final[i])
    Prev_fechada["LSTM_SIN"] = LSTM_SIN

    Prev_fechada["GRU_Seco"] = prev_gru_completa_SECO_final
    Prev_fechada["GRU_S"] = prev_gru_completa_S_final
    Prev_fechada["GRU_NE"] = prev_gru_completa_NE_final
    Prev_fechada["GRU_N"] = prev_gru_completa_N_final
    GRU_SIN = []
    for i in range(len(prev_mlp_completa_SECO_final)):
        GRU_SIN.append(prev_gru_completa_SECO_final[i]+prev_gru_completa_S_final[i]+prev_gru_completa_NE_final[i]+prev_gru_completa_N_final[i])
    Prev_fechada["GRU_SIN"] = GRU_SIN

    if (horizonte_previsao_LSTM <8) & (ons_SECO[0]>1):
        Prev_fechada["Prev_ONS_Seco"] = ons_SECO
        Prev_fechada["Prev_ONS_S"] = ons_S
        Prev_fechada["Prev_ONS_NE"] = ons_NE
        Prev_fechada["Prev_ONS_N"] = ons_N
        ONS_SIN = []
        for i in range(len(ons_SECO)):
            ONS_SIN.append(ons_SECO[i]+ons_S[i]+ons_NE[i]+ons_N[i])
        Prev_fechada["Prev ONS_SIN"] = ONS_SIN

    Prev_fechada = round(Prev_fechada,2)
    Prev_fechada.to_csv(caminho_pasta + "Previsoes/"+str(hoje_lstm.year)+'/'+'%02d' % hoje_lstm.month+'/'+str(hoje)+'/daily/'+str(data_previsao)+".csv",sep = ';',decimal=',')

    return Prev_fechada

def avalia_prev_passada(atraso,horizonte_previsao,caminho_pasta):
        
    hoje = (date.today() -timedelta(days=atraso)).isoformat()
    hoje_lstm = date.today() -timedelta(days=atraso)
    ontem = (date.today() -timedelta(days=atraso +1)).isoformat()
    ontem_lstm = (date.today() -timedelta(days=atraso +1))
    antes_ontem = (date.today() -timedelta(days=atraso +2)).isoformat()
    antes_ontem_weekly = (date.today() -timedelta(days=atraso +1+horizonte_previsao)).isoformat()
    antes_ontem_lstm = (date.today() -timedelta(days=atraso +2))
    antes_ontem_lstm_weekly = (date.today() -timedelta(days=atraso +1+horizonte_previsao))
    
    caminho_arquivos = caminho_pasta +'DeckCorrigido/'
    caminho_previsao = caminho_pasta + 'Previsoes/'

    #2 DIAS DE JANELA / 5 TIME STEPS - S e SECO           #3 DIAS DE JANELA / 5 TIME STEPS - S e SECO
    #2 DIAS DE JANELA / 2 TIME STEPS - NE e N             #2 DIAS DE JANELA / 2 TIME STEPS - NE         3 DIAS DE JANELA / 2 TIME STEPS -N

    #montagem do input da rede MLP
    tamanho_janela_MLP_seco = 2 #dias
    tamanho_janela_MLP_s = 2 #dias
    tamanho_janela_MLP_ne = 2 #dias
    tamanho_janela_MLP_n = 2 #dias

    horizonte_previsao_MLP = horizonte_previsao #dia
    horizonte_temp_MLP= horizonte_previsao_MLP #dia

    #montagem do input da rede LSTM
    tamanho_janela_LSTM_seco = tamanho_janela_MLP_seco #dias
    tamanho_janela_LSTM_s = tamanho_janela_MLP_s #dias
    tamanho_janela_LSTM_ne = tamanho_janela_MLP_ne #dias
    tamanho_janela_LSTM_n = tamanho_janela_MLP_n #dias

    horizonte_previsao_LSTM = horizonte_previsao_MLP #dia
    horizonte_temp_LSTM= horizonte_previsao_LSTM #dia

    time_steps_seco = 5
    time_steps_s = 5
    time_steps_ne = 2
    time_steps_n = 2

    modelo_previsao = 'LSTM' #MLP, LSTM OU GRU
    data_prevista = (date.today() -timedelta(days=atraso +horizonte_previsao_MLP)).isoformat()
    
    #####################################################################################################################################################################
    #Lê arquivos e trata os dados
    SECO_arquivo_carga = pd.read_csv(caminho_arquivos + 'SECO/seco_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    SECO_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'SECO/seco_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    SECO_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'SECO/seco_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    SECO_feriados = pd.read_csv(caminho_arquivos + 'SECO/SECO_'+hoje+'_FERIADOS.csv', delimiter = ',',decimal = ',')

    S_arquivo_carga = pd.read_csv(caminho_arquivos + 'S/s_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    S_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'S/s_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    S_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'S/s_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    S_feriados = pd.read_csv(caminho_arquivos + 'S/S_'+hoje+'_FERIADOS.csv', delimiter = ',')

    NE_arquivo_carga = pd.read_csv(caminho_arquivos + 'NE/ne_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    NE_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'NE/ne_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    NE_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'NE/ne_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    NE_feriados = pd.read_csv(caminho_arquivos + 'NE/NE_'+hoje+'_FERIADOS.csv', delimiter = ',')

    N_arquivo_carga = pd.read_csv(caminho_arquivos + 'N/n_carga_deck_'+hoje+'.csv',delimiter = ',',decimal = ',')
    N_arquivo_temp_hist = pd.read_csv(caminho_arquivos + 'N/n_temp_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    N_arquivo_temp_prevista = pd.read_csv(caminho_arquivos + 'N/n_temp_prev_deck_'+hoje+'.csv', delimiter = ',',decimal = ',')
    N_feriados = pd.read_csv(caminho_arquivos + 'N/N_'+hoje+'_FERIADOS.csv', delimiter = ',',decimal = ',')
    
    ########################################################################################################################################################################
    previsao_de_ontem_feita = 0 #0 caso nao tenha previsão de ontem, 1 caso tenha
    if horizonte_previsao ==1:
        if os.path.isfile(caminho_previsao+str(antes_ontem_lstm.year)+'/'+'%02d' % antes_ontem_lstm.month+'/'+str(antes_ontem)+'/daily/'+data_prevista+'.csv') == True:
            previsao_de_ontem_feita = 1
    else:
        if os.path.isfile(caminho_previsao+str(antes_ontem_lstm_weekly.year)+'/'+'%02d' % antes_ontem_lstm_weekly.month+'/'+str(antes_ontem_weekly)+'/weekly/'+data_prevista+'.csv') == True:
            previsao_de_ontem_feita = 1
    
    if previsao_de_ontem_feita == 1:
        SECO_carga_verificada =  SECO_arquivo_carga["Carga"][SECO_arquivo_carga.shape[0]-horizonte_previsao_MLP*24:]
        S_carga_verificada =  S_arquivo_carga["Carga"][S_arquivo_carga.shape[0]-horizonte_previsao_MLP*24:]
        NE_carga_verificada =  NE_arquivo_carga["Carga"][NE_arquivo_carga.shape[0]-horizonte_previsao_MLP*24:]
        N_carga_verificada =  N_arquivo_carga["Carga"][N_arquivo_carga.shape[0]-horizonte_previsao_MLP*24:]
        
        ###################################################
        #interpola para discretizacao semi horaria
        x = np.array(list(range(horizonte_previsao_MLP*24)))
        from scipy.interpolate import CubicSpline
        cs_SECO_carga_verificada = CubicSpline(x,SECO_carga_verificada,bc_type='natural')
        cs_S_carga_verificada = CubicSpline(x,S_carga_verificada,bc_type='natural')
        cs_NE_carga_verificada = CubicSpline(x,NE_carga_verificada,bc_type='natural')
        cs_N_carga_verificada = CubicSpline(x,N_carga_verificada,bc_type='natural')
        
        SECO_verificada_aux = []
        SECO_verificada_final = []
        S_verificada_aux = []
        S_verificada_final = []
        NE_verificada_aux = []
        NE_verificada_final = []
        N_verificada_aux = []
        N_verificada_final = []

        j = 0
        for i in range(horizonte_previsao_MLP*48):
            SECO_verificada_aux.append(cs_SECO_carga_verificada(j))
            S_verificada_aux.append(cs_S_carga_verificada(j))
            NE_verificada_aux.append(cs_NE_carga_verificada(j))
            N_verificada_aux.append(cs_N_carga_verificada(j))
            j = j+0.5

        SECO_verificada_aux= np.array(SECO_verificada_aux)
        SECO_verificada_aux = SECO_verificada_aux.tolist()
        S_verificada_aux= np.array(S_verificada_aux)
        S_verificada_aux = S_verificada_aux.tolist()
        NE_verificada_aux= np.array(NE_verificada_aux)
        NE_verificada_aux = NE_verificada_aux.tolist()
        N_verificada_aux= np.array(N_verificada_aux)
        N_verificada_aux = N_verificada_aux.tolist()

        for i in range(horizonte_previsao_MLP*48):
            veri_SECO = SECO_verificada_aux[i]
            SECO_verificada_final.append(veri_SECO)
            veri_S = S_verificada_aux[i]
            S_verificada_final.append(veri_S)
            veri_NE = NE_verificada_aux[i]
            NE_verificada_final.append(veri_NE)
            veri_N = N_verificada_aux[i]
            N_verificada_final.append(veri_N)
        
        SIN_verificada_final = []
        for i in range(len(SECO_verificada_final)):
            SIN_verificada_final.append(SECO_verificada_final[i]+S_verificada_final[i]+NE_verificada_final[i]+N_verificada_final[i])
        ###################################################
        #Le arquivo de previsão passada
        if horizonte_previsao ==1:
            previsao_passada = pd.read_csv(caminho_previsao+str(antes_ontem_lstm.year)+'/'+ '%02d' % antes_ontem_lstm.month +'/'+str(antes_ontem)+'/daily/'+data_prevista+'.csv',delimiter = ';',decimal = ',')
        else:
            previsao_passada = pd.read_csv(caminho_previsao+str(antes_ontem_lstm_weekly.year)+'/'+ '%02d' % antes_ontem_lstm_weekly.month +'/'+str(antes_ontem_weekly)+'/weekly/'+data_prevista+'.csv',delimiter = ';',decimal = ',')   
 
        if len(previsao_passada) == 48*horizonte_previsao:
            SECO_previsao_verificada_MLP,SECO_previsao_verificada_LSTM,SECO_previsao_verificada_GRU =  previsao_passada["MLP_Seco"],previsao_passada["LSTM_Seco"],previsao_passada["GRU_Seco"]
            S_previsao_verificada_MLP,S_previsao_verificada_LSTM,S_previsao_verificada_GRU =  previsao_passada["MLP_S"],previsao_passada["LSTM_S"],previsao_passada["GRU_S"]
            NE_previsao_verificada_MLP,NE_previsao_verificada_LSTM,NE_previsao_verificada_GRU =  previsao_passada["MLP_NE"],previsao_passada["LSTM_NE"],previsao_passada["GRU_NE"]
            N_previsao_verificada_MLP,N_previsao_verificada_LSTM,N_previsao_verificada_GRU =  previsao_passada["MLP_N"],previsao_passada["LSTM_N"],previsao_passada["GRU_N"]
            
            SIN_previsao_verificada_MLP,SIN_previsao_verificada_LSTM ,SIN_previsao_verificada_GRU= [],[],[]
            for i in range(len(SECO_verificada_final)):
                SIN_previsao_verificada_MLP.append(SECO_previsao_verificada_MLP[i]+S_previsao_verificada_MLP[i]+NE_previsao_verificada_MLP[i]+N_previsao_verificada_MLP[i])
                SIN_previsao_verificada_LSTM.append(SECO_previsao_verificada_LSTM[i]+S_previsao_verificada_LSTM[i]+NE_previsao_verificada_LSTM[i]+N_previsao_verificada_LSTM[i])
                SIN_previsao_verificada_GRU.append(SECO_previsao_verificada_GRU[i]+S_previsao_verificada_GRU[i]+NE_previsao_verificada_GRU[i]+N_previsao_verificada_GRU[i])

            try:
                SECO_previsao_ONS =  previsao_passada["Prev_ONS_Seco"]
                S_previsao_ONS =  previsao_passada["Prev_ONS_S"]
                NE_previsao_ONS =  previsao_passada["Prev_ONS_NE"]
                N_previsao_ONS =  previsao_passada["Prev_ONS_N"]

                SIN_previsao_ONS = []
                for i in range(len(SECO_verificada_final)):
                    SIN_previsao_ONS.append(SECO_previsao_ONS[i]+S_previsao_ONS[i]+NE_previsao_ONS[i]+N_previsao_ONS[i])
                
            except:
                True

            #####################################################
            #Avalia e plota resultados
            mape_MLP_SECO = []
            mape_MLP_S = []
            mape_MLP_NE = []
            mape_MLP_N = []
            mape_MLP_SIN = []

            mape_LSTM_SECO = []
            mape_LSTM_S = []
            mape_LSTM_NE = []
            mape_LSTM_N = []
            mape_LSTM_SIN = []

            mape_GRU_SECO = []
            mape_GRU_S = []
            mape_GRU_NE = []
            mape_GRU_N = []
            mape_GRU_SIN = []

            mape_ONS_SECO = []
            mape_ONS_S = []
            mape_ONS_NE = []
            mape_ONS_N = []
            mape_ONS_SIN = []

            mae_MLP_SECO = []
            mae_MLP_S = []
            mae_MLP_NE = []
            mae_MLP_N = []
            mae_MLP_SIN = []

            mae_LSTM_SECO = []
            mae_LSTM_S = []
            mae_LSTM_NE = []
            mae_LSTM_N = []
            mae_LSTM_SIN = []

            mae_GRU_SECO = []
            mae_GRU_S = []
            mae_GRU_NE = []
            mae_GRU_N = []
            mae_GRU_SIN = []

            mae_ONS_SECO = []
            mae_ONS_S = []
            mae_ONS_NE = []
            mae_ONS_N = []
            mae_ONS_SIN = []


            #CALCULA MAPE
            for i in range(48*horizonte_previsao_MLP):
                mape_MLP_SECO.append((np.abs(SECO_previsao_verificada_MLP[i] - SECO_verificada_final[i])/SECO_verificada_final[i])*100)
                mape_LSTM_SECO.append((np.abs(SECO_previsao_verificada_LSTM[i] - SECO_verificada_final[i])/SECO_verificada_final[i])*100)
                mape_GRU_SECO.append((np.abs(SECO_previsao_verificada_GRU[i] - SECO_verificada_final[i])/SECO_verificada_final[i])*100)
                
                mape_MLP_S.append((np.abs(S_previsao_verificada_MLP[i] - S_verificada_final[i])/S_verificada_final[i])*100)
                mape_LSTM_S.append((np.abs(S_previsao_verificada_LSTM[i] - S_verificada_final[i])/S_verificada_final[i])*100)
                mape_GRU_S.append((np.abs(S_previsao_verificada_GRU[i] - S_verificada_final[i])/S_verificada_final[i])*100)

                mape_MLP_NE.append((np.abs(NE_previsao_verificada_MLP[i] - NE_verificada_final[i])/NE_verificada_final[i])*100)
                mape_LSTM_NE.append((np.abs(NE_previsao_verificada_LSTM[i] - NE_verificada_final[i])/NE_verificada_final[i])*100)
                mape_GRU_NE.append((np.abs(NE_previsao_verificada_GRU[i] - NE_verificada_final[i])/NE_verificada_final[i])*100)

                mape_MLP_N.append((np.abs(N_previsao_verificada_MLP[i] - N_verificada_final[i])/N_verificada_final[i])*100)
                mape_LSTM_N.append((np.abs(N_previsao_verificada_LSTM[i] - N_verificada_final[i])/N_verificada_final[i])*100)
                mape_GRU_N.append((np.abs(N_previsao_verificada_GRU[i] - N_verificada_final[i])/N_verificada_final[i])*100)

                mape_MLP_SIN.append((np.abs(SIN_previsao_verificada_MLP[i] - SIN_verificada_final[i])/SIN_verificada_final[i])*100)
                mape_LSTM_SIN.append((np.abs(SIN_previsao_verificada_LSTM[i] - SIN_verificada_final[i])/SIN_verificada_final[i])*100)
                mape_GRU_SIN.append((np.abs(SIN_previsao_verificada_GRU[i] - SIN_verificada_final[i])/SIN_verificada_final[i])*100)
                try:
                    mape_ONS_SECO.append((np.abs(SECO_previsao_ONS[i] - SECO_verificada_final[i])/SECO_verificada_final[i])*100)
                    mape_ONS_S.append((np.abs(S_previsao_ONS[i] - S_verificada_final[i])/S_verificada_final[i])*100)
                    mape_ONS_NE.append((np.abs(NE_previsao_ONS[i] - NE_verificada_final[i])/NE_verificada_final[i])*100)
                    mape_ONS_N.append((np.abs(N_previsao_ONS[i] - N_verificada_final[i])/N_verificada_final[i])*100)
                    mape_ONS_SIN.append((np.abs(SIN_previsao_ONS[i] - SIN_verificada_final[i])/SIN_verificada_final[i])*100)
                except:
                    True
            
            #CALCULA ERRO ABSOLUTO
            for i in range(48*horizonte_previsao_MLP):
                mae_MLP_SECO.append(SECO_previsao_verificada_MLP[i] - SECO_verificada_final[i])
                mae_LSTM_SECO.append(SECO_previsao_verificada_LSTM[i] - SECO_verificada_final[i])
                mae_GRU_SECO.append(SECO_previsao_verificada_GRU[i] - SECO_verificada_final[i])
                
                mae_MLP_S.append(S_previsao_verificada_MLP[i] - S_verificada_final[i])
                mae_LSTM_S.append(S_previsao_verificada_LSTM[i] - S_verificada_final[i])
                mae_GRU_S.append(S_previsao_verificada_GRU[i] - S_verificada_final[i])

                mae_MLP_NE.append(NE_previsao_verificada_MLP[i] - NE_verificada_final[i])
                mae_LSTM_NE.append(NE_previsao_verificada_LSTM[i] - NE_verificada_final[i])
                mae_GRU_NE.append(NE_previsao_verificada_GRU[i] - NE_verificada_final[i])

                mae_MLP_N.append(N_previsao_verificada_MLP[i] - N_verificada_final[i])
                mae_LSTM_N.append(N_previsao_verificada_LSTM[i] - N_verificada_final[i])
                mae_GRU_N.append(N_previsao_verificada_GRU[i] - N_verificada_final[i])

                mae_MLP_SIN.append(SIN_previsao_verificada_MLP[i] - SIN_verificada_final[i])
                mae_LSTM_SIN.append(SIN_previsao_verificada_LSTM[i] - SIN_verificada_final[i])
                mae_GRU_SIN.append(SIN_previsao_verificada_GRU[i] - SIN_verificada_final[i])
                try:
                    mae_ONS_SECO.append(SECO_previsao_ONS[i] - SECO_verificada_final[i])
                    mae_ONS_S.append(S_previsao_ONS[i] - S_verificada_final[i])
                    mae_ONS_NE.append(NE_previsao_ONS[i] - NE_verificada_final[i])
                    mae_ONS_N.append(N_previsao_ONS[i] - N_verificada_final[i])
                    mae_ONS_SIN.append(SIN_previsao_ONS[i] - SIN_verificada_final[i])
                except:
                    True

            mape_MLP_SECO_medio=np.average(mape_MLP_SECO)
            mape_MLP_S_medio=np.average(mape_MLP_S)
            mape_MLP_NE_medio=np.average(mape_MLP_NE)
            mape_MLP_N_medio=np.average(mape_MLP_N)

            mape_LSTM_SECO_medio=np.average(mape_LSTM_SECO)
            mape_LSTM_S_medio=np.average(mape_LSTM_S)
            mape_LSTM_NE_medio=np.average(mape_LSTM_NE)
            mape_LSTM_N_medio=np.average(mape_LSTM_N)

            mape_GRU_SECO_medio=np.average(mape_GRU_SECO)
            mape_GRU_S_medio=np.average(mape_GRU_S)
            mape_GRU_NE_medio=np.average(mape_GRU_NE)
            mape_GRU_N_medio=np.average(mape_GRU_N)

            mape_ONS_SECO_medio=np.average(mape_ONS_SECO)
            mape_ONS_S_medio=np.average(mape_ONS_S)
            mape_ONS_NE_medio=np.average(mape_ONS_NE)
            mape_ONS_N_medio=np.average(mape_ONS_N)
                    
            previsao_passada["SECO Carga Verificada"] = SECO_verificada_final
            previsao_passada["S Carga Verificada"] = S_verificada_final
            previsao_passada["NE Carga Verificada"] = NE_verificada_final
            previsao_passada["N Carga Verificada"] = N_verificada_final
            previsao_passada["SIN Carga Verificada"] = SIN_verificada_final

            #################################################################################################################################################################################
            #Pega carga no DESSEM
            caminho_entdados = '/home/previsorpld-data/dessem/ccee/'+str(ontem_lstm.year)+'/'+ '%02d' % ontem_lstm.month +'/'+str(ontem)+'/entdados.dat'
            verifica_caminho_entdados =  os.path.isfile(caminho_entdados)

            if (verifica_caminho_entdados == True)&(horizonte_previsao ==1):

                entdados = open(caminho_entdados)
                arq = entdados.readlines() 

                for i in range(len(arq)):
                    if 'CARGA' in arq[i]:
                        linha = i + 4
                        break
                
                dia_data_prevista = data_prevista[8:]
                dia_arq_dessem = arq[linha][8:10]

                if dia_arq_dessem == dia_data_prevista:
                
                    carga_seco, carga_s, carga_ne, carga_n = [],[],[],[]
                    for i in range(192):
                        submercado = arq[linha+i][5]
                        carga = arq[linha+i][25:34].replace(" ","")

                        if submercado == "1":
                            carga_seco.append(carga)
                        if submercado == "2":
                            carga_s.append(carga)
                        if submercado == "3":
                            carga_ne.append(carga)
                        if submercado == "4":
                            carga_n.append(carga)
                    
                    df = pd.DataFrame()
                    df["carga_seco"] = carga_seco[:48]
                    df["carga_s"] = carga_s[:48]
                    df["carga_ne"] = carga_ne[:48]
                    df["carga_n"] = carga_n[:48]

                    cargas_dessem = df.astype('float64')

                    previsao_passada['SECO carga dessem'] = cargas_dessem["carga_seco"]
                    previsao_passada['S carga dessem'] = cargas_dessem["carga_s"]
                    previsao_passada['NE carga dessem'] = cargas_dessem["carga_ne"]
                    previsao_passada['N carga dessem'] = cargas_dessem["carga_n"]

                else:
                    print("ERRO ao pegar carga do DESSEM\nArquivo com datas diferente")
                    print("Data prevista: "+str(data_prevista))
                    print("Dia do arquivo ENTDADOS: "+dia_arq_dessem)
            else:
                if (verifica_caminho_entdados == False)&(horizonte_previsao ==1):
                    print('Arquivo ENTDADOS não encontrado!')
            #################################################################################################################################################################################
            previsao_passada["MLP_SIN"] = SIN_previsao_verificada_MLP
            previsao_passada["LSTM_SIN"] = SIN_previsao_verificada_LSTM
            previsao_passada["GRU_SIN"] = SIN_previsao_verificada_GRU
            previsao_passada["Prev ONS_SIN"] = SIN_previsao_ONS
            
            previsao_passada["MAPE MLP Seco"] = mape_MLP_SECO
            previsao_passada["MAPE MLP S"] = mape_MLP_S
            previsao_passada["MAPE MLP NE"] = mape_MLP_NE
            previsao_passada["MAPE MLP N"] = mape_MLP_N
            previsao_passada["MAPE MLP SIN"] = mape_MLP_SIN

            previsao_passada["MAPE LSTM Seco"] = mape_LSTM_SECO
            previsao_passada["MAPE LSTM S"] = mape_LSTM_S
            previsao_passada["MAPE LSTM NE"] = mape_LSTM_NE
            previsao_passada["MAPE LSTM N"] = mape_LSTM_N
            previsao_passada["MAPE LSTM SIN"] = mape_LSTM_SIN

            previsao_passada["MAPE GRU Seco"] = mape_GRU_SECO
            previsao_passada["MAPE GRU S"] = mape_GRU_S
            previsao_passada["MAPE GRU NE"] = mape_GRU_NE
            previsao_passada["MAPE GRU N"] = mape_GRU_N
            previsao_passada["MAPE GRU SIN"] = mape_GRU_SIN

            try:
                previsao_passada["MAPE ONS Seco"] = mape_ONS_SECO
                previsao_passada["MAPE ONS S"] = mape_ONS_S
                previsao_passada["MAPE ONS NE"] = mape_ONS_NE
                previsao_passada["MAPE ONS N"] = mape_ONS_N
                previsao_passada["MAPE ONS SIN"] = mape_ONS_SIN
            except:
                True
        
            previsao_passada["MAE MLP Seco"] = mae_MLP_SECO
            previsao_passada["MAE MLP S"] = mae_MLP_S
            previsao_passada["MAE MLP NE"] = mae_MLP_NE
            previsao_passada["MAE MLP N"] = mae_MLP_N
            previsao_passada["MAE MLP SIN"] = mae_MLP_SIN

            previsao_passada["MAE LSTM Seco"] = mae_LSTM_SECO
            previsao_passada["MAE LSTM S"] = mae_LSTM_S
            previsao_passada["MAE LSTM NE"] = mae_LSTM_NE
            previsao_passada["MAE LSTM N"] = mae_LSTM_N
            previsao_passada["MAE LSTM SIN"] = mae_LSTM_SIN

            previsao_passada["MAE GRU Seco"] = mae_GRU_SECO
            previsao_passada["MAE GRU S"] = mae_GRU_S
            previsao_passada["MAE GRU NE"] = mae_GRU_NE
            previsao_passada["MAE GRU N"] = mae_GRU_N
            previsao_passada["MAE GRU SIN"] = mae_GRU_SIN

            try:
                previsao_passada["MAE ONS Seco"] = mae_ONS_SECO
                previsao_passada["MAE ONS S"] = mae_ONS_S
                previsao_passada["MAE ONS NE"] = mae_ONS_NE
                previsao_passada["MAE ONS N"] = mae_ONS_N
                previsao_passada["MAE ONS SIN"] = mae_ONS_SIN
            except:
                True

            previsao_passada = previsao_passada.drop(columns=previsao_passada.columns[0])
            previsao_passada = round(previsao_passada,2)
            
            if horizonte_previsao ==1:
                previsao_passada.to_csv(caminho_previsao +str(antes_ontem_lstm.year)+'/'+ '%02d' % antes_ontem_lstm.month +'/'+str(antes_ontem)+'/daily/'+data_prevista+".csv",sep = ';',decimal=',')
            else:
                previsao_passada.to_csv(caminho_previsao+str(antes_ontem_lstm_weekly.year)+'/'+ '%02d' % antes_ontem_lstm_weekly.month+'/'+str(antes_ontem_weekly)+'/weekly/'+data_prevista+'.csv',sep = ';',decimal=',')

            ################################################################################################################################################################################################################
            
            if horizonte_previsao == 1:#plota grafico
                hora_prevista = (date.today() -timedelta(days=atraso +horizonte_previsao_MLP))
                hora_d = datetime(hora_prevista.year,hora_prevista.month,hora_prevista.day)

                h = []
                for i in range(48):
                    h.append((hora_d+timedelta(minutes=i*30)).strftime("%H:%M"))

                plot = previsao_passada
                plot['d'] = h

                fig = go.Figure()

                fig.add_traces(
                    list(px.line(
                    previsao_passada,
                    x = "d",
                    y = ['MLP_SIN'],
                    hover_data=  ["MAPE MLP SIN","MAPE MLP Seco","MAPE MLP S","MAPE MLP NE","MAPE MLP N"],
                    markers = True,color_discrete_map={
                                "MLP_SIN": "blue"}
                ).select_traces())
                )

                fig.add_traces(
                    list(px.line(
                    previsao_passada,
                    x = "d",
                    y = ['LSTM_SIN'],
                    hover_data=  ["MAPE LSTM SIN","MAPE LSTM Seco","MAPE LSTM S","MAPE LSTM NE","MAPE LSTM N"],
                    markers = True,color_discrete_map={
                                "LSTM_SIN": "purple"}
                ).select_traces())
                )

                fig.add_traces(
                    list(px.line(
                    previsao_passada,
                    x = "d",
                    y = ['GRU_SIN'],
                    hover_data=  ["MAPE GRU SIN","MAPE GRU Seco","MAPE GRU S","MAPE GRU NE","MAPE GRU N"],
                    markers = True,color_discrete_map={
                                "GRU_SIN": "green"}
                ).select_traces())
                )

                fig.add_traces(
                    list(px.line(
                    previsao_passada,
                    x = "d",
                    y = ['Prev ONS_SIN'],
                    hover_data=  ["MAPE ONS SIN","MAPE ONS Seco","MAPE ONS S","MAPE ONS NE","MAPE ONS N"],
                    markers = True,color_discrete_map={
                                "Prev ONS_SIN": "orange"}
                ).select_traces())
                )

                fig.add_traces(
                    list(px.line(
                    previsao_passada,
                    x = "d",
                    y = ['SIN Carga Verificada'],
                    markers = True,color_discrete_map={
                                "SIN Carga Verificada": "red"}
                ).select_traces())
                )
                fig.update_layout(title='Carga Prevista para o dia '+data_prevista,
                                xaxis_title='Hora',
                                yaxis_title='Carga [MW]')

                fig.write_html("/home/previsorpld-back/app/Plugins/DESSEM/Carga/v2/Prev_Graficos/"+data_prevista+".html")
            return previsao_passada
    

def carga_decomp(previsao,atraso,horizonte_previsao,caminho_pasta):

    horizonte_previsao = 1

    hoje = (date.today() -timedelta(days=atraso)).isoformat()
    hoje_lstm = date.today() -timedelta(days=atraso)
    ontem = (date.today() -timedelta(days=atraso +1)).isoformat()
    ontem_lstm = (date.today() -timedelta(days=atraso +1))
    antes_ontem = (date.today() -timedelta(days=atraso +2)).isoformat()
    antes_ontem_weekly = (date.today() -timedelta(days=atraso +1+horizonte_previsao)).isoformat()
    antes_ontem_lstm = (date.today() -timedelta(days=atraso +2))
    antes_ontem_lstm_weekly = (date.today() -timedelta(days=atraso +1+horizonte_previsao))
    
    data_previsao = (date.today() +timedelta(days=-atraso +1)).isoformat()

    caminho_arquivos = caminho_pasta +'DeckCorrigido/'
    caminho_previsao = caminho_pasta + 'Previsoes/'
        
    patamares = pd.read_excel(caminho_pasta + "Patamares_Carga.xlsx")

    index_patamar_ini = patamares.loc[patamares.DIA == str(data_previsao)].index.tolist()[0]
    patamares_prev = patamares.loc[index_patamar_ini:index_patamar_ini+24*7-1].reset_index(drop = True)

    patamar = []
    cont_leve,cont_media,cont_pesada = 0,0,0
    for i in range(len(patamares_prev)):
        if patamares_prev.PATAMAR[i] =='LEVE':
            cont_leve =  cont_leve + 1
            for j in range(2):
                patamar.append('leve')
        if patamares_prev.PATAMAR[i] =='MÉDIO':
            cont_media =  cont_media + 1
            for j in range(2):
                patamar.append('media')
        if patamares_prev.PATAMAR[i] =='PESADO':
            cont_pesada =  cont_pesada + 1
            for j in range(2):
                patamar.append('pesada')

    previsao['patamares'] = patamar

    prev_leve = previsao.loc[previsao.patamares == 'leve']
    prev_media = previsao.loc[previsao.patamares == 'media']
    prev_pesada = previsao.loc[previsao.patamares == 'pesada']

    prev_leve_LSTM_SECO,prev_media_LSTM_SECO,prev_pesada_LSTM_SECO = prev_leve.LSTM_Seco.mean(),prev_media.LSTM_Seco.mean(),prev_pesada.LSTM_Seco.mean()
    prev_leve_LSTM_S,prev_media_LSTM_S,prev_pesada_LSTM_S = prev_leve.LSTM_S.mean(),prev_media.LSTM_S.mean(),prev_pesada.LSTM_S.mean()
    prev_leve_LSTM_NE,prev_media_LSTM_NE,prev_pesada_LSTM_NE= prev_leve.LSTM_NE.mean(),prev_media.LSTM_NE.mean(),prev_pesada.LSTM_NE.mean()
    prev_leve_LSTM_N,prev_media_LSTM_N,prev_pesada_LSTM_N = prev_leve.LSTM_N.mean(),prev_media.LSTM_N.mean(),prev_pesada.LSTM_N.mean()

    prev_leve_GRU_SECO,prev_media_GRU_SECO,prev_pesada_GRU_SECO = prev_leve.GRU_Seco.mean(),prev_media.GRU_Seco.mean(),prev_pesada.GRU_Seco.mean()
    prev_leve_GRU_S,prev_media_GRU_S,prev_pesada_GRU_S= prev_leve.GRU_S.mean(),prev_media.GRU_S.mean(),prev_pesada.GRU_S.mean()
    prev_leve_GRU_NE,prev_media_GRU_NE,prev_pesada_GRU_NE = prev_leve.GRU_NE.mean(),prev_media.GRU_NE.mean(),prev_pesada.GRU_NE.mean()
    prev_leve_GRU_N,prev_media_GRU_N,prev_pesada_GRU_N = prev_leve.GRU_N.mean(),prev_media.GRU_N.mean(),prev_pesada.GRU_N.mean()

    prev_leve_MLP_SECO,prev_media_MLP_SECO,prev_pesada_MLP_SECO = prev_leve.MLP_Seco.mean(),prev_media.MLP_Seco.mean(),prev_pesada.MLP_Seco.mean()
    prev_leve_MLP_S,prev_media_MLP_S,prev_pesada_MLP_S = prev_leve.MLP_S.mean(),prev_media.MLP_S.mean(),prev_pesada.MLP_S.mean()
    prev_leve_MLP_NE,prev_media_MLP_NE,prev_pesada_MLP_NE = prev_leve.MLP_NE.mean(),prev_media.MLP_NE.mean(),prev_pesada.MLP_NE.mean()
    prev_leve_MLP_N,prev_media_MLP_N,prev_pesada_MLP_N = prev_leve.MLP_N.mean(),prev_media.MLP_N.mean(),prev_pesada.MLP_N.mean()

    prev_leve_ONS_SECO,prev_media_ONS_SECO,prev_pesada_ONS_SECO = prev_leve.Prev_ONS_Seco.mean(),prev_media.Prev_ONS_Seco.mean(),prev_pesada.Prev_ONS_Seco.mean()
    prev_leve_ONS_S,prev_media_ONS_S,prev_pesada_ONS_S = prev_leve.Prev_ONS_S.mean(),prev_media.Prev_ONS_S.mean(),prev_pesada.Prev_ONS_S.mean()
    prev_leve_ONS_NE,prev_media_ONS_NE,prev_pesada_ONS_NE = prev_leve.Prev_ONS_NE.mean(),prev_media.Prev_ONS_NE.mean(),prev_pesada.Prev_ONS_NE.mean()
    prev_leve_ONS_N,prev_media_ONS_N,prev_pesada_ONS_N = prev_leve.Prev_ONS_N.mean(),prev_media.Prev_ONS_N.mean(),prev_pesada.Prev_ONS_N.mean()

    patamares_lstm = [prev_leve_LSTM_SECO,prev_media_LSTM_SECO,prev_pesada_LSTM_SECO,prev_leve_LSTM_S,prev_media_LSTM_S,prev_pesada_LSTM_S,prev_leve_LSTM_NE,prev_media_LSTM_NE,prev_pesada_LSTM_NE,prev_leve_LSTM_N,prev_media_LSTM_N,prev_pesada_LSTM_N]
    patamares_gru = [prev_leve_GRU_SECO,prev_media_GRU_SECO,prev_pesada_GRU_SECO,prev_leve_GRU_S,prev_media_GRU_S,prev_pesada_GRU_S,prev_leve_GRU_NE,prev_media_GRU_NE,prev_pesada_GRU_NE,prev_leve_GRU_N,prev_media_GRU_N,prev_pesada_GRU_N]
    patamares_mlp = [prev_leve_MLP_SECO,prev_media_MLP_SECO,prev_pesada_MLP_SECO,prev_leve_MLP_S,prev_media_MLP_S,prev_pesada_MLP_S,prev_leve_MLP_NE,prev_media_MLP_NE,prev_pesada_MLP_NE,prev_leve_MLP_N,prev_media_MLP_N,prev_pesada_MLP_N]
    patamares_ons = [prev_leve_ONS_SECO,prev_media_ONS_SECO,prev_pesada_ONS_SECO,prev_leve_ONS_S,prev_media_ONS_S,prev_pesada_ONS_S,prev_leve_ONS_NE,prev_media_ONS_NE,prev_pesada_ONS_NE,prev_leve_ONS_N,prev_media_ONS_N,prev_pesada_ONS_N]

    indice = ['Leve SECO','Media SECO','Pesada SECO','Leve S','Media S','Pesada S','Leve NE','Media NE','Pesada NE','Leve N','Media N','Pesada N']

    patamares_decomp = pd.DataFrame()
    patamares_decomp['Patamares'] = indice
    patamares_decomp['MLP'] = patamares_mlp
    patamares_decomp['LSTM'] = patamares_lstm
    patamares_decomp['GRU'] = patamares_gru
    patamares_decomp['ONS'] = patamares_ons
    patamares_decomp = patamares_decomp.set_index('Patamares')

    prev_leve_LSTM_SECO_ordenada,prev_media_LSTM_SECO_ordenada,prev_pesada_LSTM_SECO_ordenada = previsao.LSTM_Seco.sort_values()[:160].mean(),previsao.LSTM_Seco.sort_values()[160:160+96].mean(),previsao.LSTM_Seco.sort_values()[160+96:].mean()
    prev_leve_LSTM_S_ordenada,prev_media_LSTM_S_ordenada,prev_pesada_LSTM_S_ordenada = previsao.LSTM_S.sort_values()[:160].mean(),previsao.LSTM_S.sort_values()[160:160+96].mean(),previsao.LSTM_S.sort_values()[160+96:].mean()
    prev_leve_LSTM_NE_ordenada,prev_media_LSTM_NE_ordenada,prev_pesada_LSTM_NE_ordenada = previsao.LSTM_NE.sort_values()[:160].mean(),previsao.LSTM_NE.sort_values()[160:160+96].mean(),previsao.LSTM_NE.sort_values()[160+96:].mean()
    prev_leve_LSTM_N_ordenada,prev_media_LSTM_N_ordenada,prev_pesada_LSTM_N_ordenada = previsao.LSTM_N.sort_values()[:160].mean(),previsao.LSTM_N.sort_values()[160:160+96].mean(),previsao.LSTM_N.sort_values()[160+96:].mean()

    prev_leve_MLP_SECO_ordenada,prev_media_MLP_SECO_ordenada,prev_pesada_MLP_SECO_ordenada = previsao.MLP_Seco.sort_values()[:160].mean(),previsao.MLP_Seco.sort_values()[160:160+96].mean(),previsao.MLP_Seco.sort_values()[160+96:].mean()
    prev_leve_MLP_S_ordenada,prev_media_MLP_S_ordenada,prev_pesada_MLP_S_ordenada = previsao.MLP_S.sort_values()[:160].mean(),previsao.MLP_S.sort_values()[160:160+96].mean(),previsao.MLP_S.sort_values()[160+96:].mean()
    prev_leve_MLP_NE_ordenada,prev_media_MLP_NE_ordenada,prev_pesada_MLP_NE_ordenada = previsao.MLP_NE.sort_values()[:160].mean(),previsao.MLP_NE.sort_values()[160:160+96].mean(),previsao.MLP_NE.sort_values()[160+96:].mean()
    prev_leve_MLP_N_ordenada,prev_media_MLP_N_ordenada,prev_pesada_MLP_N_ordenada = previsao.MLP_N.sort_values()[:160].mean(),previsao.MLP_N.sort_values()[160:160+96].mean(),previsao.MLP_N.sort_values()[160+96:].mean()

    prev_leve_GRU_SECO_ordenada,prev_media_GRU_SECO_ordenada,prev_pesada_GRU_SECO_ordenada = previsao.GRU_Seco.sort_values()[:160].mean(),previsao.GRU_Seco.sort_values()[160:160+96].mean(),previsao.GRU_Seco.sort_values()[160+96:].mean()
    prev_leve_GRU_S_ordenada,prev_media_GRU_S_ordenada,prev_pesada_GRU_S_ordenada = previsao.GRU_S.sort_values()[:160].mean(),previsao.GRU_S.sort_values()[160:160+96].mean(),previsao.GRU_S.sort_values()[160+96:].mean()
    prev_leve_GRU_NE_ordenada,prev_media_GRU_NE_ordenada,prev_pesada_GRU_NE_ordenada = previsao.GRU_NE.sort_values()[:160].mean(),previsao.GRU_NE.sort_values()[160:160+96].mean(),previsao.GRU_NE.sort_values()[160+96:].mean()
    prev_leve_GRU_N_ordenada,prev_media_GRU_N_ordenada,prev_pesada_GRU_N_ordenada = previsao.GRU_N.sort_values()[:160].mean(),previsao.GRU_N.sort_values()[160:160+96].mean(),previsao.GRU_N.sort_values()[160+96:].mean()

    patamares_lstm_ordenada = [prev_leve_LSTM_SECO_ordenada,prev_media_LSTM_SECO_ordenada,prev_pesada_LSTM_SECO_ordenada,prev_leve_LSTM_S_ordenada,prev_media_LSTM_S_ordenada,prev_pesada_LSTM_S_ordenada,prev_leve_LSTM_NE_ordenada,prev_media_LSTM_NE_ordenada,prev_pesada_LSTM_NE_ordenada,prev_leve_LSTM_N_ordenada,prev_media_LSTM_N_ordenada,prev_pesada_LSTM_N_ordenada]
    patamares_gru_ordenada = [prev_leve_GRU_SECO_ordenada,prev_media_GRU_SECO_ordenada,prev_pesada_GRU_SECO_ordenada,prev_leve_GRU_S_ordenada,prev_media_GRU_S_ordenada,prev_pesada_GRU_S_ordenada,prev_leve_GRU_NE_ordenada,prev_media_GRU_NE_ordenada,prev_pesada_GRU_NE_ordenada,prev_leve_GRU_N_ordenada,prev_media_GRU_N_ordenada,prev_pesada_GRU_N_ordenada]
    patamares_mlp_ordenada = [prev_leve_MLP_SECO_ordenada,prev_media_MLP_SECO_ordenada,prev_pesada_MLP_SECO_ordenada,prev_leve_MLP_S_ordenada,prev_media_MLP_S_ordenada,prev_pesada_MLP_S_ordenada,prev_leve_MLP_NE_ordenada,prev_media_MLP_NE_ordenada,prev_pesada_MLP_NE_ordenada,prev_leve_MLP_N_ordenada,prev_media_MLP_N_ordenada,prev_pesada_MLP_N_ordenada]

    patamares_decomp_ordenada = pd.DataFrame()
    patamares_decomp_ordenada['Patamares'] = indice
    patamares_decomp_ordenada['MLP'] = patamares_mlp_ordenada
    patamares_decomp_ordenada['LSTM'] = patamares_lstm_ordenada
    patamares_decomp_ordenada['GRU'] = patamares_gru_ordenada
    patamares_decomp_ordenada = patamares_decomp_ordenada.set_index('Patamares')

    return patamares_decomp,patamares_decomp_ordenada
