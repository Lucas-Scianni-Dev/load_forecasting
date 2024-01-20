import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date
from datetime import datetime
from datetime import timedelta
import timeit


def Tratamento_dos_dados(arquivo_carga,arquivo_temp_hist,feriados,horizonte_prev):
    #Carrega os dados e junta os dados de carga e temperatura com base na data    
    ano_carga = arquivo_carga["Ano"] 
    mes_carga = arquivo_carga["Mes"]
    dia_carga = arquivo_carga["Dia"]

    ano_temp = arquivo_temp_hist["Ano"]
    mes_temp = arquivo_temp_hist["Mes"]
    dia_temp = arquivo_temp_hist["Dia"]

    carga_h = pd.DataFrame()
    temp_h = pd.DataFrame()

    carga_h["Hora"] = arquivo_carga["Hora"]
    carga_h["Carga"] = arquivo_carga["Carga"]

    temp_h["Hora"] = arquivo_temp_hist["Hora"]
    temp_h["Temperatura"] = arquivo_temp_hist["Temperatura"]

    data_carga = []
    data_temp = []
    for i in range(len(ano_carga)):
      Data_i=dt.date(int(ano_carga[i]), int(mes_carga[i]), int(dia_carga[i]))
      data_carga.append(Data_i)

    for i in range(len(ano_temp)):
      Data_i=dt.date(int(ano_temp[i]), int(mes_temp[i]), int(dia_temp[i]))
      data_temp.append(Data_i)

    carga_h["time"] = data_carga
    carga = carga_h.set_index('time')

    temp_h["time"] = data_temp
    temp_h = temp_h.set_index('time')

    dataset = carga.merge(temp_h, on = ['time','Hora'])

    dataset.reset_index(inplace=True)

        
    #Separa os dados de tempo do dataframe montado anteriormente e cria uma coluna para cada novamente
    dia_semana = []
    mes = []
    dia_mes = []
    for i in range(len(dataset["time"])):
      dia_semana_i = dataset["time"][i].weekday()
      dia_semana.append(dia_semana_i)
      mes_i = dataset["time"][i].strftime("%m")
      mes.append(mes_i)
      dia_i = dataset["time"][i].strftime("%d")
      dia_mes.append(dia_i)

    dados = pd.DataFrame() 
    dados["data"] = dataset["time"]
    dados["mes"] = mes
    dados["dia_mes"] = dia_mes
    dados["dia_semana"] = dia_semana
    dados["hora"] = dataset["Hora"]
    dados["Temperatura"] = dataset["Temperatura"]
    dados["carga"] = dataset["Carga"]

    dataset.drop(['time','Hora','Temperatura','Carga'],axis=1,inplace = True)

    #converte os dados para float
    dados["mes"] = dados["mes"].astype(float)
    dados["dia_mes"] = dados["dia_mes"].astype(float)
    dados["dia_semana"] = dados["dia_semana"].astype(float)
    dados["hora"] = dados["hora"].values.astype(np.int)

    #dados["carga"] = dados["carga"].str.replace(',','.')
    dados["carga"] = dados["carga"].astype(float)

    #dados["Temperatura"] = dados["Temperatura"].str.replace(',','.')
    dados["Temperatura"] = dados["Temperatura"].astype(float)


    dados.reset_index(inplace = True, drop = True) 

    ano_feriados = feriados["Ano"] 
    mes_feriados = feriados["Mes"]
    dia_feriados = feriados["Dia"]

    data_feriados = []
    for i in range(len(ano_feriados)):
        Data_i=dt.date(int(ano_feriados[i]), int(mes_feriados[i]), int(dia_feriados[i]))
        data_feriados.append(Data_i)

    datas_diarias = []
    for i in range(int((dados['data'].shape[0])/24)):
        datas_diarias.append(dados['data'][i*24])

    col_fer = np.zeros(len(datas_diarias))

    for i in range(len(ano_feriados)):
        for j in range(len(datas_diarias)):
            if datas_diarias[j] == data_feriados[i]:
                col_fer[j] = 1

    fer = pd.DataFrame(datas_diarias,columns = ['data'])
    fer['cod'] = col_fer

    col_feriados = np.zeros(dados['data'].shape[0])

    for i in range(len(datas_diarias)):
        col_feriados[i*24:(i+1)*24] = fer['cod'][i]
        
    #one hot encoding 
    mes = dados["mes"].astype(int)
    dia_semana = dados["dia_semana"].astype(int)
    dia_mes = dados["dia_mes"].astype(int)
    hora = dados["hora"].astype(int)

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

    one_hot_enc = []
    for i in range(len(mes)):
      one_hot_i = One_Hot(mes[i],dia_semana[i],dia_mes[i],col_feriados[i])  
      one_hot_enc.append(one_hot_i) 
        

    one_hot_enc = pd.DataFrame(one_hot_enc, columns = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24','col25','col26','col27','col28','col29','col30','col31','col32','col33','col34','col35','col36','col37','col38','col39','col40','col41','col42','col43','col44','col45','col46','col47','col48','col49','col50'])

  
    #Monta dataframe com os dados codificados
    dados_enc = pd.DataFrame() 
    dados_enc = dados_enc.append(one_hot_enc)
    dados_enc["temp"] = dados["Temperatura"]
    dados_enc["carga"] = dados["carga"]

    #Normaliza os valores de carga e temperatura e interpola linhas com dados faltantes
    c_min=np.min(dados_enc["carga"])
    c_std = np.std(dados_enc["carga"])
    dados_enc["carga"] = (dados_enc["carga"]-c_min)/c_std

    t_min=np.min(dados_enc["temp"])
    t_std = np.std(dados_enc["temp"])
    dados_enc["temp"] = (dados_enc["temp"]-t_min)/t_std

    dados_enc.interpolate(method='linear', limit_direction='forward', inplace=True)

    datas = dados["data"]

    #print(dados_enc.shape)

    var_norm = {"c_min":c_min,"c_std":c_std,"t_min":t_min, "t_std":t_std}
    return dados_enc, one_hot_enc, datas, var_norm

def Tratamento_dos_dados_diarios(arquivo_carga,arquivo_temp_hist,feriados,horizonte_prev):

    arquivo_carga_diaria = pd.DataFrame(columns=['Ano','Mes','Dia','Carga'])
    arquivo_temp_diaria = pd.DataFrame(columns=['Ano','Mes','Dia','Temperatura','Max','Min'])

    for i in range(int(len(arquivo_carga)/24)):
        ano = arquivo_carga.loc[i*24].Ano
        mes = arquivo_carga.loc[i*24].Mes
        dia = arquivo_carga.loc[i*24].Dia
        
        carga_media = []
        for j in range(24):
            carga_media.append(arquivo_carga.loc[i*24+j].Carga)
        max = np.max(carga_media)
        min = np.min(carga_media)
        carga_media = np.average(carga_media)
        
        data = {"Ano":ano,'Mes':mes,'Dia':dia,'Carga':carga_media,'C_Max':max,'C_Min':min}
        arquivo_carga_diaria = arquivo_carga_diaria.append(data,ignore_index=True)   
    
    for i in range(int(len(arquivo_temp_hist)/24)):
        ano = arquivo_temp_hist.loc[i*24].Ano
        mes = arquivo_temp_hist.loc[i*24].Mes
        dia = arquivo_temp_hist.loc[i*24].Dia
        
        temp_media = []
        for j in range(24):
            temp_media.append(arquivo_temp_hist.loc[i*24+j].Temperatura)
        max = np.max(temp_media)
        min = np.min(temp_media)
        temp_media = np.average(temp_media)
        
        data = {"Ano":ano,'Mes':mes,'Dia':dia,'Temperatura':temp_media,'T_Max':max,'T_Min':min}
        arquivo_temp_diaria = arquivo_temp_diaria.append(data,ignore_index=True) 

    #Carrega os dados e junta os dados de carga e temperatura com base na data    
    ano_carga = arquivo_carga_diaria["Ano"] 
    mes_carga = arquivo_carga_diaria["Mes"]
    dia_carga = arquivo_carga_diaria["Dia"]

    ano_temp = arquivo_temp_diaria["Ano"]
    mes_temp = arquivo_temp_diaria["Mes"]
    dia_temp = arquivo_temp_diaria["Dia"]

    carga_h = pd.DataFrame()
    temp_h = pd.DataFrame()

    carga_h["Carga"] = arquivo_carga_diaria["Carga"]
    carga_h['C_Max'] = arquivo_carga_diaria["C_Max"]
    carga_h['C_Min'] = arquivo_carga_diaria["C_Min"]

    temp_h["Temperatura"] = arquivo_temp_diaria["Temperatura"]
    temp_h['T_Max'] = arquivo_temp_diaria["T_Max"]
    temp_h['T_Min'] = arquivo_temp_diaria["T_Min"]

    data_carga = []
    data_temp = []
    for i in range(len(ano_carga)):
      Data_i=dt.date(int(ano_carga[i]), int(mes_carga[i]), int(dia_carga[i]))
      data_carga.append(Data_i)

    for i in range(len(ano_temp)):
      Data_i=dt.date(int(ano_temp[i]), int(mes_temp[i]), int(dia_temp[i]))
      data_temp.append(Data_i)

    carga_h["time"] = data_carga
    carga = carga_h.set_index('time')

    temp_h["time"] = data_temp
    temp_h = temp_h.set_index('time')

    dataset = carga.merge(temp_h, on = ['time'])

    dataset.reset_index(inplace=True)

    #Separa os dados de tempo do dataframe montado anteriormente e cria uma coluna para cada novamente
    dia_semana = []
    mes = []
    dia_mes = []
    for i in range(len(dataset["time"])):
      dia_semana_i = dataset["time"][i].weekday()
      dia_semana.append(dia_semana_i)
      mes_i = dataset["time"][i].strftime("%m")
      mes.append(mes_i)
      dia_i = dataset["time"][i].strftime("%d")
      dia_mes.append(dia_i)

    dados = pd.DataFrame() 
    dados["data"] = dataset["time"]
    dados["mes"] = mes
    dados["dia_mes"] = dia_mes
    dados["dia_semana"] = dia_semana
    dados["Temperatura"] = dataset["Temperatura"]
    dados['T_Max'] = dataset["T_Max"]
    dados['T_Min'] = dataset["T_Min"]
    dados["carga"] = dataset["Carga"]
    dados['C_Max'] = dataset["C_Max"]
    dados['C_Min'] = dataset["C_Min"]

    #converte os dados para float
    dados["mes"] = dados["mes"].astype(float)
    dados["dia_mes"] = dados["dia_mes"].astype(float)
    dados["dia_semana"] = dados["dia_semana"].astype(float)

    #dados["carga"] = dados["carga"].str.replace(',','.')
    dados["carga"] = dados["carga"].astype(float)
    dados["C_Max"] = dados["C_Max"].astype(float)
    dados["C_Min"] = dados["C_Min"].astype(float)

    #dados["Temperatura"] = dados["Temperatura"].str.replace(',','.')
    dados["Temperatura"] = dados["Temperatura"].astype(float)
    dados["T_Max"] = dados["T_Max"].astype(float)
    dados["T_Min"] = dados["T_Min"].astype(float)


    dados.reset_index(inplace = True, drop = True) 

    ano_feriados = feriados["Ano"] 
    mes_feriados = feriados["Mes"]
    dia_feriados = feriados["Dia"]

    data_feriados = []
    for i in range(len(ano_feriados)):
        Data_i=dt.date(int(ano_feriados[i]), int(mes_feriados[i]), int(dia_feriados[i]))
        data_feriados.append(Data_i)

    datas_diarias = []
    for i in range(int((dados['data'].shape[0])/24)):
        datas_diarias.append(dados['data'][i*24])

    col_fer = np.zeros(len(datas_diarias))

    for i in range(len(ano_feriados)):
        for j in range(len(datas_diarias)):
            if datas_diarias[j] == data_feriados[i]:
                col_fer[j] = 1

    fer = pd.DataFrame(datas_diarias,columns = ['data'])
    fer['cod'] = col_fer

    col_feriados = np.zeros(dados['data'].shape[0])

    for i in range(len(datas_diarias)):
        col_feriados[i*24:(i+1)*24] = fer['cod'][i]
        
    #one hot encoding 
    mes = dados["mes"].astype(int)
    dia_semana = dados["dia_semana"].astype(int)
    dia_mes = dados["dia_mes"].astype(int)

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

    one_hot_enc = []
    for i in range(len(mes)):
      one_hot_i = One_Hot(mes[i],dia_semana[i],dia_mes[i],col_feriados[i])  
      one_hot_enc.append(one_hot_i) 
        

    one_hot_enc = pd.DataFrame(one_hot_enc, columns = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24','col25','col26','col27','col28','col29','col30','col31','col32','col33','col34','col35','col36','col37','col38','col39','col40','col41','col42','col43','col44','col45','col46','col47','col48','col49','col50'])

    #Monta dataframe com os dados codificados
    dados_enc = pd.DataFrame() 
    dados_enc = dados_enc.append(one_hot_enc)
    dados_enc["temp"] = dados["Temperatura"]
    dados_enc["temp_max"] = dados["T_Max"]
    dados_enc["temp_min"] = dados["T_Min"]
    dados_enc["carga"] = dados["carga"]
    dados_enc["carga_max"] = dados["C_Max"]
    dados_enc["carga_min"] = dados["C_Min"]

    #Normaliza os valores de carga e temperatura e interpola linhas com dados faltantes
    c_min=np.min(dados_enc["carga_min"])
    c_std = np.std(dados_enc["carga"])
    dados_enc["carga"] = (dados_enc["carga"]-c_min)/c_std
    dados_enc["carga_max"] = (dados_enc["carga_max"]-c_min)/c_std
    dados_enc["carga_min"] = (dados_enc["carga_min"]-c_min)/c_std

    t_min=np.min(dados_enc["temp_min"])
    t_std = np.std(dados_enc["temp"])
    dados_enc["temp"] = (dados_enc["temp"]-t_min)/t_std
    dados_enc["temp_max"] = (dados_enc["temp_max"]-t_min)/t_std
    dados_enc["temp_min"] = (dados_enc["temp_min"]-t_min)/t_std

    dados_enc.interpolate(method='linear', limit_direction='forward', inplace=True)

    datas = dados["data"]

    #print(dados_enc.shape)

    var_norm = {"c_min":c_min,"c_std":c_std,"t_min":t_min, "t_std":t_std}
    return dados_enc, one_hot_enc, datas, var_norm
   
