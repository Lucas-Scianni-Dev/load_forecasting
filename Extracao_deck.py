import pandas as pd
import numpy as np
from datetime import date, datetime
from datetime import timedelta,timezone
import os
from zipfile import ZipFile
import shutil
import time
import json
import functools
print = functools.partial(print, flush=True)

def Extrair(hoje_date_format,caminho_arquivos,caminho_pasta):

    hoje_corrigido = hoje_date_format.isoformat()
    arquivo_base = "Deck_" + hoje_corrigido +".zip"
    arquivo_previsao = "PrevCargaDESSEM_" + hoje_corrigido +".zip"

    f = open('./env.json',)
    dataenv = json.load(f)
    f.close()
    pasta_arquivos = dataenv['paths']['decks_ons']
    # pasta_arquivos = "caminho da pasta com os arquivos"

    arquivo_base_pasta_download = pasta_arquivos + "Deck_" + hoje_corrigido +".zip"
    arquivo_base_pasta_correta = caminho_pasta + "Deck_" + hoje_corrigido +".zip"
    shutil.copy(arquivo_base_pasta_download,arquivo_base_pasta_correta)
    os.chdir(caminho_pasta)

    arquivo_previsao_pasta_download = pasta_arquivos +"PrevCargaDESSEM_" + hoje_corrigido +".zip"
    arquivo_previsao_pasta_correta = caminho_pasta +"PrevCargaDESSEM_" + hoje_corrigido +".zip"
    shutil.copy(arquivo_previsao_pasta_download,arquivo_previsao_pasta_correta)
    os.chdir(caminho_pasta)

    arquivo_seco = "Deck_SECO_"+ hoje_corrigido +".zip"
    arquivo_s = "Deck_S_"+ hoje_corrigido +".zip"
    arquivo_ne = "Deck_NE_"+ hoje_corrigido +".zip"
    arquivo_n = "Deck_N_"+ hoje_corrigido +".zip"
   
    
    deck = caminho_pasta + arquivo_base
    z_deck = ZipFile(deck,'r')
    z_deck.extractall()
    z_deck.close()
    
    previsao = caminho_pasta + arquivo_previsao
    z_previsao = ZipFile(previsao,'r')
    z_previsao.extractall()
    z_previsao.close()
    
    Seco  = caminho_pasta + arquivo_seco
    S = caminho_pasta + arquivo_s
    NE = caminho_pasta + arquivo_ne
    N = caminho_pasta + arquivo_n
    
    z_seco = ZipFile(Seco,'r')
    z_s = ZipFile(S,'r')
    z_ne = ZipFile(NE,'r')
    z_n = ZipFile(N,'r')
    
    z_seco.extract('SECO_'+hoje_corrigido+'_CARGAHIST.csv')
    z_seco.extract('SECO_'+hoje_corrigido+'_EXOGENAHIST.csv')
    z_seco.extract('SECO_'+hoje_corrigido+'_EXOGENAPREV.csv')
    z_seco.extract('SECO_'+hoje_corrigido+'_FERIADOS.csv')
    
    z_s.extract('S_'+hoje_corrigido+'_CARGAHIST.csv')
    z_s.extract('S_'+hoje_corrigido+'_EXOGENAHIST.csv')
    z_s.extract('S_'+hoje_corrigido+'_EXOGENAPREV.csv')
    z_s.extract('S_'+hoje_corrigido+'_FERIADOS.csv')
    
    z_ne.extract('NE_'+hoje_corrigido+'_CARGAHIST.csv')
    z_ne.extract('NE_'+hoje_corrigido+'_EXOGENAHIST.csv')
    z_ne.extract('NE_'+hoje_corrigido+'_EXOGENAPREV.csv')
    z_ne.extract('NE_'+hoje_corrigido+'_FERIADOS.csv')
    
    z_n.extract('N_'+hoje_corrigido+'_CARGAHIST.csv')
    z_n.extract('N_'+hoje_corrigido+'_EXOGENAHIST.csv')
    z_n.extract('N_'+hoje_corrigido+'_EXOGENAPREV.csv')
    z_n.extract('N_'+hoje_corrigido+'_FERIADOS.csv')
    
    
def Monta_deck(hoje_date_format,diferenca,ontem,ultima_atualizacao,caminho_arquivos,caminho_pasta):    
    hoje_corrigido = hoje_date_format.isoformat()

    seco_carga_hist_corrigida = pd.read_csv(caminho_arquivos+'SECO/seco_carga_deck_'+ontem+'.csv',delimiter=',')            
    seco_temp_hist_corrigida = pd.read_csv(caminho_arquivos+'SECO/seco_temp_deck_'+ontem+'.csv',delimiter=',')
    
    s_carga_hist_corrigida = pd.read_csv(caminho_arquivos+'S/s_carga_deck_'+ontem+'.csv',delimiter=',')            
    s_temp_hist_corrigida = pd.read_csv(caminho_arquivos+'S/s_temp_deck_'+ontem+'.csv',delimiter=',')
    
    ne_carga_hist_corrigida = pd.read_csv(caminho_arquivos+'NE/ne_carga_deck_'+ontem+'.csv',delimiter=',')            
    ne_temp_hist_corrigida = pd.read_csv(caminho_arquivos+'NE/ne_temp_deck_'+ontem+'.csv',delimiter=',')
    
    n_carga_hist_corrigida = pd.read_csv(caminho_arquivos+'N/n_carga_deck_'+ontem+'.csv',delimiter=',')            
    n_temp_hist_corrigida = pd.read_csv(caminho_arquivos+'N/n_temp_deck_'+ontem+'.csv',delimiter=',')
    
    #########################################################################################
    
    carga_seco = caminho_pasta +'SECO_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_seco = caminho_pasta +'SECO_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_seco = caminho_pasta +'SECO_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_seco = caminho_pasta +'PrevCargaDESSEM_SECO_'+hoje_corrigido+'.csv'
    feriados_seco = caminho_pasta +'SECO_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_s = caminho_pasta +'S_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_s = caminho_pasta +'S_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_s = caminho_pasta +'S_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_s = caminho_pasta +'PrevCargaDESSEM_S_'+hoje_corrigido+'.csv'
    feriados_s = caminho_pasta +'S_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_ne = caminho_pasta +'NE_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_ne = caminho_pasta +'NE_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_ne = caminho_pasta +'NE_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_ne = caminho_pasta +'PrevCargaDESSEM_NE_'+hoje_corrigido+'.csv'
    feriados_ne = caminho_pasta +'NE_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_n = caminho_pasta +'N_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_n = caminho_pasta +'N_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_n = caminho_pasta +'N_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_n = caminho_pasta +'PrevCargaDESSEM_N_'+hoje_corrigido+'.csv'
    feriados_n = caminho_pasta +'N_'+hoje_corrigido+'_FERIADOS.csv'
    
    #########################################################################################
    def converte_formato(arquivo,variavel,nome):
        ano,mes,dia,hora,min,valor = [],[],[],[],[],[]
        for i in range(len(arquivo)):
            data = datetime.strptime(datetime.fromisoformat(arquivo.DataHora[i][:-1]).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0], '%Y-%m-%d').date()
            horas = datetime.fromisoformat(arquivo.DataHora[i][:-1]).strftime('%Y-%m-%d %H:%M:%S').split(' ')[1]
            val = arquivo[variavel][i]
            ano.append(data.year)
            mes.append(data.month)
            dia.append(data.day)
            hora.append(int(horas[:2]))
            valor.append(val)
            min.append(0)
        arq_convertido = pd.DataFrame()
        arq_convertido["Ano"] = ano
        arq_convertido["Mes"] = mes
        arq_convertido["Dia"] = dia
        arq_convertido["Hora"] = hora
        arq_convertido["Min"] = min
        arq_convertido[nome] = valor
        return arq_convertido

    def converte_feriados(arquivo):
        ano,mes,dia = [],[],[]
        for i in range(len(arquivo)):
            data = datetime.strptime(arquivo.Data[i], '%Y-%m-%d').date()
            ano.append(data.year)
            mes.append(data.month)
            dia.append(data.day)
        arq_convertido = pd.DataFrame()
        arq_convertido["Ano"] = ano
        arq_convertido["Mes"] = mes
        arq_convertido["Dia"] = dia
        return arq_convertido

    seco_carga_hist_atual = pd.read_csv(carga_seco,delimiter=';')
    seco_temp_hist_atual = pd.read_csv(temp_seco,delimiter=';')
    seco_temp_prevista = pd.read_csv(temp_prev_seco,delimiter=';')
    seco_carga_hist_atual = converte_formato(seco_carga_hist_atual,'Carga','Carga')
    seco_temp_hist_atual = converte_formato(seco_temp_hist_atual,'Exo_Temperatura','Temperatura')
    seco_temp_prevista = converte_formato(seco_temp_prevista,'Exo_Temperatura','Temperatura')
    seco_carga_prevista = pd.read_csv(carga_prev_seco,delimiter=';')
    seco_carga_prevista = seco_carga_prevista.sort_values('din_referencia')
    seco_carga_prevista.reset_index(drop=True, inplace=True)
    seco_feriados = pd.read_csv(feriados_seco,delimiter=';')
    seco_feriados = converte_feriados(seco_feriados)
    
    s_carga_hist_atual = pd.read_csv(carga_s,delimiter=';')
    s_temp_hist_atual = pd.read_csv(temp_s,delimiter=';')
    s_temp_prevista = pd.read_csv(temp_prev_s,delimiter=';')
    s_carga_hist_atual = converte_formato(s_carga_hist_atual,'Carga','Carga')
    s_temp_hist_atual = converte_formato(s_temp_hist_atual,'Exo_Temperatura','Temperatura')
    s_temp_prevista = converte_formato(s_temp_prevista,'Exo_Temperatura','Temperatura')
    s_carga_prevista = pd.read_csv(carga_prev_s,delimiter=';')
    s_carga_prevista = s_carga_prevista.sort_values('din_referencia')
    s_carga_prevista.reset_index(drop=True, inplace=True)
    s_feriados = pd.read_csv(feriados_s,delimiter=';')
    s_feriados = converte_feriados(s_feriados)
    
    ne_carga_hist_atual = pd.read_csv(carga_ne,delimiter=';')
    ne_temp_hist_atual = pd.read_csv(temp_ne,delimiter=';')
    ne_temp_prevista = pd.read_csv(temp_prev_ne,delimiter=';')
    ne_carga_hist_atual = converte_formato(ne_carga_hist_atual,'Carga','Carga')
    ne_temp_hist_atual = converte_formato(ne_temp_hist_atual,'Exo_Temperatura','Temperatura')
    ne_temp_prevista = converte_formato(ne_temp_prevista,'Exo_Temperatura','Temperatura')
    ne_carga_prevista = pd.read_csv(carga_prev_ne,delimiter=';')
    ne_carga_prevista = ne_carga_prevista.sort_values('din_referencia')
    ne_carga_prevista.reset_index(drop=True, inplace=True)
    ne_feriados = pd.read_csv(feriados_ne,delimiter=';')
    ne_feriados = converte_feriados(ne_feriados)
    
    n_carga_hist_atual = pd.read_csv(carga_n,delimiter=';')
    n_temp_hist_atual = pd.read_csv(temp_n,delimiter=';')
    n_temp_prevista = pd.read_csv(temp_prev_n,delimiter=';')
    n_carga_hist_atual = converte_formato(n_carga_hist_atual,'Carga','Carga')
    n_temp_hist_atual = converte_formato(n_temp_hist_atual,'Exo_Temperatura','Temperatura')
    n_temp_prevista = converte_formato(n_temp_prevista,'Exo_Temperatura','Temperatura')
    n_carga_prevista = pd.read_csv(carga_prev_n,delimiter=';')
    n_carga_prevista = n_carga_prevista.sort_values('din_referencia')
    n_carga_prevista.reset_index(drop=True, inplace=True)
    n_feriados = pd.read_csv(feriados_n,delimiter=';')
    n_feriados = converte_feriados(n_feriados)
    
    #########################################################################################
    
    seco_indice_c = len(seco_carga_hist_atual) - (diferenca + 1) * 24 - 1
    seco_indice_t = len(seco_temp_hist_atual) - (diferenca + 1) * 24
    
    s_indice_c = len(s_carga_hist_atual) - (diferenca + 1) * 24 - 1
    s_indice_t = len(s_temp_hist_atual) - (diferenca + 1) * 24
    
    ne_indice_c = len(ne_carga_hist_atual) - (diferenca + 1) * 24 - 1
    ne_indice_t = len(ne_temp_hist_atual) - (diferenca + 1) * 24
    
    n_indice_c = len(n_carga_hist_atual) - (diferenca + 1) * 24 - 1
    n_indice_t = len(n_temp_hist_atual) - (diferenca + 1) * 24
    
    #########################################################################################
    
    seco_carga_dia_anterior = seco_carga_hist_atual.iloc[seco_indice_c:seco_indice_c+ (diferenca+1) * 24,:]
    seco_temp_dia_anterior = seco_temp_hist_atual.iloc[seco_indice_t:,:]
    
    s_carga_dia_anterior = s_carga_hist_atual.iloc[s_indice_c:s_indice_c+ (diferenca+1) * 24,:]
    s_temp_dia_anterior = s_temp_hist_atual.iloc[s_indice_t:,:]
    
    ne_carga_dia_anterior = ne_carga_hist_atual.iloc[ne_indice_c:ne_indice_c+ (diferenca+1) * 24,:]
    ne_temp_dia_anterior = ne_temp_hist_atual.iloc[ne_indice_t:,:]
    
    n_carga_dia_anterior = n_carga_hist_atual.iloc[n_indice_c:n_indice_c+ (diferenca+1) * 24,:]
    n_temp_dia_anterior = n_temp_hist_atual.iloc[n_indice_t:,:]
    
    ########################################################################################
    
    seco_carga_final = pd.DataFrame(seco_carga_hist_corrigida)
    seco_temp_final = pd.DataFrame(seco_temp_hist_corrigida)
    seco_temp_prev_final = seco_temp_prevista
    seco_carga_prev_final = seco_carga_prevista["val_previsaocarga"].loc[47:]
    
    s_carga_final = pd.DataFrame(s_carga_hist_corrigida)
    s_temp_final = pd.DataFrame(s_temp_hist_corrigida)
    s_temp_prev_final = s_temp_prevista
    s_carga_prev_final = s_carga_prevista["val_previsaocarga"].loc[47:]
    
    ne_carga_final = pd.DataFrame(ne_carga_hist_corrigida)
    ne_temp_final = pd.DataFrame(ne_temp_hist_corrigida)
    ne_temp_prev_final = ne_temp_prevista
    ne_carga_prev_final = ne_carga_prevista["val_previsaocarga"].loc[47:]
    
    n_carga_final = pd.DataFrame(n_carga_hist_corrigida)
    n_temp_final = pd.DataFrame(n_temp_hist_corrigida)
    n_temp_prev_final = n_temp_prevista
    n_carga_prev_final = n_carga_prevista["val_previsaocarga"].loc[47:]
    
    ########################################################################################
    
    seco_carga_final = seco_carga_final.append(seco_carga_dia_anterior)
    seco_temp_final = seco_temp_final.append(seco_temp_dia_anterior)
    seco_carga_final = seco_carga_final.drop(seco_carga_final.columns[0], axis=1)
    seco_temp_final = seco_temp_final.drop(seco_temp_final.columns[0], axis=1)
    seco_carga_final.reset_index(drop = True,inplace=True)
    seco_temp_final.reset_index(drop=True,inplace=True)
    seco_temp_prev_final.reset_index(drop=True, inplace=True)
    seco_carga_prev_final.reset_index(drop=True, inplace=True)
    
    s_carga_final = s_carga_final.append(s_carga_dia_anterior)
    s_temp_final = s_temp_final.append(s_temp_dia_anterior)
    s_carga_final = s_carga_final.drop(s_carga_final.columns[0], axis=1)
    s_temp_final = s_temp_final.drop(s_temp_final.columns[0], axis=1)
    s_carga_final.reset_index(drop = True,inplace=True)
    s_temp_final.reset_index(drop=True,inplace=True)
    s_temp_prev_final.reset_index(drop=True, inplace=True)
    s_carga_prev_final.reset_index(drop=True, inplace=True)
    
    ne_carga_final = ne_carga_final.append(ne_carga_dia_anterior)
    ne_temp_final = ne_temp_final.append(ne_temp_dia_anterior)
    ne_carga_final = ne_carga_final.drop(ne_carga_final.columns[0], axis=1)
    ne_temp_final = ne_temp_final.drop(ne_temp_final.columns[0], axis=1)
    ne_carga_final.reset_index(drop = True,inplace=True)
    ne_temp_final.reset_index(drop=True,inplace=True)
    ne_temp_prev_final.reset_index(drop=True, inplace=True)
    ne_carga_prev_final.reset_index(drop=True, inplace=True)
    
    n_carga_final = n_carga_final.append(n_carga_dia_anterior)
    n_temp_final = n_temp_final.append(n_temp_dia_anterior)
    n_carga_final = n_carga_final.drop(n_carga_final.columns[0], axis=1)
    n_temp_final = n_temp_final.drop(n_temp_final.columns[0], axis=1)
    n_carga_final.reset_index(drop = True,inplace=True)
    n_temp_final.reset_index(drop=True,inplace=True)
    n_temp_prev_final.reset_index(drop=True, inplace=True)
    n_carga_prev_final.reset_index(drop=True, inplace=True)
      
    #########################################################################################
    
    seco_carga_final.to_csv(caminho_arquivos+'SECO/seco_carga_deck_'+hoje_corrigido+'.csv')
    seco_temp_final.to_csv(caminho_arquivos+'SECO/seco_temp_deck_'+hoje_corrigido+'.csv')
    seco_temp_prev_final.to_csv(caminho_arquivos+'SECO/seco_temp_prev_deck_'+hoje_corrigido+'.csv')
    seco_carga_prev_final.to_csv(caminho_arquivos+'SECO/seco_carga_prev_deck_'+hoje_corrigido+'.csv')
    seco_feriados.to_csv(caminho_arquivos+'SECO/SECO_'+hoje_corrigido+'_FERIADOS.csv')
    
    s_carga_final.to_csv(caminho_arquivos+'S/s_carga_deck_'+hoje_corrigido+'.csv')
    s_temp_final.to_csv(caminho_arquivos+'S/s_temp_deck_'+hoje_corrigido+'.csv')
    s_temp_prev_final.to_csv(caminho_arquivos+'S/s_temp_prev_deck_'+hoje_corrigido+'.csv')
    s_carga_prev_final.to_csv(caminho_arquivos+'S/s_carga_prev_deck_'+hoje_corrigido+'.csv')
    s_feriados.to_csv(caminho_arquivos+'S/S_'+hoje_corrigido+'_FERIADOS.csv')
    
    ne_carga_final.to_csv(caminho_arquivos+'NE/ne_carga_deck_'+hoje_corrigido+'.csv')
    ne_temp_final.to_csv(caminho_arquivos+'NE/ne_temp_deck_'+hoje_corrigido+'.csv')
    ne_temp_prev_final.to_csv(caminho_arquivos+'NE/ne_temp_prev_deck_'+hoje_corrigido+'.csv')
    ne_carga_prev_final.to_csv(caminho_arquivos+'NE/ne_carga_prev_deck_'+hoje_corrigido+'.csv')
    ne_feriados.to_csv(caminho_arquivos+'NE/NE_'+hoje_corrigido+'_FERIADOS.csv')
    
    n_carga_final.to_csv(caminho_arquivos+'N/n_carga_deck_'+hoje_corrigido+'.csv')
    n_temp_final.to_csv(caminho_arquivos+'N/n_temp_deck_'+hoje_corrigido+'.csv')
    n_temp_prev_final.to_csv(caminho_arquivos+'N/n_temp_prev_deck_'+hoje_corrigido+'.csv')
    n_carga_prev_final.to_csv(caminho_arquivos+'N/n_carga_prev_deck_'+hoje_corrigido+'.csv')
    n_feriados.to_csv(caminho_arquivos+'N/N_'+hoje_corrigido+'_FERIADOS.csv')
    
    print("Arquivos modificados com sucesso!")
    
def Apagar(hoje_date_format,ontem,antes_ontem,caminho_arquivos,caminho_pasta):
    hoje_corrigido = hoje_date_format.isoformat()
    arquivo_base = "Deck_" + hoje_corrigido +".zip"
    arquivo_previsao = "PrevCargaDESSEM_" + hoje_corrigido +".zip"
    arquivo_seco = "Deck_SECO_"+hoje_corrigido+".zip"
    arquivo_s = "Deck_S_"+hoje_corrigido+".zip"
    arquivo_ne = "Deck_NE_"+hoje_corrigido+".zip"
    arquivo_n = "Deck_N_"+hoje_corrigido+".zip"
    
    caminho_base = caminho_pasta + arquivo_base
    caminho_previsao = caminho_pasta + arquivo_previsao
    caminho_seco = caminho_pasta + arquivo_seco
    caminho_s =  caminho_pasta + arquivo_s
    caminho_ne =  caminho_pasta + arquivo_ne
    caminho_n =  caminho_pasta + arquivo_n
    
    carga_seco = caminho_pasta + 'SECO_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_seco = caminho_pasta + 'SECO_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_seco = caminho_pasta + 'SECO_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_seco = caminho_pasta + 'PrevCargaDESSEM_SECO_'+hoje_corrigido+'.csv'
    feriados_seco = caminho_pasta +'SECO_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_s = caminho_pasta + 'S_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_s = caminho_pasta + 'S_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_s = caminho_pasta + 'S_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_s = caminho_pasta + 'PrevCargaDESSEM_S_'+hoje_corrigido+'.csv'
    feriados_s = caminho_pasta +'S_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_ne = caminho_pasta + 'NE_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_ne = caminho_pasta + 'NE_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_ne = caminho_pasta + 'NE_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_ne = caminho_pasta + 'PrevCargaDESSEM_NE_'+hoje_corrigido+'.csv'
    feriados_ne = caminho_pasta +'NE_'+hoje_corrigido+'_FERIADOS.csv'
    
    carga_n = caminho_pasta + 'N_'+hoje_corrigido+'_CARGAHIST.csv'
    temp_n = caminho_pasta + 'N_'+hoje_corrigido+'_EXOGENAHIST.csv'
    temp_prev_n = caminho_pasta + 'N_'+hoje_corrigido+'_EXOGENAPREV.csv'
    carga_prev_n = caminho_pasta + 'PrevCargaDESSEM_N_'+hoje_corrigido+'.csv'
    feriados_n = caminho_pasta +'N_'+hoje_corrigido+'_FERIADOS.csv'
    
    arquivo_carga_anterior_seco = caminho_arquivos + 'SECO/seco_carga_deck_'+ontem+'.csv'
    arquivo_temp_anterior_seco = caminho_arquivos + 'SECO/seco_temp_deck_'+ontem+'.csv'
    arquivo_prev_temp_anterior_seco = caminho_arquivos + 'SECO/seco_temp_prev_deck_'+ontem+'.csv'
    arquivo_prev_carga_anterior_seco = caminho_arquivos + 'SECO/seco_carga_prev_deck_'+ontem+'.csv'
    feriados_ontem_seco = caminho_arquivos +'SECO/SECO_'+ontem+'_FERIADOS.csv'
    
    arquivo_carga_anterior_s = caminho_arquivos + 'S/s_carga_deck_'+ontem+'.csv'
    arquivo_temp_anterior_s = caminho_arquivos + 'S/s_temp_deck_'+ontem+'.csv'
    arquivo_prev_temp_anterior_s = caminho_arquivos + 'S/s_temp_prev_deck_'+ontem+'.csv'
    arquivo_prev_carga_anterior_s = caminho_arquivos + 'S/s_carga_prev_deck_'+ontem+'.csv'
    feriados_ontem_s = caminho_arquivos +'S/S_'+ontem+'_FERIADOS.csv'
    
    arquivo_carga_anterior_ne = caminho_arquivos + 'NE/ne_carga_deck_'+ontem+'.csv'
    arquivo_temp_anterior_ne = caminho_arquivos + 'NE/ne_temp_deck_'+ontem+'.csv'
    arquivo_prev_temp_anterior_ne = caminho_arquivos + 'NE/ne_temp_prev_deck_'+ontem+'.csv'
    arquivo_prev_carga_anterior_ne = caminho_arquivos + 'NE/ne_carga_prev_deck_'+ontem+'.csv'
    feriados_ontem_ne = caminho_arquivos +'NE/NE_'+ontem+'_FERIADOS.csv'
    
    arquivo_carga_anterior_n = caminho_arquivos + 'N/n_carga_deck_'+ontem+'.csv'
    arquivo_temp_anterior_n = caminho_arquivos + 'N/n_temp_deck_'+ontem+'.csv'
    arquivo_prev_temp_anterior_n = caminho_arquivos + 'N/n_temp_prev_deck_'+ontem+'.csv'
    arquivo_prev_carga_anterior_n = caminho_arquivos + 'N/n_carga_prev_deck_'+ontem+'.csv'
    feriados_ontem_n = caminho_arquivos +'N/N_'+ontem+'_FERIADOS.csv'

    
    try:
        os.remove(caminho_base)
        os.remove(caminho_previsao)
        os.remove(caminho_seco)
        os.remove(caminho_s)
        os.remove(caminho_ne)
        os.remove(caminho_n)
        
        os.remove(carga_seco)
        os.remove(carga_s)
        os.remove(carga_ne)
        os.remove(carga_n)
        os.remove(carga_prev_seco)
        os.remove(carga_prev_s)
        os.remove(carga_prev_ne)
        os.remove(carga_prev_n)
        os.remove(temp_seco)
        os.remove(temp_s)
        os.remove(temp_ne)
        os.remove(temp_n)
        os.remove(temp_prev_seco)
        os.remove(temp_prev_s)
        os.remove(temp_prev_ne)
        os.remove(temp_prev_n)
        os.remove(feriados_seco)
        os.remove(feriados_s)
        os.remove(feriados_ne)
        os.remove(feriados_n)
        
        os.remove(arquivo_carga_anterior_seco)
        os.remove(arquivo_carga_anterior_s)
        os.remove(arquivo_carga_anterior_ne)
        os.remove(arquivo_carga_anterior_n)
        os.remove(arquivo_temp_anterior_seco)
        os.remove(arquivo_temp_anterior_s)
        os.remove(arquivo_temp_anterior_ne)
        os.remove(arquivo_temp_anterior_n)
        os.remove(arquivo_prev_temp_anterior_seco)
        os.remove(arquivo_prev_temp_anterior_s)
        os.remove(arquivo_prev_temp_anterior_ne)
        os.remove(arquivo_prev_temp_anterior_n)
        os.remove(arquivo_prev_carga_anterior_seco)
        os.remove(arquivo_prev_carga_anterior_s)
        os.remove(arquivo_prev_carga_anterior_ne)
        os.remove(arquivo_prev_carga_anterior_n)
        os.remove(feriados_ontem_seco)
        os.remove(feriados_ontem_s)
        os.remove(feriados_ontem_ne)
        os.remove(feriados_ontem_n)
        

    except OSError as e:
        print(e)
          
    print("Arquivos deletados com sucesso!")  


