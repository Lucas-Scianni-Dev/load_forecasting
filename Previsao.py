import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from datetime import date,datetime,timedelta,timezone
import timeit
import os
import sys
from zipfile import ZipFile
import shutil
import time
import functools
import plotly.graph_objects as go
import plotly.express as px
print = functools.partial(print, flush=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from Extracao_deck import Monta_deck,Apagar,Extrair

from Tratamento_dos_dados import Tratamento_dos_dados
from Previsao_funcoes import montar_conjuntos_previsao,avalia_prev_passada,prever,carga_decomp

parametros = sys.argv[1:]

ano_previsao,mes_previsao,dia_previsao = int(parametros[0][0:4]),int(parametros[0][5:7]),int(parametros[0][8:])
horizonte_previsao = int(parametros[1])
feriados_futuros = [0,0,0,0,0,0,0,0,0,0,0] #se for feriado, colocar 1 (começando por hoje)

caminho_pasta = parametros[2]       
caminho_arquivos = caminho_pasta +'DeckCorrigido/'
caminho_arquivos_antigos = caminho_pasta +'DeckCorrigido_storage'
caminho_previsoes = caminho_pasta + 'Previsoes/'
caminho_verifica_data = caminho_pasta +'DeckCorrigido/SECO'


verifica_pasta_deck =  os.path.isdir(caminho_arquivos)
if verifica_pasta_deck == False:
    shutil.copytree(caminho_arquivos_antigos,caminho_arquivos)

data_previsao = date(ano_previsao,mes_previsao,dia_previsao)
hoje_date_format = data_previsao
arq_prev_passada_1 = caminho_pasta + "Previsoes/"+str((hoje_date_format - timedelta(1)).year)+'/'+ '%02d' % (hoje_date_format - timedelta(1)).month +'/'+str((hoje_date_format - timedelta(1)).isoformat())+'/daily/'+hoje_date_format.isoformat()+'.csv'
arq_prev_passada_2 = caminho_pasta + "Previsoes/"+str((hoje_date_format - timedelta(2)).year)+'/'+'%02d' % (hoje_date_format - timedelta(2)).month +'/'+str((hoje_date_format - timedelta(2)).isoformat())+'/daily/'+hoje_date_format.isoformat()+'.csv'

verifica_prev_necessaria_1 =  os.path.isfile(arq_prev_passada_1)
verifica_prev_necessaria_2 =  os.path.isfile(arq_prev_passada_2)

####################################################################################################
if horizonte_previsao > 1:
    if verifica_prev_necessaria_1 == False:
        try:
            arqs = os.listdir(caminho_verifica_data)
            arq = arqs[0]
            for i in range(len(arq)):
                if arq[i] == '2':
                    n_i = i
                    break
            data = arq[n_i:n_i + 10]
            ano_arquivo_existente, mes_arquivo_existente, dia_arquivo_existente = int(data[0:4]),int(data[5:7]), int(data[8:10])

            hoje_date_format = date.today()
            data_previsao = date(ano_previsao,mes_previsao,dia_previsao) - timedelta(1)
            atraso = (hoje_date_format - data_previsao).days

            ultima_atualizacao = date(ano_arquivo_existente,mes_arquivo_existente,dia_arquivo_existente)

            defasagem = int((data_previsao - ultima_atualizacao).days) -1

            if defasagem < -1:
                shutil.rmtree(caminho_arquivos)
                shutil.copytree(caminho_arquivos_antigos,caminho_arquivos)

            arqs = os.listdir(caminho_verifica_data)
            arq = arqs[0]
            for i in range(len(arq)):
                if arq[i] == '2':
                    n_i = i
                    break
            data = arq[n_i:n_i + 10]
            ano_arquivo_existente, mes_arquivo_existente, dia_arquivo_existente = int(data[0:4]),int(data[5:7]), int(data[8:10])

            ultima_atualizacao = date(ano_arquivo_existente,mes_arquivo_existente,dia_arquivo_existente)
            diferenca = int((data_previsao - ultima_atualizacao).days) - 1
            hoje_date_format = data_previsao
            ontem = (date.today() -timedelta(days=atraso + diferenca+1)).isoformat()
            antes_ontem = (date.today() -timedelta(days=atraso + diferenca+2)).isoformat()

            if ultima_atualizacao != hoje_date_format:
                Extrair(hoje_date_format,caminho_arquivos,caminho_pasta)
                Monta_deck(hoje_date_format,diferenca,ontem,ultima_atualizacao,caminho_arquivos,caminho_pasta)
                Apagar(hoje_date_format,ontem,antes_ontem,caminho_arquivos,caminho_pasta)
            
            if os.path.isdir(caminho_previsoes):
                True
            else:
                os.mkdir(caminho_previsoes)

            if os.path.isdir(caminho_previsoes + str(hoje_date_format.year)):
                True
            else:
                os.mkdir(caminho_previsoes + str(hoje_date_format.year))

            if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month):
                True
            else:
                os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month)

            if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())):
                True
            else:
                os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat()))

            if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat()) + '/daily'):
                True
            else:
                os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+ '/daily')

            with open('Carga.txt') as f:
                lines = f.readlines()
            cabecalho = lines[0:4]
            lines = lines[4:]
            
            previsao_passada = avalia_prev_passada(atraso,horizonte_previsao,caminho_pasta)
            previsao_dia_passado = avalia_prev_passada(atraso,1,caminho_pasta)
            
            previsoes = pd.DataFrame()
            
            dias_a_frente =  1
            conjuntos_mlp,conjuntos_lstm,var_norm = montar_conjuntos_previsao(dias_a_frente,atraso,feriados_futuros,caminho_pasta)
        
            previsao = prever(atraso,dias_a_frente,conjuntos_mlp,conjuntos_lstm,var_norm,caminho_pasta)

            seco = (previsao['LSTM_Seco'].values).tolist()
            s = (previsao['LSTM_S'].values).tolist()
            ne = (previsao['LSTM_NE'].values).tolist()
            n = (previsao['LSTM_N'].values).tolist()

            for m in range(len(s)):
                seco.append(s[m])
            for p in range(len(ne)):
                seco.append(ne[p])
            for q in range(len(n)):
                seco.append(n[q])

            data  = (data_previsao+timedelta(1)).isoformat()
            dia = data[8:]
            k = 0
            with open('Previsoes/'+ str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())+'/daily/'+data+'.txt','w') as w:
                for j in range(4):
                    w.write(cabecalho[j])
                for linha in lines: 
                    x = linha[0:8] + dia + linha[10:26] + "%.2f" %round(seco[k],2) + linha[34:]
                    w.write(x)
                    k = k+1
        except:
            print("Previsão do dia anterior nao foi feita e os arquivos nao disponiveis no SINtegre")
            if verifica_prev_necessaria_2 == False:
                print("Previsão do dia atual nao encontrada nos ultimos dois dias! \nRode a previsao de 2 dias atras..")
                print("A carga será prevista apenas para o proximo dia.")
                horizonte_previsao = 1
                #sys.exit()
            else:
                print("Utilizando previsão de dois dias atrás")

    else:
        print("previsao de hoje foi realizada ontem!")

#################################################################################################

arqs = os.listdir(caminho_verifica_data)
arq = arqs[0]
for i in range(len(arq)):
    if arq[i] == '2':
        n_i = i
        break
data = arq[n_i:n_i + 10]
ano_arquivo_existente, mes_arquivo_existente, dia_arquivo_existente = int(data[0:4]),int(data[5:7]), int(data[8:10])

hoje_date_format = date.today()
data_previsao = date(ano_previsao,mes_previsao,dia_previsao)
atraso = (hoje_date_format - data_previsao).days

ultima_atualizacao = date(ano_arquivo_existente,mes_arquivo_existente,dia_arquivo_existente)

defasagem = int((data_previsao - ultima_atualizacao).days) -1

if defasagem < -1:
    shutil.rmtree(caminho_arquivos)
    shutil.copytree(caminho_arquivos_antigos,caminho_arquivos)

arqs = os.listdir(caminho_verifica_data)
arq = arqs[0]
for i in range(len(arq)):
    if arq[i] == '2':
        n_i = i
        break
data = arq[n_i:n_i + 10]
ano_arquivo_existente, mes_arquivo_existente, dia_arquivo_existente = int(data[0:4]),int(data[5:7]), int(data[8:10])

ultima_atualizacao = date(ano_arquivo_existente,mes_arquivo_existente,dia_arquivo_existente)
diferenca = int((data_previsao - ultima_atualizacao).days) - 1
hoje_date_format = data_previsao
ontem = (date.today() -timedelta(days=atraso + diferenca+1)).isoformat()
antes_ontem = (date.today() -timedelta(days=atraso + diferenca+2)).isoformat()

if ultima_atualizacao != hoje_date_format:
    Extrair(hoje_date_format,caminho_arquivos,caminho_pasta)
    Monta_deck(hoje_date_format,diferenca,ontem,ultima_atualizacao,caminho_arquivos,caminho_pasta)
    Apagar(hoje_date_format,ontem,antes_ontem,caminho_arquivos,caminho_pasta)

if os.path.isdir(caminho_previsoes):
    True
else:
    os.mkdir(caminho_previsoes)

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year)):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year))

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month)

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat()))

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat()) + '/daily'):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+ '/daily')

with open('Carga.txt') as f:
    lines = f.readlines()
cabecalho = lines[0:4]
lines = lines[4:]

previsao_passada = avalia_prev_passada(atraso,horizonte_previsao,caminho_pasta)
previsao_dia_passado = avalia_prev_passada(atraso,1,caminho_pasta)

data_previsao_passada = ((date.today() -timedelta(days=atraso))-timedelta(days=1))
data_pasta_passada = ((date.today() -timedelta(days=atraso))-timedelta(days=2))
arquivo_prev_verificada = caminho_previsoes + str(data_pasta_passada.year) + '/' + '%02d' % data_pasta_passada.month+ '/' + str(data_pasta_passada.isoformat())+'/daily/'+(data_previsao_passada).isoformat()+'.csv'

if  os.path.isfile(arquivo_prev_verificada) == True:

    arq_previsao_passada = pd.read_csv(caminho_previsoes + str(data_pasta_passada.year) + '/' + '%02d' % data_pasta_passada.month+ '/' + str(data_pasta_passada.isoformat())+'/daily/'+(data_previsao_passada).isoformat()+'.csv',decimal = ',',delimiter=';')

    with open('Carga.txt') as f:
        lines = f.readlines()
    cabecalho = lines[0:4]
    lines = lines[4:]

    seco = (arq_previsao_passada['SECO Carga Verificada'].values).tolist()
    s = (arq_previsao_passada['S Carga Verificada'].values).tolist()
    ne = (arq_previsao_passada['NE Carga Verificada'].values).tolist()
    n = (arq_previsao_passada['N Carga Verificada'].values).tolist()

    for i in range(len(s)):
        seco.append(s[i])
    for i in range(len(ne)):
        seco.append(ne[i])
    for i in range(len(n)):
        seco.append(n[i])

    data = data_previsao_passada.isoformat()
    dia = data[8:]
    k = 0
    with open(caminho_previsoes + str(data_pasta_passada.year) + '/' + '%02d' % data_pasta_passada.month+ '/' + str(data_pasta_passada.isoformat())+'/daily/Verificado_'+(data_previsao_passada).isoformat()+'.txt','w') as w:
        for j in range(4):
            w.write(cabecalho[j])
        for linha in lines: 
            x = linha[0:8] + dia + linha[10:26] + "%.2f" %round(seco[k],2) + linha[34:]
            w.write(x)
            k = k+1

previsoes = pd.DataFrame()
for i in range(horizonte_previsao):
    dias_a_frente = i + 1
    data  = (data_previsao+timedelta(days=1+i)).isoformat()
    conjuntos_mlp,conjuntos_lstm,var_norm = montar_conjuntos_previsao(dias_a_frente,atraso,feriados_futuros,caminho_pasta)
    if os.path.isfile(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+ '/daily/'+data+'.csv') == True:
        print('\n')
        previsao = pd.read_csv(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+ '/daily/'+data+'.csv',sep = ';',decimal=',')
    else:
        previsao = prever(atraso,dias_a_frente,conjuntos_mlp,conjuntos_lstm,var_norm,caminho_pasta)

    seco = (previsao['LSTM_Seco'].values).tolist()
    s = (previsao['LSTM_S'].values).tolist()
    ne = (previsao['LSTM_NE'].values).tolist()
    n = (previsao['LSTM_N'].values).tolist()

    for m in range(len(s)):
        seco.append(s[m])
    for p in range(len(ne)):
        seco.append(ne[p])
    for q in range(len(n)):
        seco.append(n[q])

    data  = (data_previsao+timedelta(days=1+i)).isoformat()
    dia = data[8:]
    k = 0
    with open('Previsoes/'+ str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())+'/daily/'+data+'.txt','w') as w:
        for j in range(4):
            w.write(cabecalho[j])
        for linha in lines: 
            x = linha[0:8] + dia + linha[10:26] + "%.2f" %round(seco[k],2) + linha[34:]
            w.write(x)
            k = k+1

    previsoes = previsoes.append(previsao)
    print(str(dias_a_frente) + " dias a frente previsto\n")



if os.path.isdir(caminho_previsoes):
    True
else:
    os.mkdir(caminho_previsoes)

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year)):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year))

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month)

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat()))

if os.path.isdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat()) + '/weekly'):
    True
else:
    os.mkdir(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+ '/weekly')

previsoes.reset_index(drop=True)
previsoes = round(previsoes,2)
previsoes.to_csv(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' %hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+'/weekly/'+str(data_previsao + timedelta(1))+".csv",sep = ';',decimal=',')

if horizonte_previsao == 7:
    patamares_decomp,patamares_decomp_ordenada = carga_decomp(previsoes,atraso,horizonte_previsao,caminho_pasta)
    patamares_decomp.to_csv(caminho_pasta + "Previsoes/"+str(hoje_date_format.year)+'/'+'%02d' % hoje_date_format.month+'/'+str(hoje_date_format.isoformat())+'/weekly/Patamares_'+str(data_previsao+ timedelta(1))+".csv",sep = ';',decimal=',')
    patamares_decomp_ordenada.to_csv(caminho_pasta + "Previsoes/"+str(hoje_date_format.year)+'/'+'%02d' % hoje_date_format.month+'/'+str(hoje_date_format.isoformat())+'/weekly/Patamares_ordenados_'+str(data_previsao+ timedelta(1))+".csv",sep = ';',decimal=',')

if horizonte_previsao == 7:
    with open(caminho_previsoes + str(hoje_date_format.year) + '/' + '%02d' %hoje_date_format.month+ '/' + str(hoje_date_format.isoformat())+'/weekly/'+str(data_previsao + timedelta(1))+".txt",'w') as w:
        for i in range(7):
            data  = (data_previsao+timedelta(days=1+i)).isoformat()
            with open('Previsoes/'+ str(hoje_date_format.year) + '/' + '%02d' % hoje_date_format.month + '/' + str(hoje_date_format.isoformat())+'/daily/'+data+'.txt') as f:
                lines = f.readlines()
                cabecalho = lines[0:4]
                lines = lines[4:]
                
                if i == 0:
                    for j in range(4):
                        w.write(cabecalho[j])
                
                for linha in lines:
                    w.write(linha)
