## Scripts: 
    • Extracao_deck: partindo do pressuposto que o arquivo .zip de carga disponível em https://sintegre.ons.org.br/sites/9/46/Produtos/479/PrevCargaDESSEM_+data+.zip já foi baixado, este script irá fazer a extração do conteúdo, atualizar os arquivos de input para a previsão de carga e apagar os arquivos baixados. (os arquivos desnecessários sempre serão apagados).

    • Tratamento_dos_dados: Irá tratar os dados, fazer a codificação das variáveis de calendário e normalizar. Retorna os dados codificados e normalizados, as datas e as variáveis de normalização.

    • Previsao_funcoes: contém as funções necessárias para realizar a previsão:

        i. montar_conjuntos_previsao;
        ii. prever
        iii. avalia_prev_passada 

    • Previsao: script para rodar a previsão de fato.

## Funcionamento: 
	Para que a previsão possa ser realizada é necessário que a pasta com o DeckCorrigido_storage exista, além do arquivo Carga.txt, e a pasta Modelos_Treinados com as redes neurais treinadas. (além dos arquivos zip obtidos no SINtegre).
	Ao executar o comando python Previsao.py no prompt de comando alguns argumentos devem ser passados: data prevista (primeiro dia do horizonte, sendo ano, mês e dia); horizonte de previsão (de 1 a 7 dias a frente); caminho da pasta onde se encontram os códigos e arquivos (colocando barra no final do nome). 

## Exemplo:

> python Previsao.py 2021 11 23 1 /home/script/path/

Durante a execução, a cada dia previsto do horizonte, uma mensagem irá aparecer na tela dizendo que a carga do dia X foi prevista e o tempo de execução.
Ao final do processamento, terá sido criadas as pastas Previsoes/ Horizonte dias/Ano/Mês/Data do dia em que se rodou a previsão

A previsão deve ser feita diariamente para um horizonte de 7 dias. Serão então criados arquivos separados com a previsão de cada dia (que estarão na pasta 1 dias) e um arquivo com a previsão agrupada (na pasta 7 dias). Para as previsões separadas, além do arquivo .csv, será criado um arquivo .txt com a formatação do deck do dessem.


# BIBLIOTECAS E COMANDOS PARA INSTALAÇÃO 	
conda install pandas
conda install numpy
pip install matplotlib
pip install DateTime
pip install tensorflow – gpu
pip install nbimporter
pip install pytest-timeit
pip install zipfile36
pip install pytest-shutil