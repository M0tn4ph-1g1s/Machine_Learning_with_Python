#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

brazil = pd.read_csv('BRA.csv', delimiter=',')
argentina = pd.read_csv('ARG.csv', delimiter=',')

"""
polonia = pd.read_csv('POL.csv' ,delimiter=',')
normandia = pd.read_csv('NOR.csv' ,delimiter=',')
mexico =  pd.read_csv('MEX.csv' ,delimiter=',')"""


#função que vai retornar os melhores dados para o aprendizado de maquina usadno o SelectKBest
def Features_Select(features,labels):
    features_List = ('HG','AG','PH','PD','PA','MaxH','MaxD','MaxA','AvgH','AvgD','AvgA')
    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features,labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_List[1:0], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs,key=lambda x: x[1])))

    k_best_features_final = dict(ordered_pairs[:15])
    

    return "0"

def campeonatos_pelo_globo(brazilDados, argentinaDados):
    print("processando os dados obtidos")
    jogosBrasil = brazilDados.shape[0]
    jogosArgentina = argentinaDados.shape[0]

    # H --> vitoria time da casa
    # D --> empate
    # A --> time visitante
    vitorias_casa_br = len(brazilDados[brazilDados.Res == "H"])
    vitorias_visitante_br = len(brazilDados[brazilDados.Res == "A"])
    empate_brasil = len(brazilDados[brazilDados.Res == "D"])
    win_rate_br = (float(vitorias_casa_br / jogosBrasil)) * 100
    val_br = [vitorias_casa_br, vitorias_visitante_br, empate_brasil]

    vitorias_casa_AR = len(argentinaDados[argentinaDados.Res == "H"])
    vitorias_visitante_AR = len(argentinaDados[argentinaDados.Res == "A"])
    empate_argentina = len(argentinaDados[argentinaDados.Res == "D"])
    win_rate_AR = (float(vitorias_casa_AR / jogosArgentina)) * 100
    val_AR = [vitorias_casa_AR, vitorias_visitante_AR, empate_argentina]

    print("Total de jogos no brasil:.", jogosBrasil)
    print("Total de jogos na argentina:.", jogosArgentina)
    print(".................................................")
    print("Total de jogos ganhos em casa;brasil:.", vitorias_visitante_br)
    print("Total de jogos ganhos em casa;argentina:.", vitorias_casa_AR)
    print(".................................................")
    print("Total de jogos ganhos pelo visitante;brasil ", vitorias_visitante_br)
    print("Total de jogos ganhos pelo visitante;argentina", vitorias_visitante_AR)
    print(".................................................")
    print("Total de jogos empatados;brasil", empate_brasil)
    print("Total de jogos empatados;argentina", empate_argentina)
    print(".................................................")
    print("Percentual de jogos ganhos em casa;brasil", win_rate_br)
    print("Percentual de jogos ganhos em casa;argentina", win_rate_AR)

    # exibindo em graficos os dados dos campeonatos brasileiros
    """x = np.arange(3)
    plt.bar(x, val_br)
    plt.xticks(x, ('Home', 'Away', 'Draw'))
    plt.show()

    # exibindo em graficos os dados dos campeonatos argentinos
    y = np.arange(3)
    plt.bar(y, val_AR)
    plt.xticks(y, ('Home', 'Away', 'Draw'))
    plt.show()"""

    # preparando os dados
    # deixar somente as variaveis de interesse
    dados_brasil_formatados = brazilDados.drop(['Country', 'League', 'Season', 'Date', 'Time', 'Home', 'Away'], 1)
    dados_argentina_formatados = argentinaDados.drop(['Country', 'League', 'Season', 'Date', 'Time', 'Home', 'Away'], 1)

    display(dados_brasil_formatados.head())
    display(dados_argentina_formatados.head())

    #separar os features das labels
    features_argentina = dados_argentina_formatados.drop(['Res'],1)
    features_brazil = dados_brasil_formatados.drop(['Res'],1)

    #separar as labels
    labels_argentina = dados_argentina_formatados['Res']
    labels_brazil = dados_brasil_formatados['Res']





if __name__ == '__main__':
    campeonatos_pelo_globo(brazil, argentina)
