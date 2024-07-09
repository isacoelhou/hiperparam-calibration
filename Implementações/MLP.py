import sklearn as sk
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from skopt import dummy_minimize
from skopt import gp_minimize

import time

def mlp_random_search():
  maior = -1

  for n in range(50):
    i = random.choice((5,6,10,12))
    j = random.choice(('constant','invscaling', 'adaptive'))
    k = random.choice((50,100,150,300,500,1000))
    l = random.choice(('identity', 'logistic', 'tanh', 'relu'))

    MLP = MLPClassifier(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l, verbose=False)
    MLP.fit(x_treino,y_treino)

    opiniao = MLP.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)

    if (Acc > maior):
      maior = Acc
      Melhor_i = i
      Melhor_j = j
      Melhor_k = k
      Melhor_l = l

  #print("Avaliação da grid MLP search\n")

  #print("Acc do  MLP:",maior)
  #print("hidden_layer_sizes =", Melhor_i)
  #print("learning_rate =", Melhor_j)
  #print("max_iter =", Melhor_k)
  #print("activation =", Melhor_l)

  #print("\nDesempenho sobre o teste")
  MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i,Melhor_i,Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l, verbose=False)
  MLP.fit(x_treino, y_treino)
  opiniao =  MLP.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc 
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")

def mlp_grid_search():

  maior = -1

  for i in (5,6,10,12):
    for j in ('constant','invscaling', 'adaptive'):
        for k in (50,100,150,300,500,1000):
          for l in ('identity', 'logistic', 'tanh', 'relu'):
              MLP = MLPClassifier(hidden_layer_sizes=(i,i,i), learning_rate=j, max_iter=k, activation=l )
              MLP.fit(x_treino,y_treino)

              opiniao = MLP.predict(x_validacao)
              Acc = accuracy_score(y_validacao, opiniao)

              if (Acc > maior):
                maior = Acc
                Melhor_i = i
                Melhor_j = j
                Melhor_k = k
                Melhor_l = l

  #print("Avaliação da random MLP search\n")

  #print("Acc do  MLP:",maior)
  #print("C =", Melhor_i)
  #print("Kernel =", Melhor_j)

  #print("\nDesempenho sobre o teste")
  MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i,Melhor_i,Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l, verbose=False)
  MLP.fit(x_treino, y_treino)
  opiniao =  MLP.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc 
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")



def mlp_cross_validation():
  parametros = {'hidden_layer_sizes' : [(5,5,5), (6,6,6), (10,10,10), (12,12,12)], 'learning_rate' : ('constant','invscaling', 'adaptive'), 'max_iter' : [50,100,150,300,500,1000], 'activation': ('identity', 'logistic', 'tanh', 'relu')}
  MLP = MLPClassifier()

  Classificador = GridSearchCV(estimator=MLP,param_grid=parametros,scoring='accuracy',cv=5)
  Classificador.fit(Vetor_X,Vetor_Y)
  pd.DataFrame(Classificador.cv_results_)

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)
  MLP =  MLPClassifier(
        hidden_layer_sizes=Classificador.best_params['hidden_layer_sizes'],
        learning_rate=Classificador.best_params['learning_rate'],
        max_iter=Classificador.best_params['max_iter'],
        activation=Classificador.best_params['activation']
    )

  MLP.fit(x_treino,y_treino)
  opiniao = MLP.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)

  return Acc  

def mlp_sucessive_halving():
    parametros = {
        'hidden_layer_sizes': [(5, 5, 5), (6, 6, 6), (10, 10, 10), (12, 12, 12)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [50, 100, 150, 300, 500, 1000],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }

    MLP = MLPClassifier()
    Classificador = HalvingGridSearchCV(MLP, parametros, cv=5)


    Classificador.fit(Vetor_X, Vetor_Y)
    results_df = pd.DataFrame(Classificador.cv_results_)

    #print("Melhores parâmetros: ", Classificador.best_params_)
    #print("Melhor desempenho: ", Classificador.best_score_)

    best_params = Classificador.best_params_

    #print("\nDesempenho sobre o teste")

    MLP_best = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter'],
        activation=best_params['activation']
    )

    MLP_best.fit(Vetor_X, Vetor_Y)

    opiniao = MLP_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    return Acc

    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")

def treinar_modelo_mlp(params):
    hidden_layers = params[0]
    learning_rate = params[1]
    max_iter = params[2]
    activation = params[3]

    MLP = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate=learning_rate, max_iter=max_iter, activation=activation)
    MLP.fit(x_treino, y_treino)
    opiniao = MLP.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def mlp_dummy_optimization():
    parametros = [([(5, 5, 5), (6, 6, 6), (10, 10, 10), (12, 12, 12)]), ('constant', 'invscaling', 'adaptive'), (50, 100, 150, 300, 500, 1000), ('identity', 'logistic', 'tanh', 'relu')]

    #print("Avaliação Teste usando Dummy")
    Resultado_rs = dummy_minimize(treinar_modelo_mlp, parametros, verbose=0, n_calls=30)

    #print("\nMelhores parâmetros")
    #print("hidden_layer_sizes: ", Resultado_rs.x[0], "learning_rate: ", Resultado_rs.x[1], "max_iter: ", Resultado_rs.x[2], "activation: ", Resultado_rs.x[3])

    #print("\nDesempenho sobre o teste")
    MLP = MLPClassifier(hidden_layer_sizes=Resultado_rs.x[0], learning_rate=Resultado_rs.x[1], max_iter=Resultado_rs.x[2], activation=Resultado_rs.x[3])
    MLP.fit(Vetor_X, Vetor_Y)
    opiniao = MLP.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    return Acc
    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")

def mlp_bayesian_optimization():
    #parametros = [([(5, 5, 5), (6, 6, 6), (10, 10, 10), (12, 12, 12)]), ('constant', 'invscaling', 'adaptive'), (50, 100, 150, 300, 500, 1000), ('identity', 'logistic', 'tanh', 'relu')]
    parametros = [(5,10,15), ('constant', 'invscaling', 'adaptive'), (50, 100, 150, 300, 500, 1000), ('identity', 'logistic', 'tanh', 'relu')]

    #print("Avaliação da Otimização Bayesiana")
    Resultado_go = gp_minimize(treinar_modelo_mlp, parametros, verbose=0, n_calls=30, n_random_starts=10)

    #print("\nMelhores parâmetros")
    #print("hidden_layer_sizes: ", Resultado_go.x[0], "learning_rate: ", Resultado_go.x[1], "max_iter: ", Resultado_go.x[2], "activation: ", Resultado_go.x[3])

    #print("\nDesempenho sobre o teste")
    MLP = MLPClassifier(hidden_layer_sizes=Resultado_go.x[0], learning_rate=Resultado_go.x[1], max_iter=Resultado_go.x[2], activation=Resultado_go.x[3])
    MLP.fit(Vetor_X, Vetor_Y)
    opiniao = MLP.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    #print("Acurácia: ", Acc)

    return Acc


def media_valores(lista):

    menor_numero = min(lista)
    maior_numero = max(lista)

    lista.remove(maior_numero)
    lista.remove(menor_numero)

    return (sum(lista)/len(lista))


dados = pd.read_csv("../datasets/Diabetes.csv")
dados.head()

df_dados = pd.DataFrame(dados)

df_dados = df_dados.iloc[:, :-1]
df_dados.info()

Vetor_X = df_dados
Vetor_Y = dados["Class"]

gs_tempo = []
rs_tempo = []
cv_tempo = []
sh_tempo = []
bo_tempo = []
do_tempo = []

gs_acc = []
rs_acc = []
cv_acc = []
sh_acc = []
bo_acc = []
do_acc = []

for i in range(10):

    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste= train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    inicio = time.time()
    gs_acc.append(mlp_grid_search())
    fim = time.time()
    tempo_total = fim - inicio
    gs_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp grid search: {tempo_total} segundos")

    inicio = time.time()
    rs_acc.append(mlp_random_search())
    fim = time.time()
    tempo_total = fim - inicio
    rs_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp random search: {tempo_total} segundos")

    inicio = time.time()
    do_acc.append(mlp_dummy_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    do_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp dummy opt: {tempo_total} segundos")

    inicio = time.time()
    bo_acc.append(mlp_bayesian_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    bo_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp bayesian opt: {tempo_total} segundos")

    inicio = time.time()
    cv_acc.append(mlp_cross_validation())
    fim = time.time()
    tempo_total = fim - inicio
    cv_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp cross validation: {tempo_total} segundos")

    inicio = time.time()
    sh_acc.append(mlp_sucessive_halving())
    fim = time.time()
    tempo_total = fim - inicio
    sh_tempo.append(tempo_total)
    #print(f"Tempo de execução no mlp sucessive halving: {tempo_total} segundos")

with open('./stats/MLPstats.txt', 'w') as arquivo:

  arquivo.write("\nTempo:\n")

  arquivo.write("Média de tempo no grid search: ",media_valores(gs_tempo))
  arquivo.write("Média de tempo no random search: ", media_valores(rs_tempo))
  arquivo.write("Média de tempo no dummy opt " ,media_valores(do_tempo))
  arquivo.write("Média de tempo no bayesian opt: " , media_valores(bo_tempo))
  arquivo.write("Média de tempo no sucessive halving: " , media_valores(sh_tempo))
  arquivo.write("Média de tempo no cross validation: " , media_valores(cv_tempo))

  arquivo.write("\nACC:\n")

  arquivo.write("Média de acc no grid search: ",media_valores(gs_acc))
  arquivo.write("Média de acc no random search: ", media_valores(rs_acc))
  arquivo.write("Média de acc no dummy opt " ,media_valores(do_acc))
  arquivo.write("Média de acc no bayesian opt: " , media_valores(bo_acc))
  arquivo.write("Média de acc no sucessive halving: " , media_valores(sh_acc))
  arquivo.write("Média de acc no cross validation: " , media_valores(cv_acc))