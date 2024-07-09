import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from skopt import dummy_minimize
from skopt import gp_minimize

import time

def rf_grid_search():
  maior = -1

  for i in (10,20,30,50,75,100):
      for j in ("gini", "entropy"):
          for k in (3,4,5,6,7):
            for l in (5,6,8,10):
              for m in (3,4,5,6):
                RF = RandomForestClassifier(n_estimators= i, criterion=j,max_depth=k, min_samples_split = l,min_samples_leaf=m)
                RF.fit(x_treino,y_treino)

                opiniao = RF.predict(x_validacao)
                Acc = accuracy_score(y_validacao, opiniao)

                if (Acc > maior):
                  maior = Acc
                  Melhor_i = i
                  Melhor_j = j
                  Melhor_k = k
                  Melhor_l = l
                  Melhor_m = m

  #print("Avaliação da grid random forest search\n")

  #print("ACC da Random Forest: ", maior)
  #print("Número de estimadores:", Melhor_i,"\nCritério:", Melhor_j,"\nProfundidade:", Melhor_k, "\nMínimo para divisão:", Melhor_l, "\nMínimo por folha:", Melhor_m)

  #print("\nDesempenho sobre o teste")
  RF = RandomForestClassifier(n_estimators= Melhor_i, criterion=Melhor_j,max_depth=Melhor_k, min_samples_split = Melhor_l,min_samples_leaf=Melhor_m)
  RF.fit(x_treino,y_treino)
  opiniao = RF.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")


def rf_random_search():

  for _ in range(15):

    maior = -1

    i = random.choice((10,20,30,50,75,100))
    j = random.choice(("gini", "entropy"))
    k = random.choice((3,4,5,6,7))
    l = random.choice((5,6,8,10))
    m = random.choice((3,4,5,6))

    RF = RandomForestClassifier(n_estimators= i, criterion=j,max_depth=k, min_samples_split = l,min_samples_leaf=m)
    RF.fit(x_treino,y_treino)

    opiniao = RF.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)

    if (Acc > maior):
       maior = Acc
       Melhor_i = i
       Melhor_j = j
       Melhor_k = k
       Melhor_l = l
       Melhor_m = m

  #print("Avaliação da random random forest search\n")

  #print("ACC da Random Forest: ", maior)
  #print("Número de estimadores:", Melhor_i,"\nCritério:", Melhor_j,"\nProfundidade:", Melhor_k, "\nMínimo para divisão:", Melhor_l, "\nMínimo por folha:", Melhor_m)

  #print("\nDesempenho sobre o teste")
  RF = RandomForestClassifier(n_estimators= Melhor_i, criterion=Melhor_j,max_depth=Melhor_k, min_samples_split = Melhor_l,min_samples_leaf=Melhor_m)
  RF.fit(x_treino,y_treino)
  opiniao = RF.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")

def rf_cross_validation():
  parametros = {'criterion': ['gini', 'entropy'], 'n_estimators': [10,20,30,50,75,100], 'max_depth': [3,4,5,6,7], 'min_samples_split':[5,6,8,10], 'min_samples_leaf': [3,4,5,6]}

  RF = RandomForestClassifier()
  Classificador = GridSearchCV(estimator=RF, param_grid=parametros,scoring='accuracy',cv=5)
  Classificador.fit(Vetor_X,Vetor_Y)

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")
  RF = RandomForestClassifier(n_estimators= Classificador.best_params_['n_estimators'], criterion=Classificador.best_params_['criterion'],max_depth=Classificador.best_params_['max_depth'], min_samples_split =Classificador.best_params_['min_samples_split'],min_samples_leaf=Classificador.best_params_['min_samples_leaf'])
  RF.fit(x_treino,y_treino)
  opiniao = RF.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")

def rf_sucessive_halving():
  parametros = {'criterion': ['gini', 'entropy'], 'n_estimators': [10,20,30,50,75,100], 'max_depth': [3,4,5,6,7], 'min_samples_split':[5,6,8,10], 'min_samples_leaf': [3,4,5,6]}

  RF = RandomForestClassifier()
  Classificador =  HalvingGridSearchCV(RF, parametros,cv=5)
  Classificador.fit(Vetor_X,Vetor_Y)

  RF = RandomForestClassifier(criterion=Classificador.best_params_['criterion'], 
                              n_estimators = Classificador.best_params_['n_estimators'], 
                              max_depth = Classificador.best_params_['max_depth'],  
                              min_samples_split = Classificador.best_params_['min_samples_split'], 
                              min_samples_leaf = Classificador.best_params_['min_samples_leaf'])
  
  RF.fit(x_treino, y_treino)
  opiniao = RF.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc

def treinar_modelo_rf(params):
    j = params[0]
    i = params[1]
    k = params[2]
    l = params[3]
    m = params[4]
    RF = RandomForestClassifier(n_estimators=i, criterion=j, max_depth=k, min_samples_split=l, min_samples_leaf=m)
    RF.fit(Vetor_X, Vetor_Y)
    opiniao = RF.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def rf_dummy_optimization():

  parametros = [('gini', 'entropy'), (10, 20, 30, 50, 75, 100), (3, 4, 5, 6, 7), (5, 6, 8, 10), (3, 4, 5, 6)]

  #print("\nAvaliação Teste usando Dummy")
  Resultado_rs = dummy_minimize(treinar_modelo_rf, parametros, verbose=0, n_calls=30)

  #print("\nMelhores parâmetros")
  #print("criterion: ", Resultado_rs.x[0], "n_estimators: ", Resultado_rs.x[1], "max_depth: ", Resultado_rs.x[2], "min_samples_split: ", Resultado_rs.x[3], "min_samples_leaf: ", Resultado_rs.x[4])

  #print("\nDesempenho sobre o teste")
  RF = RandomForestClassifier(n_estimators=Resultado_rs.x[1], criterion=Resultado_rs.x[0],
  max_depth=Resultado_rs.x[2], min_samples_split=Resultado_rs.x[3],
  min_samples_leaf=Resultado_rs.x[4])

  RF.fit(Vetor_X, Vetor_Y)
  opiniao = RF.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")

def rf_bayesian_optimization():
    parametros = [('gini', 'entropy'), (10, 20, 30, 50, 75, 100), (3, 4, 5, 6, 7), (5, 6, 8, 10), (3, 4, 5, 6)]

    #print("\nAvaliação da Otimização Bayesiana")
    Resultado_go = gp_minimize(treinar_modelo_rf, parametros, verbose=0, n_calls=30, n_random_starts=10)

    #print("\nMelhores parâmetros")
    #print("criterion: ", Resultado_go.x[0], "n_estimators: ", Resultado_go.x[1], "max_depth: ", Resultado_go.x[2],"min_samples_split: ", Resultado_go.x[3], "min_samples_leaf: ", Resultado_go.x[4])

    #print("\nDesempenho sobre o teste")
    RF = RandomForestClassifier(n_estimators=Resultado_go.x[1], criterion=Resultado_go.x[0],
                                max_depth=Resultado_go.x[2], min_samples_split=Resultado_go.x[3],
                                min_samples_leaf=Resultado_go.x[4])
    RF.fit(Vetor_X, Vetor_Y)
    opiniao = RF.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    return Acc
    #print("Acurácia: ", Acc)


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
    gs_acc.append(rf_grid_search())
    fim = time.time()
    tempo_total = fim - inicio
    gs_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf grid search: {tempo_total} segundos")

    inicio = time.time()
    rs_acc.append(rf_random_search())
    fim = time.time()
    tempo_total = fim - inicio
    rs_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf random search: {tempo_total} segundos")

    inicio = time.time()
    do_acc.append(rf_dummy_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    do_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf dummy opt: {tempo_total} segundos")

    inicio = time.time()
    bo_acc.append(rf_bayesian_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    bo_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf bayesian opt: {tempo_total} segundos")

    inicio = time.time()
    cv_acc.append(rf_cross_validation())
    fim = time.time()
    tempo_total = fim - inicio
    cv_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf cross validation: {tempo_total} segundos")

    inicio = time.time()
    sh_acc.append(rf_sucessive_halving())
    fim = time.time()
    tempo_total = fim - inicio
    sh_tempo.append(tempo_total)
    #print(f"Tempo de execução no rf sucessive halving: {tempo_total} segundos")

with open('./stats/RFstats.txt', 'w') as arquivo:
   
  arquivo.write(f"Média de tempo no grid search: {media_valores(gs_tempo)}\n")
  arquivo.write(f"Média de tempo no random search: {media_valores(rs_tempo)}\n")
  arquivo.write(f"Média de tempo no dummy opt: {media_valores(do_tempo)}\n")
  arquivo.write(f"Média de tempo no bayesian opt: {media_valores(bo_tempo)}\n")
  arquivo.write(f"Média de tempo no sucessive halving: {media_valores(sh_tempo)}\n")
  arquivo.write(f"Média de tempo no cross validation: {media_valores(cv_tempo)}\n")
    
  arquivo.write("\nACC:\n")
    
  arquivo.write(f"Média de acc no grid search: {media_valores(gs_acc)}\n")
  arquivo.write(f"Média de acc no random search: {media_valores(rs_acc)}\n")
  arquivo.write(f"Média de acc no dummy opt: {media_valores(do_acc)}\n")
  arquivo.write(f"Média de acc no bayesian opt: {media_valores(bo_acc)}\n")
  arquivo.write(f"Média de acc no sucessive halving: {media_valores(sh_acc)}\n")
  arquivo.write(f"Média de acc no cross validation: {media_valores(cv_acc)}\n")