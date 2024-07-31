from sklearn.model_selection import train_test_split

import pandas as pd
import random

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from skopt import dummy_minimize
from skopt import gp_minimize

import time

def knn_grid_search():
  
  #print("\nAvaliação usando Grid search\n")
  maior = -1

  for j in ("distance","uniform"):
    for i in range (1,50):
      KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
      KNN.fit(x_treino,y_treino)

      opiniao = KNN.predict(x_validacao)
      Acc = accuracy_score(y_validacao, opiniao)

      if (Acc > maior):
        maior = Acc
        Melhor_k = i
        Melhor_metrica = j


  #print("Melhor Acc com grid search:", maior)
  #print("Melhor número de vizinhos:", Melhor_k)
  #print("Melhor tipo de distancia:", Melhor_metrica)

  #print("\nDesempenho sobre o teste")
  KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)

  KNN.fit(x_treino,y_treino)

  opiniao = KNN.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")


def knn_random_search():
  
  maior = -1

  for k in range (100):

    j =  random.choice(("distance","uniform"))
    i = random.randint(1,50)

    KNN = KNeighborsClassifier(n_neighbors=i,weights=j)
    KNN.fit(x_treino,y_treino)

    opiniao = KNN.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)

    if (Acc > maior):
      maior = Acc
      Melhor_k = i
      Melhor_metrica = j

  #print("Avaliação Teste usando busca aleatória\n")
  #print("\nMelhor acc com randomized knn random:", maior)
  #print("Melhor número de vizinhos:", Melhor_k)
  #print("Melhor tipo de distancia:", Melhor_metrica)

  #print("\nDesempenho sobre o teste")

  KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)
  KNN.fit(x_treino,y_treino)
  opiniao = KNN.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)

  return Acc  
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")


def knn_cross_validation():
  parametros = {'weights': ['distance', 'uniform'], 'n_neighbors': list(range(1, 50))}

  KNN = KNeighborsClassifier()
  Classificador = GridSearchCV(estimator=KNN, param_grid=parametros,scoring='accuracy',cv=5)

  Classificador.fit(Vetor_X,Vetor_Y)
  pd.DataFrame(Classificador.cv_results_)

  #print("\n=========================================================================\n")
  #print("Avaliação do cross validation")

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")
  KNN = KNeighborsClassifier(n_neighbors=Classificador.best_params_['n_neighbors'], weights=Classificador.best_params_['weights'])
  KNN.fit(x_treino,y_treino)
  opiniao = KNN.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)

  return Acc  
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")

def knn_sucessive_halving():
  n_samples = len(Vetor_X)
  max_n_neighbors = min(n_samples, 20)

  parametros = {'weights': ['distance', 'uniform'], 'n_neighbors': list(range(1, max_n_neighbors))}

  KNN = KNeighborsClassifier()
  Classificador = HalvingGridSearchCV(KNN, parametros,cv=5)

  Classificador.fit(Vetor_X,Vetor_Y)
  pd.DataFrame(Classificador.cv_results_)

  #print("Avaliação do sucessive halving")

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")
  KNN = KNeighborsClassifier(n_neighbors=Classificador.best_params_['n_neighbors'], weights=Classificador.best_params_['weights'])
  KNN.fit(x_treino, y_treino)
  opiniao = KNN.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)
  
  return Acc  

  #print("Acurácia: ", Acc)

def treinar_modelo_knn(params):
    n_neighbor = params[1]
    weight = params[0]

    KNN = KNeighborsClassifier(n_neighbors=n_neighbor, weights=weight)
    KNN.fit(x_treino, y_treino)

    opiniao = KNN.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)


def knn_dummy_optimization():
    parametros = [('distance', 'uniform'), (list(range(1, 50)))]

    #print("\nAvaliação Teste usando Dummy")
    Resultado_rs = dummy_minimize(treinar_modelo_knn, parametros, verbose=0, n_calls=30)

    #print("\n\nMelhores parâmetros")
    #print("Tipo de distancia: ", Resultado_rs.x[0], " Número de vizinhos: ", Resultado_rs.x[1])

    #print("\nDesempenho sobre o teste")
    KNN = KNeighborsClassifier(n_neighbors=Resultado_rs.x[1], weights=Resultado_rs.x[0])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    return Acc  

    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")


def knn_bayesian_optimization():
    parametros = [('distance', 'uniform'), (list(range(1, 50)))]

    #print("Avaliação da Otimização Bayesiana")
    Resultado_go = gp_minimize(treinar_modelo_knn, parametros, verbose=0, n_calls=30, n_random_starts=10)

    #print("Melhores parâmetros")
    #print("Tipo de distancia: ", Resultado_go.x[0], " Número de vizinhos: ", Resultado_go.x[1])

    #print("\nDesempenho sobre o teste")
    KNN = KNeighborsClassifier(n_neighbors=Resultado_go.x[1], weights=Resultado_go.x[0])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    return Acc  

    #print("Acurácia: ", Acc)

def media_valores(lista):

    menor_numero = min(lista)
    maior_numero = max(lista)

    lista.remove(maior_numero)
    lista.remove(menor_numero)

    return (sum(lista)/len(lista))


dados = pd.read_csv("../datasets/letter-recognition.csv")
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
    gs_acc.append(knn_grid_search())
    fim = time.time()
    tempo_total = fim - inicio
    gs_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN grid search: {tempo_total} segundos")

    inicio = time.time()
    rs_acc.append(knn_random_search())
    fim = time.time()
    tempo_total = fim - inicio
    rs_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN random search: {tempo_total} segundos")

    inicio = time.time()
    do_acc.append(knn_dummy_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    do_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN dummy opt: {tempo_total} segundos")

    inicio = time.time()
    bo_acc.append(knn_bayesian_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    bo_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN bayesian opt: {tempo_total} segundos")

    inicio = time.time()
    cv_acc.append(knn_cross_validation())
    fim = time.time()
    tempo_total = fim - inicio
    cv_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN cross validation: {tempo_total} segundos")

    inicio = time.time()
    sh_acc.append(knn_sucessive_halving())
    fim = time.time()
    tempo_total = fim - inicio
    sh_tempo.append(tempo_total)
    #print(f"Tempo de execução no KNN sucessive halving: {tempo_total} segundos")

import pandas as pd

estrategias = ["Grid Search", "Random Search", "Dummy Opt", "Bayesian Opt", "Successive Halving", "Cross Validation"]
tempos = [media_valores(gs_tempo), media_valores(rs_tempo), media_valores(do_tempo), media_valores(bo_tempo), media_valores(sh_tempo), media_valores(cv_tempo)]
acuracias = [media_valores(gs_acc), media_valores(rs_acc), media_valores(do_acc), media_valores(bo_acc), media_valores(sh_acc), media_valores(cv_acc)]

df = pd.DataFrame({
    'Estrategia': estrategias,
    'Tempo': tempos,
    'Acc': acuracias
})

df.set_index('Estrategia', inplace=True)

df.to_csv('./stats/KNNstats.csv')
