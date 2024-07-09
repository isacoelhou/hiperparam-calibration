import pandas as pd
import random
from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from skopt import dummy_minimize
from skopt import gp_minimize

import time

def dt_grid_search():
  maior = -1

  for i in  ("best", "random"):
    for j in ("gini", "entropy"):
        for k in (3,4,5,6,7):
          for l in (5,6,8,10):
            for m in (3,4,5,6):

              DT = tree.DecisionTreeClassifier(criterion=j,splitter=i,max_depth=k, min_samples_split = l,min_samples_leaf=m)
              DT.fit(x_treino,y_treino)

              opiniao = DT.predict(x_validacao)
              Acc = accuracy_score(y_validacao, opiniao)

              if (Acc > maior):
                maior = Acc
                Melhor_i = i
                Melhor_j = j
                Melhor_k = k
                Melhor_l = l
                Melhor_m = m

  #print("Avaliação da decision tree grid search\n")
  #print("ACC: ", maior)
  #print("\nCritério:",Melhor_j,"\nSplit:",Melhor_i,"\nProfundidade:",Melhor_k, "\nMínimo para divisão:",Melhor_l, "\nMínimo por folha:",Melhor_m)

  #print("Desempenho sobre o teste")
  DT = tree.DecisionTreeClassifier(criterion=Melhor_j,splitter=Melhor_i,max_depth=Melhor_k, min_samples_split = Melhor_k,min_samples_leaf=Melhor_m)
  DT.fit(x_treino,y_treino)
  opiniao = DT.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)

  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")

def dt_random_search():
  maior = -1
  for n in range (100):
    i =  random.choice(("best", "random"))
    j = random.choice(("gini", "entropy"))
    k = random.randint(3,7)
    l = random.choice((5,6,8,10))
    m = random.randint(3,6)

    DT = tree.DecisionTreeClassifier(criterion=j,splitter=i,max_depth=k, min_samples_split = l,min_samples_leaf=m)
    DT.fit(x_treino,y_treino)

    opiniao = DT.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)

    if (Acc > maior):
      maior = Acc
      Melhor_i = i
      Melhor_j = j
      Melhor_k = k
      Melhor_l = l
      Melhor_m = m

  #print("Avaliação da decision tree random search\n")


  #print("ACC: ", maior)
  #print("\nCritério:",Melhor_j,"\nSplit:",Melhor_i,"\nProfundidade:",Melhor_k, "\nMínimo para divisão:",Melhor_l, "\nMínimo por folha:",Melhor_m)

  #print("\nDesempenho sobre o teste")
  DT = tree.DecisionTreeClassifier(criterion=Melhor_j,splitter=Melhor_i,max_depth=Melhor_k, min_samples_split = Melhor_k,min_samples_leaf=Melhor_m)
  DT.fit(x_treino,y_treino)
  opiniao = DT.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)

  return Acc
  #print("Acurácia: ",Acc)
  #print("\n=========================================================================\n")

def dt_cross_validation():
  parametros = {'criterion':('gini', 'entropy'), 'splitter':('best','random'),'max_depth':[3,4,5,7,10], 'min_samples_split':[3,7] ,'min_samples_leaf':[1,3,5]}

  DT = tree.DecisionTreeClassifier()
  Classificador = GridSearchCV(estimator=DT,param_grid=parametros,scoring='accuracy',cv=5)

  Classificador.fit(Vetor_X,Vetor_Y)
  pd.DataFrame(Classificador.cv_results_)

  #print("Avaliação da decision tree cross validation\n")


  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("Avaliação do sucessive halving")

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")
  DT = tree.DecisionTreeClassifier(criterion=Classificador.best_params_['criterion'], splitter=Classificador.best_params_['splitter'], max_depth=Classificador.best_params_['max_depth'], min_samples_split=Classificador.best_params_['min_samples_split'], min_samples_leaf=Classificador.best_params_['min_samples_leaf'])
  DT.fit(x_treino, y_treino)
  opiniao = DT.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")


def dt_sucessive_halving():
  parametros = {'criterion':('gini', 'entropy'), 'splitter':('best','random'),'min_samples_split':[3,7], 'max_depth':[3,4,5,7,10],'min_samples_leaf':[1,3,5]}
  DT = tree.DecisionTreeClassifier()
  Classificador = HalvingGridSearchCV(DT, parametros,cv=5)

  Classificador.fit(Vetor_X,Vetor_Y)

  #print("Avaliação da decision tree sucessive halving\n")

  #print("Melhor configuração: ",Classificador.best_params_ )
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")
  DT = tree.DecisionTreeClassifier(criterion=Classificador.best_params_['criterion'], splitter=Classificador.best_params_['splitter'], max_depth=Classificador.best_params_['max_depth'], min_samples_split=Classificador.best_params_['min_samples_split'], min_samples_leaf=Classificador.best_params_['min_samples_leaf'])
  DT.fit(x_treino, y_treino)
  opiniao = DT.predict(x_teste)
  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")

def treinar_modelo_dt(params):
    crit = params[0]
    split = params[1]
    max_d = params[2]
    min_sl = params[3]

#faltou adicionar o parâmetro min_samples_split no otimização bayesiana e na dummy
    parametros = {'criterion':('gini', 'entropy'), 'splitter':('best','random'), 'min_samples_split':[3,7],'max_depth':[3,4,5,7,10],'min_samples_leaf':[1,3,5]}

    DT = tree.DecisionTreeClassifier(criterion=crit, splitter=split, max_depth=max_d, min_samples_leaf=min_sl)
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def dt_dummy_optimization():
    parametros = [('gini', 'entropy'), ('best', 'random'), (3,7), (3, 4, 5, 7, 10), (1, 3, 5)]

    #print("Avaliação Teste usando Dummy")
    Resultado_rs = dummy_minimize(treinar_modelo_dt, parametros, verbose=0, n_calls=30)

    #print("\nMelhores parâmetros")
    #print("Critério: ", Resultado_rs.x[0], "Splitter: ", Resultado_rs.x[1], "Profundidade Max: ", Resultado_rs.x[2], "Min Folha: ", Resultado_rs.x[3])

    #print("\nDesempenho sobre o teste")
    DT = tree.DecisionTreeClassifier(criterion=Resultado_rs.x[0], splitter=Resultado_rs.x[1], max_depth=Resultado_rs.x[2], min_samples_leaf=Resultado_rs.x[3])
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)
    return Acc
    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")

def dt_bayesian_optimization():
    parametros = [('gini', 'entropy'), ('best', 'random'), (3,7), (3, 4, 5, 7, 10), (1, 3, 5)]

    #print("Avaliação da Otimização Bayesiana")
    Resultado_go = gp_minimize(treinar_modelo_dt, parametros, verbose=0, n_calls=30, n_random_starts=10)

    #print("\nMelhores parâmetros")
    #print("Critério: ", Resultado_go.x[0], "Splitter: ", Resultado_go.x[1], "Profundidade Max: ", Resultado_go.x[2], "Min Folha: ", Resultado_go.x[3])

    #print("\nDesempenho sobre o teste")
    DT = tree.DecisionTreeClassifier(criterion=Resultado_go.x[0], splitter=Resultado_go.x[1], max_depth=Resultado_go.x[2], min_samples_leaf=Resultado_go.x[3])
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_validacao)
    Acc = accuracy_score(y_validacao, opiniao)
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

for _ in range(10):

    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste= train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    inicio = time.time()
    gs_acc.append(dt_grid_search())
    fim = time.time()
    tempo_total = fim - inicio
    gs_tempo.append(tempo_total)
    #print(f"Tempo de execução no dt grid search: {tempo_total} segundos")

    inicio = time.time()
    rs_acc.append(dt_random_search())
    fim = time.time()
    tempo_total = fim - inicio
    rs_tempo.append(tempo_total)
    #print(f"Tempo de execução no dt random search: {tempo_total} segundos")

    inicio = time.time()
    do_acc.append(dt_dummy_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    do_tempo.append(tempo_total)
    #print(f"Tempo de execução no dt dummy opt: {tempo_total} segundos")

    inicio = time.time()
    bo_acc.append(dt_bayesian_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    bo_tempo.append(tempo_total)
    ##print(f"Tempo de execução no dt bayesian opt: {tempo_total} segundos")

    inicio = time.time()
    cv_acc.append(dt_cross_validation())
    fim = time.time()
    tempo_total = fim - inicio
    cv_tempo.append(tempo_total)
    ##print(f"Tempo de execução no dt cross validation: {tempo_total} segundos")

    inicio = time.time()
    sh_acc.append(dt_sucessive_halving())
    fim = time.time()
    tempo_total = fim - inicio
    sh_tempo.append(tempo_total)
    ##print(f"Tempo de execução no dt sucessive halving: {tempo_total} segundos")

print("Média de tempo no grid search: ",media_valores(gs_tempo))
print("Média de tempo no random search: ", media_valores(rs_tempo))
print("Média de tempo no dummy opt " ,media_valores(do_tempo))
print("Média de tempo no bayesian opt: " , media_valores(bo_tempo))
print("Média de tempo no sucessive halving: " , media_valores(sh_tempo))
print("Média de tempo no cross validation: " , media_valores(cv_tempo))

print(gs_acc)

print("Média de acc no grid search: ",media_valores(gs_acc))
print("Média de acc no random search: ", media_valores(rs_acc))
print("Média de acc no dummy opt " ,media_valores(do_acc))
print("Média de acc no bayesian opt: " , media_valores(bo_acc))
print("Média de acc no sucessive halving: " , media_valores(sh_acc))
print("Média de acc no cross validation: " , media_valores(cv_acc))