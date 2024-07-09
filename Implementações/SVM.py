import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from skopt import dummy_minimize
from skopt import gp_minimize

import time

def svm_grid_search():
    maior = -1

    for i in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1):
        for j in ('linear', 'poly', 'rbf', 'sigmoid'):
            SVM = svm.SVC(C=i, kernel=j)
            SVM.fit(x_treino, y_treino)

            opiniao = SVM.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)

            if Acc > maior:
                maior = Acc
                Melhor_i = i
                Melhor_j = j

    #print("Avaliação da grid SVM search\n")

    #print("Acc do svm:", maior)
    #print("C =", Melhor_i)
    #print("Kernel =", Melhor_j)

    #print("\nDesempenho sobre o teste")
    SVM = svm.SVC(C=Melhor_i, kernel=Melhor_j)
    SVM.fit(x_treino, y_treino)
    opiniao = SVM.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    return Acc
    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")


def svm_random_search():
    maior = -1

    for _ in range(15):
        i = random.choice((0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1))
        j = random.choice(('linear', 'poly', 'rbf', 'sigmoid'))
        SVM = svm.SVC(C=i, kernel=j)
        SVM.fit(x_treino, y_treino)

        opiniao = SVM.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if Acc > maior:
            maior = Acc
            Melhor_i = i
            Melhor_j = j

    #print("Avaliação da random SVM search\n")

    #print("Acc do random svm:", maior)
    #print("C =", Melhor_i)
    #print("Kernel =", Melhor_j)

    #print("\nDesempenho sobre o teste")
    SVM = svm.SVC(C=Melhor_i, kernel=Melhor_j)
    SVM.fit(x_treino, y_treino)
    opiniao = SVM.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    return Acc
    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")

def svm_cross_validation():
  parametros= {'C': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
  SVM = svm.SVC()

  #print("Avaliação da cross validation SVM\n")

  Classificador = GridSearchCV(estimator=SVM,param_grid=parametros,scoring='accuracy',cv=5)
  Classificador.fit(Vetor_X,Vetor_Y)

  pd.DataFrame(Classificador.cv_results_)

  #print("Melhores parâmetros: ",Classificador.best_params_)
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste")

  SVM = svm.SVC(C=Classificador.best_params_['C'], kernel = Classificador.best_params_['kernel'])
  SVM.fit(x_treino, y_treino)
  opiniao = SVM.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")

def svm_sucessive_halving():

  parametros= {'C': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
  SVM = svm.SVC()

  #print("Avaliação da sucessive halving SVM\n")


  Classificador = HalvingGridSearchCV(SVM, parametros,cv=5)
  Classificador.fit(x_treino,y_treino)

  pd.DataFrame(Classificador.cv_results_)

  #print("Melhor configuração: ",Classificador.best_params_ )
  #print("Melhor desempenho: ",Classificador.best_score_)

  #print("\nDesempenho sobre o teste ")

  SVM = svm.SVC(C=Classificador.best_params_['C'], kernel = Classificador.best_params_['kernel'])
  SVM.fit(x_treino, y_treino)
  opiniao = SVM.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc
  #print("Acurácia: ", Acc)
  #print("\n=========================================================================\n")

def treinar_modelo_svm(params):
    c_rate = params[0]
    kernell = params[1]

    SVM = svm.SVC(C=c_rate, kernel=kernell)
    SVM.fit(Vetor_X, Vetor_Y)
    opiniao = SVM.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def svm_dummy_optimization():
    parametros = [(0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1), ('linear', 'poly', 'rbf', 'sigmoid')]

    #print("Avaliação Teste usando Dummy")
    Resultado_rs = dummy_minimize(treinar_modelo_svm, parametros, verbose=0, n_calls=30)

    #print("\nMelhores parâmetros")
    #print("C: ", Resultado_rs.x[0], "Kernel: ", Resultado_rs.x[1])

    #print("\nDesempenho sobre o teste")
    SVM = svm.SVC(C=Resultado_rs.x[0], kernel=Resultado_rs.x[1])
    SVM.fit(Vetor_X, Vetor_Y)
    opiniao = SVM.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    return Acc
    #print("Acurácia: ", Acc)
    #print("\n=========================================================================\n")

def svm_bayesian_optimization():
    parametros = [(0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1), ('linear', 'poly', 'rbf', 'sigmoid')]

    #print("Avaliação da Otimização Bayesiana")
    Resultado_go = gp_minimize(treinar_modelo_svm, parametros, verbose=0, n_calls=30, n_random_starts=10)

    #print("\nMelhores parâmetros")
    #print("C: ", Resultado_go.x[0], "Kernel: ", Resultado_go.x[1])

    #print("\nDesempenho sobre o teste")
    SVM = svm.SVC(C=Resultado_go.x[0], kernel=Resultado_go.x[1])
    SVM.fit(Vetor_X, Vetor_Y)
    opiniao = SVM.predict(x_teste)
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


for _ in range(10):

    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste= train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    inicio = time.time()
    gs_acc.append(svm_grid_search())
    fim = time.time()
    tempo_total = fim - inicio
    gs_tempo.append(tempo_total)
    #print(f"Tempo de execução no svm grid search: {tempo_total} segundos")

    inicio = time.time()
    rs_acc.append(svm_random_search())
    fim = time.time()
    tempo_total = fim - inicio
    rs_tempo.append(tempo_total)
    #print(f"Tempo de execução no svm random search: {tempo_total} segundos")

    inicio = time.time()
    do_acc.append(svm_dummy_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    do_tempo.append(tempo_total)
    #print(f"Tempo de execução no svm dummy opt: {tempo_total} segundos")

    inicio = time.time()
    bo_acc.append(svm_bayesian_optimization())
    fim = time.time()
    tempo_total = fim - inicio
    bo_tempo.append(tempo_total)
    ##print(f"Tempo de execução no svm bayesian opt: {tempo_total} segundos")

    inicio = time.time()
    cv_acc.append(svm_cross_validation())
    fim = time.time()
    tempo_total = fim - inicio
    cv_tempo.append(tempo_total)
    ##print(f"Tempo de execução no svm cross validation: {tempo_total} segundos")

    inicio = time.time()
    sh_acc.append(svm_sucessive_halving())
    fim = time.time()
    tempo_total = fim - inicio
    sh_tempo.append(tempo_total)
    ##print(f"Tempo de execução no svm sucessive halving: {tempo_total} segundos")

with open('./stats/SVMstats.txt', 'w') as arquivo:

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