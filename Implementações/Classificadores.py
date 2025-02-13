import pandas as pd
import random
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from skopt import gp_minimize
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import time

import os

import os

def save_params(params, filename):
    pasta_params = "Params"
    filename += ".txt"

    caminho_arquivo = os.path.join(pasta_params, filename)
    with open(caminho_arquivo, "a") as f:
        f.write(f"{params}\n")

def dt_grid_search():
    maior = -1

    for i in ("best", "random"):
        for j in ("gini", "entropy"):
            for k in (3, 4, 5, 6, 7):
                for l in (5, 6, 8, 10):
                    for m in (3, 4, 5, 6):
                        DT = tree.DecisionTreeClassifier(criterion=j, splitter=i, max_depth=k, min_samples_split=l, min_samples_leaf=m)
                        DT.fit(x_treino, y_treino)

                        opiniao = DT.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)

                        if (Acc > maior):
                            maior = Acc
                            Melhor_i = i
                            Melhor_j = j
                            Melhor_k = k
                            Melhor_l = l
                            Melhor_m = m

    DT = tree.DecisionTreeClassifier(criterion=Melhor_j, splitter=Melhor_i, max_depth=Melhor_k, min_samples_split=Melhor_l, min_samples_leaf=Melhor_m)
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l, Melhor_m), "DT_PARAMS")

    return Acc

def dt_random_search():
    maior = -1
    for n in range(100):
        i = random.choice(("best", "random"))
        j = random.choice(("gini", "entropy"))
        k = random.randint(3, 7)
        l = random.randint(5,10)
        m = random.randint(3,6)

        DT = tree.DecisionTreeClassifier(criterion=j, splitter=i, max_depth=k, min_samples_split=l, min_samples_leaf=m)
        DT.fit(x_treino, y_treino)

        opiniao = DT.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if (Acc > maior):
            maior = Acc
            Melhor_i = i
            Melhor_j = j
            Melhor_k = k
            Melhor_l = l
            Melhor_m = m

    DT = tree.DecisionTreeClassifier(criterion=Melhor_j, splitter=Melhor_i, max_depth=Melhor_k, min_samples_split=Melhor_l, min_samples_leaf=Melhor_m)
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l, Melhor_m), "DT_PARAMS")

    return Acc

def dt_cross_validation():
    parametros = {'criterion':('gini', 'entropy'), 'splitter':('best','random'),'max_depth':[3,4,5,7,10], 'min_samples_split':[3,7] ,'min_samples_leaf':[2,3,5]}

    DT = tree.DecisionTreeClassifier()
    Classificador = GridSearchCV(estimator=DT,param_grid=parametros,scoring='accuracy',cv=5)

    Classificador.fit(Vetor_X,Vetor_Y)
    pd.DataFrame(Classificador.cv_results_)
    
    DT = tree.DecisionTreeClassifier(criterion=Classificador.best_params_['criterion'], splitter=Classificador.best_params_['splitter'], max_depth=Classificador.best_params_['max_depth'], min_samples_split=Classificador.best_params_['min_samples_split'], min_samples_leaf=Classificador.best_params_['min_samples_leaf'])
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    best_params = Classificador.best_params_
    
    save_params((best_params['splitter'], best_params['criterion'], best_params['max_depth'], best_params['min_samples_split'], best_params['min_samples_leaf']),  "DT_PARAMS")

    return Acc

def dt_sucessive_halving():

    parametros = {'criterion':('gini', 'entropy'), 'splitter':('best','random'),'min_samples_split':[3,7], 'max_depth':[3,4,5,7,10],'min_samples_leaf':[3,5]}
    DT = tree.DecisionTreeClassifier()
    Classificador = HalvingGridSearchCV(DT, parametros,cv=5)

    Classificador.fit(Vetor_X,Vetor_Y)

    DT = tree.DecisionTreeClassifier(criterion=Classificador.best_params_['criterion'], splitter=Classificador.best_params_['splitter'], max_depth=Classificador.best_params_['max_depth'], min_samples_split=Classificador.best_params_['min_samples_split'], min_samples_leaf=Classificador.best_params_['min_samples_leaf'])
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    best_params = Classificador.best_params_
    
    save_params((best_params['splitter'], best_params['criterion'], best_params['max_depth'], best_params['min_samples_split'], best_params['min_samples_leaf']), "DT_PARAMS")

    return Acc

def treinar_modelo_dt(params):
    crit = params[0]
    split = params[1]
    min_ss = params[2]
    max_d = params[3]
    min_sl = params[4]

    DT = tree.DecisionTreeClassifier(criterion=crit, splitter=split, max_depth=max_d, min_samples_leaf=min_sl, min_samples_split=min_ss)
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def dt_bayesian_optimization():
    parametros = [('gini', 'entropy'), ('best', 'random'), (3,7), (3, 4, 5, 7, 10), (3, 5)]

    Resultado_go = gp_minimize(treinar_modelo_dt, parametros, verbose=0, n_calls=30, n_random_starts=10)

    DT = tree.DecisionTreeClassifier(criterion=Resultado_go.x[0], splitter=Resultado_go.x[1], max_depth=Resultado_go.x[2], min_samples_leaf=Resultado_go.x[3], min_samples_split = Resultado_go.x[4])
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    save_params((Resultado_go.x[0], Resultado_go.x[1], Resultado_go.x[2], Resultado_go.x[3], Resultado_go.x[4]), "DT_PARAMS")
    return Acc

def knn_grid_search():  
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

  KNN = KNeighborsClassifier(n_neighbors=Melhor_k,weights=Melhor_metrica)

  KNN.fit(x_treino,y_treino)

  opiniao = KNN.predict(x_teste)

  Acc = accuracy_score(y_teste, opiniao)
  return Acc

def knn_grid_search():  
    maior = -1

    for j in ("distance", "uniform"):
        for i in range(1, 50):
            KNN = KNeighborsClassifier(n_neighbors=i, weights=j)
            KNN.fit(x_treino, y_treino)

            opiniao = KNN.predict(x_validacao)
            Acc = accuracy_score(y_validacao, opiniao)

            if Acc > maior:
                maior = Acc
                Melhor_k = i
                Melhor_metrica = j

    KNN = KNeighborsClassifier(n_neighbors=Melhor_k, weights=Melhor_metrica)
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_k, Melhor_metrica), "KNN_params")

    return Acc

def knn_random_search():
    maior = -1

    for k in range(100):
        j = random.choice(("distance", "uniform"))
        i = random.randint(1, 50)

        KNN = KNeighborsClassifier(n_neighbors=i, weights=j)
        KNN.fit(x_treino, y_treino)

        opiniao = KNN.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if Acc > maior:
            maior = Acc
            Melhor_k = i
            Melhor_metrica = j

    KNN = KNeighborsClassifier(n_neighbors=Melhor_k, weights=Melhor_metrica)
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_k, Melhor_metrica), "KNN_params")

    return Acc  

def knn_cross_validation():
    parametros = {'weights': ['distance', 'uniform'], 'n_neighbors': list(range(1, 50))}

    KNN = KNeighborsClassifier()
    Classificador = GridSearchCV(estimator=KNN, param_grid=parametros, scoring='accuracy', cv=5)

    Classificador.fit(Vetor_X, Vetor_Y)
    pd.DataFrame(Classificador.cv_results_)

    KNN = KNeighborsClassifier(n_neighbors=Classificador.best_params_['n_neighbors'], weights=Classificador.best_params_['weights'])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Classificador.best_params_['n_neighbors'], Classificador.best_params_['weights']), "KNN_params")

    return Acc  

def knn_sucessive_halving():
    n_samples = len(Vetor_X)
    max_n_neighbors = min(n_samples, 20)

    parametros = {'weights': ['distance', 'uniform'], 'n_neighbors': list(range(1, max_n_neighbors))}

    KNN = KNeighborsClassifier()
    Classificador = HalvingGridSearchCV(KNN, parametros, cv=5)

    Classificador.fit(Vetor_X, Vetor_Y)
    pd.DataFrame(Classificador.cv_results_)

    KNN = KNeighborsClassifier(n_neighbors=Classificador.best_params_['n_neighbors'], weights=Classificador.best_params_['weights'])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Classificador.best_params_['n_neighbors'], Classificador.best_params_['weights']), "KNN_params")

    return Acc  

def treinar_modelo_knn(params):
    n_neighbor = params[1]
    weight = params[0]

    KNN = KNeighborsClassifier(n_neighbors=n_neighbor, weights=weight)
    KNN.fit(x_treino, y_treino)

    opiniao = KNN.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def knn_bayesian_optimization():
    parametros = [('distance', 'uniform'), list(range(1, 50))]

    Resultado_go = gp_minimize(treinar_modelo_knn, parametros, verbose=0, n_calls=30, n_random_starts=10)

    KNN = KNeighborsClassifier(n_neighbors=Resultado_go.x[1], weights=Resultado_go.x[0])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Resultado_go.x[1], Resultado_go.x[0]), "KNN_params")

    return Acc

def mlp_random_search(numero_colunas):
    maior = -1

    for _ in range(50):
        i = random.choice((numero_colunas, (2 * numero_colunas)))
        j = random.choice(('constant', 'invscaling', 'adaptive'))
        k = random.randint(50, 1000)
        l = random.choice(('identity', 'logistic', 'tanh', 'relu'))

        MLP = MLPClassifier(hidden_layer_sizes=(i, i, i), learning_rate=j, max_iter=k, activation=l, verbose=False)
        MLP.fit(x_treino, y_treino)

        opiniao = MLP.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if Acc > maior:
            maior = Acc
            Melhor_i = i
            Melhor_j = j
            Melhor_k = k
            Melhor_l = l

    MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i, Melhor_i, Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l, verbose=False)
    MLP.fit(x_treino, y_treino)
    opiniao = MLP.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l), "MLP_params")
    
    return Acc

def mlp_grid_search(numero_colunas):
    maior = -1

    for i in (numero_colunas, 2 * numero_colunas):
        for j in ('constant', 'invscaling', 'adaptive'):
            for k in (50, 100, 150, 300, 500, 1000):
                for l in ('identity', 'logistic', 'tanh', 'relu'):
                    MLP = MLPClassifier(hidden_layer_sizes=(i, i, i), learning_rate=j, max_iter=k, activation=l)
                    MLP.fit(x_treino, y_treino)

                    opiniao = MLP.predict(x_validacao)
                    Acc = accuracy_score(y_validacao, opiniao)

                    if Acc > maior:
                        maior = Acc
                        Melhor_i = i
                        Melhor_j = j
                        Melhor_k = k
                        Melhor_l = l

    MLP = MLPClassifier(hidden_layer_sizes=(Melhor_i, Melhor_i, Melhor_i), learning_rate=Melhor_j, max_iter=Melhor_k, activation=Melhor_l, verbose=False)
    MLP.fit(x_treino, y_treino)
    opiniao = MLP.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l), "MLP_params")
    
    return Acc

def mlp_cross_validation(numero_colunas):
    parametros = {
        'hidden_layer_sizes': [(numero_colunas, numero_colunas, numero_colunas), (2 * numero_colunas, 2 * numero_colunas, 2 * numero_colunas)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [50, 100, 150, 300, 500, 1000],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }
    MLP = MLPClassifier()

    Classificador = GridSearchCV(estimator=MLP, param_grid=parametros, scoring='accuracy', cv=5)
    Classificador.fit(Vetor_X, Vetor_Y)

    MLP_best = MLPClassifier(
        hidden_layer_sizes=Classificador.best_params_['hidden_layer_sizes'],
        learning_rate=Classificador.best_params_['learning_rate'],
        max_iter=Classificador.best_params_['max_iter'],
        activation=Classificador.best_params_['activation']
    )

    MLP_best.fit(x_treino, y_treino)
    opiniao = MLP_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Classificador.best_params_['hidden_layer_sizes'], Classificador.best_params_['learning_rate'], Classificador.best_params_['max_iter'], Classificador.best_params_['activation']), "MLP_params")

    return Acc

def mlp_sucessive_halving(numero_colunas):
    parametros = {
        'hidden_layer_sizes': [(numero_colunas, numero_colunas, numero_colunas), (2 * numero_colunas, 2 * numero_colunas, 2 * numero_colunas)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [50, 100, 150, 300, 500, 1000],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    }

    MLP = MLPClassifier()
    Classificador = HalvingGridSearchCV(MLP, parametros, cv=5)

    Classificador.fit(Vetor_X, Vetor_Y)
    best_params = Classificador.best_params_

    MLP_best = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter'],
        activation=best_params['activation']
    )

    MLP_best.fit(x_treino, y_treino)
    opiniao = MLP_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params['hidden_layer_sizes'], best_params['learning_rate'], best_params['max_iter'], best_params['activation']), "MLP_params")
    
    return Acc

def treinar_modelo_mlp(params):
    hidden_layers = params[0]
    learning_rate = params[1]
    max_iter = params[2]
    activation = params[3]

    MLP = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate=learning_rate, max_iter=max_iter, activation=activation)
    MLP.fit(x_treino, y_treino)
    opiniao = MLP.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def mlp_bayesian_optimization(numero_colunas):
    parametros = [(numero_colunas,2*numero_colunas), ('constant', 'invscaling', 'adaptive'), (50, 100, 150, 300, 500, 1000), ('identity', 'logistic', 'tanh', 'relu')]

    Resultado_go = gp_minimize(treinar_modelo_mlp, parametros, verbose=0, n_calls=30, n_random_starts=10)

    MLP = MLPClassifier(hidden_layer_sizes=Resultado_go.x[0], learning_rate=Resultado_go.x[1], max_iter=Resultado_go.x[2], activation=Resultado_go.x[3])
    MLP.fit(Vetor_X, Vetor_Y)
    opiniao = MLP.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Resultado_go.x[0], Resultado_go.x[1], Resultado_go.x[2], Resultado_go.x[3]), "MLP_params")
    
    return Acc

def rf_grid_search():
    maior = -1

    for i in (10, 20, 30, 50, 75, 100):
        for j in ("gini", "entropy"):
            for k in (3, 4, 5, 6, 7):
                for l in (5, 6, 8, 10):
                    for m in (3, 4, 5, 6):
                        RF = RandomForestClassifier(n_estimators=i, criterion=j, max_depth=k, min_samples_split=l, min_samples_leaf=m)
                        RF.fit(x_treino, y_treino)

                        opiniao = RF.predict(x_validacao)
                        Acc = accuracy_score(y_validacao, opiniao)

                        if Acc > maior:
                            maior = Acc
                            Melhor_i = i
                            Melhor_j = j
                            Melhor_k = k
                            Melhor_l = l
                            Melhor_m = m

    RF = RandomForestClassifier(n_estimators=Melhor_i, criterion=Melhor_j, max_depth=Melhor_k, min_samples_split=Melhor_l, min_samples_leaf=Melhor_m)
    RF.fit(x_treino, y_treino)
    opiniao = RF.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l, Melhor_m), "RF_PARAMS")
    
    return Acc

def rf_random_search():
    maior = -1

    for _ in range(50):
        i = random.randint(10, 100)
        j = random.choice(("gini", "entropy"))
        k = random.randint(3,7)
        l = random.randint(5, 10)
        m = random.randint(3,6)

        RF = RandomForestClassifier(n_estimators=i, criterion=j, max_depth=k, min_samples_split=l, min_samples_leaf=m)
        RF.fit(x_treino, y_treino)

        opiniao = RF.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if Acc > maior:
            maior = Acc
            Melhor_i = i
            Melhor_j = j
            Melhor_k = k
            Melhor_l = l
            Melhor_m = m

    RF = RandomForestClassifier(n_estimators=Melhor_i, criterion=Melhor_j, max_depth=Melhor_k, min_samples_split=Melhor_l, min_samples_leaf=Melhor_m)
    RF.fit(x_treino, y_treino)
    opiniao = RF.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Melhor_i, Melhor_j, Melhor_k, Melhor_l, Melhor_m), "RF_PARAMS")
    
    return Acc

def rf_cross_validation():
    parametros = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 20, 30, 50, 75, 100],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [5, 6, 8, 10],
        'min_samples_leaf': [3, 4, 5, 6]
    }

    RF = RandomForestClassifier()
    Classificador = GridSearchCV(estimator=RF, param_grid=parametros, scoring='accuracy', cv=5)
    Classificador.fit(Vetor_X, Vetor_Y)

    RF_best = RandomForestClassifier(
        n_estimators=Classificador.best_params_['n_estimators'],
        criterion=Classificador.best_params_['criterion'],
        max_depth=Classificador.best_params_['max_depth'],
        min_samples_split=Classificador.best_params_['min_samples_split'],
        min_samples_leaf=Classificador.best_params_['min_samples_leaf']
    )
    RF_best.fit(x_treino, y_treino)
    opiniao = RF_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Classificador.best_params_['n_estimators'], Classificador.best_params_['criterion'], 
                 Classificador.best_params_['max_depth'], Classificador.best_params_['min_samples_split'], 
                 Classificador.best_params_['min_samples_leaf']), "RF_PARAMS")
    
    return Acc

def rf_sucessive_halving():
    parametros = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 20, 30, 50, 75, 100],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [5, 6, 8, 10],
        'min_samples_leaf': [3, 4, 5, 6]
    }

    RF = RandomForestClassifier()
    Classificador = HalvingGridSearchCV(RF, parametros, cv=5)
    Classificador.fit(Vetor_X, Vetor_Y)

    RF_best = RandomForestClassifier(
        criterion=Classificador.best_params_['criterion'], 
        n_estimators=Classificador.best_params_['n_estimators'], 
        max_depth=Classificador.best_params_['max_depth'],  
        min_samples_split=Classificador.best_params_['min_samples_split'], 
        min_samples_leaf=Classificador.best_params_['min_samples_leaf']
    )
    RF_best.fit(x_treino, y_treino)
    opiniao = RF_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Classificador.best_params_['criterion'], Classificador.best_params_['n_estimators'], 
                 Classificador.best_params_['max_depth'], Classificador.best_params_['min_samples_split'], 
                 Classificador.best_params_['min_samples_leaf']), "RF_PARAMS")
    
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

def rf_bayesian_optimization():
    parametros = [
        ('gini', 'entropy'), 
        (10, 20, 30, 50, 75, 100), 
        (3, 4, 5, 6, 7), 
        (5, 6, 8, 10), 
        (3, 4, 5, 6)
    ]

    Resultado_go = gp_minimize(treinar_modelo_rf, parametros, verbose=0, n_calls=30, n_random_starts=10)

    RF_best = RandomForestClassifier(
        n_estimators=Resultado_go.x[1], 
        criterion=Resultado_go.x[0],
        max_depth=Resultado_go.x[2], 
        min_samples_split=Resultado_go.x[3],
        min_samples_leaf=Resultado_go.x[4]
    )
    RF_best.fit(x_treino, y_treino)
    opiniao = RF_best.predict(x_teste)
    
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((Resultado_go.x[1], Resultado_go.x[0], Resultado_go.x[2], Resultado_go.x[3], Resultado_go.x[4]), "RF_PARAMS")
    
    return Acc

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

    SVM = svm.SVC(C=Melhor_i, kernel=Melhor_j)
    SVM.fit(x_treino, y_treino)
    opiniao = SVM.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Melhor_i, Melhor_j), "SVM_PARAMS")
    
    return Acc

def svm_random_search():
    maior = -1

    for _ in range(15):
        i = random.uniform(0.1, 1)
        j = random.choice(('linear', 'poly', 'rbf', 'sigmoid'))
        SVM = svm.SVC(C=i, kernel=j)
        SVM.fit(x_treino, y_treino)

        opiniao = SVM.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)

        if Acc > maior:
            maior = Acc
            Melhor_i = i
            Melhor_j = j

    SVM = svm.SVC(C=Melhor_i, kernel=Melhor_j)
    SVM.fit(x_treino, y_treino)
    opiniao = SVM.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Melhor_i, Melhor_j), "SVM_PARAMS")
    
    return Acc
    
def svm_cross_validation():
    parametros = {'C': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
    SVM = svm.SVC()

    Classificador = GridSearchCV(estimator=SVM, param_grid=parametros, scoring='accuracy', cv=5)
    Classificador.fit(Vetor_X, Vetor_Y)

    SVM_best = svm.SVC(C=Classificador.best_params_['C'], kernel=Classificador.best_params_['kernel'])
    SVM_best.fit(x_treino, y_treino)
    opiniao = SVM_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Classificador.best_params_['C'], Classificador.best_params_['kernel']), "SVM_PARAMS")
    
    return Acc

def svm_sucessive_halving():
    parametros = {'C': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1], 'kernel': ('linear', 'poly', 'rbf', 'sigmoid')}
    SVM = svm.SVC()

    Classificador = HalvingGridSearchCV(SVM, parametros, cv=5)
    Classificador.fit(Vetor_X, Vetor_Y)

    SVM_best = svm.SVC(C=Classificador.best_params_['C'], kernel=Classificador.best_params_['kernel'])
    SVM_best.fit(x_treino, y_treino)
    opiniao = SVM_best.predict(x_teste)

    Acc = accuracy_score(y_teste, opiniao)

    save_params((Classificador.best_params_['C'], Classificador.best_params_['kernel']), "SVM_PARAMS")
    
    return Acc

def treinar_modelo_svm(params):
    c_rate = params[0]
    kernel = params[1]

    SVM = svm.SVC(C=c_rate, kernel=kernel)
    SVM.fit(Vetor_X, Vetor_Y)
    opiniao = SVM.predict(x_validacao)
    return 1 - accuracy_score(y_validacao, opiniao)

def svm_bayesian_optimization():
    parametros = [(0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1), ('linear', 'poly', 'rbf', 'sigmoid')]

    Resultado_go = gp_minimize(treinar_modelo_svm, parametros, verbose=0, n_calls=30, n_random_starts=10)

    SVM_best = svm.SVC(C=Resultado_go.x[0], kernel=Resultado_go.x[1])
    SVM_best.fit(Vetor_X, Vetor_Y)
    opiniao = SVM_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    save_params((Resultado_go.x[0], Resultado_go.x[1]), "SVM_PARAMS")
    
    return Acc


def media_valores(lista):

    menor_numero = min(lista)
    maior_numero = max(lista)

    lista.remove(maior_numero)
    lista.remove(menor_numero)

    return (sum(lista)/len(lista))

def salvar_resultados_csv(metodo, estrategia, acc, tempo):
    file_path = f'./stats/{metodo}stats.csv'
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Estrategia", "Acc", "Tempo"]).set_index("Estrategia")
    
    df.loc[estrategia] = [acc, tempo]
    df.to_csv(file_path)

# # Inicialização das listas de tuplas
# gs_DT = []
# rs_DT = []
# cv_DT = []
# sh_DT = []
# bo_DT = []
# do_DT = []

# gs_KNN = []
# rs_KNN = []
# cv_KNN = []
# sh_KNN = []
# bo_KNN = []
# do_KNN = []

# gs_MLP = []
# rs_MLP = []
# cv_MLP = []
# sh_MLP = []
# bo_MLP = []
# do_MLP = []

# gs_RF = []
# rs_RF = []
# cv_RF = []
# sh_RF = []
# bo_RF = []
# do_RF = []

# gs_SVM = []
# rs_SVM = []
# cv_SVM = []
# sh_SVM = []
# bo_SVM = []
# do_SVM = []

for _ in range(10):

    dados = pd.read_csv("../datasets/letter-recognition.csv")
    dados = shuffle(dados)
    
    df_dados = pd.DataFrame(dados)
    numero_colunas = df_dados.shape[1]

    df_dados = df_dados.iloc[:, :-1]
    
    Vetor_X = df_dados
    Vetor_Y = dados["Class"]
    
    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste = train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

   # DT
    inicio = time.time()
    acc = dt_grid_search()
    fim = time.time()
    salvar_resultados_csv('DT', 'GS', fim - inicio, acc)
    
    inicio = time.time()
    acc = dt_random_search()
    fim = time.time()
    salvar_resultados_csv('DT', 'RS', fim - inicio, acc)


    inicio = time.time()
    acc = dt_bayesian_optimization()
    fim = time.time()
    salvar_resultados_csv('DT', 'BO', fim - inicio, acc)

    inicio = time.time()
    acc = dt_cross_validation()
    fim = time.time()
    salvar_resultados_csv('DT', 'CV', fim - inicio, acc)

    inicio = time.time()
    acc = dt_sucessive_halving()
    fim = time.time()
    salvar_resultados_csv('DT', 'SH', fim - inicio, acc)

    # KNN
    inicio = time.time()
    acc = knn_grid_search()
    fim = time.time()
    salvar_resultados_csv('KNN', 'GS', fim - inicio, acc)

    inicio = time.time()
    acc = knn_random_search()
    fim = time.time()
    salvar_resultados_csv('KNN', 'RS', fim - inicio, acc)

    inicio = time.time()
    acc = knn_bayesian_optimization()
    fim = time.time()
    salvar_resultados_csv('KNN', 'BO', fim - inicio, acc)

    inicio = time.time()
    acc = knn_cross_validation()
    fim = time.time()
    salvar_resultados_csv('KNN', 'CV', fim - inicio, acc)

    inicio = time.time()
    acc = knn_sucessive_halving()
    fim = time.time()
    salvar_resultados_csv('KNN', 'SH', fim - inicio, acc)

    # MLP
    inicio = time.time()
    acc = mlp_grid_search(numero_colunas)
    fim = time.time()
    salvar_resultados_csv('MLP', 'GS', fim - inicio, acc)

    inicio = time.time()
    acc = mlp_random_search(numero_colunas)
    fim = time.time()
    salvar_resultados_csv('MLP', 'RS', fim - inicio, acc)

    inicio = time.time()
    acc = mlp_bayesian_optimization(numero_colunas)
    fim = time.time()
    salvar_resultados_csv('MLP', 'BO', fim - inicio, acc)

    inicio = time.time()
    acc = mlp_cross_validation(numero_colunas)
    fim = time.time()
    salvar_resultados_csv('MLP', 'CV', fim - inicio, acc)

    inicio = time.time()
    acc = mlp_sucessive_halving(numero_colunas)
    fim = time.time()
    salvar_resultados_csv('MLP', 'SH', fim - inicio, acc)

    # RF
    inicio = time.time()
    acc = rf_grid_search()
    fim = time.time()
    salvar_resultados_csv('RF', 'GS', fim - inicio, acc)

    inicio = time.time()
    acc = rf_random_search()
    fim = time.time()
    salvar_resultados_csv('RF', 'RS', fim - inicio, acc)

    inicio = time.time()
    acc = rf_bayesian_optimization()
    fim = time.time()
    salvar_resultados_csv('RF', 'BO', fim - inicio, acc)

    inicio = time.time()
    acc = rf_cross_validation()
    fim = time.time()
    salvar_resultados_csv('RF', 'CV', fim - inicio, acc)

    inicio = time.time()
    acc = rf_sucessive_halving()
    fim = time.time()
    salvar_resultados_csv('RF', 'SH', fim - inicio, acc)

    # SVM
    inicio = time.time()
    acc = svm_grid_search()
    fim = time.time()
    salvar_resultados_csv('SVM', 'GS', fim - inicio, acc)

    inicio = time.time()
    acc = svm_random_search()
    fim = time.time()
    salvar_resultados_csv('SVM', 'RS', fim - inicio, acc)

    inicio = time.time()
    acc = svm_bayesian_optimization()
    fim = time.time()
    salvar_resultados_csv('SVM', 'BO', fim - inicio, acc)

    inicio = time.time()
    acc = svm_cross_validation()
    fim = time.time()
    salvar_resultados_csv('SVM', 'CV', fim - inicio, acc)

    inicio = time.time()
    acc = svm_sucessive_halving()
    fim = time.time()
    salvar_resultados_csv('SVM', 'SH', fim - inicio, acc)

import pandas as pd
import numpy as np

# def media_valores(lista_tuplas, indice):
  
#   valores = [tupla[indice] for tupla in lista_tuplas]

#   valores.remove(max(valores))
#   valores.remove(min(valores))

#   return np.mean(valores)


# estrategias = ["Grid Search", "Random Search", "Bayesian Opt", "Successive Halving", "Cross Validation"]

# tempos = [media_valores(gs_DT, 1), media_valores(rs_DT, 1), media_valores(bo_DT, 1), media_valores(sh_DT, 1), media_valores(cv_DT, 1)]
# acuracias = [media_valores(gs_DT, 0), media_valores(rs_DT, 0), media_valores(bo_DT, 0), media_valores(sh_DT, 0), media_valores(cv_DT, 0)]

# df = pd.DataFrame({
#     'Estrategia': estrategias,
#     'Tempo': tempos,
#     'Acc': acuracias
# })

# df.set_index('Estrategia', inplace=True)
# df.to_csv('./stats/DTstats.csv')

# #############################################################################################

# # KNN
# tempos = [media_valores(gs_KNN, 1), media_valores(rs_KNN, 1), media_valores(bo_KNN, 1), media_valores(sh_KNN, 1), media_valores(cv_KNN, 1)]
# acuracias = [media_valores(gs_KNN, 0), media_valores(rs_KNN, 0), media_valores(bo_KNN, 0), media_valores(sh_KNN, 0), media_valores(cv_KNN, 0)]

# df = pd.DataFrame({
#     'Estrategia': estrategias,
#     'Tempo': tempos,
#     'Acc': acuracias
# })

# df.set_index('Estrategia', inplace=True)
# df.to_csv('./stats/KNNstats.csv')

# ##########################################################################################

# # MLP
# tempos = [media_valores(gs_MLP, 1), media_valores(rs_MLP, 1), media_valores(bo_MLP, 1), media_valores(sh_MLP, 1), media_valores(cv_MLP, 1)]
# acuracias = [media_valores(gs_MLP, 0), media_valores(rs_MLP, 0), media_valores(bo_MLP, 0), media_valores(sh_MLP, 0), media_valores(cv_MLP, 0)]

# df = pd.DataFrame({
#     'Estrategia': estrategias,
#     'Tempo': tempos,
#     'Acc': acuracias
# })

# df.set_index('Estrategia', inplace=True)
# df.to_csv('./stats/MLPstats.csv')

# ##########################################################################################

# # Random Forest (RF)
# tempos = [media_valores(gs_RF, 1), media_valores(rs_RF, 1),media_valores(bo_RF, 1), media_valores(sh_RF, 1), media_valores(cv_RF, 1)]
# acuracias = [media_valores(gs_RF, 0), media_valores(rs_RF, 0),media_valores(bo_RF, 0), media_valores(sh_RF, 0), media_valores(cv_RF, 0)]

# df = pd.DataFrame({
#     'Estrategia': estrategias,
#     'Tempo': tempos,
#     'Acc': acuracias
# })

# df.set_index('Estrategia', inplace=True)
# df.to_csv('./stats/RFstats.csv')

# ##############################################################################################

# # SVM
# tempos = [media_valores(gs_SVM, 1), media_valores(rs_SVM, 1), media_valores(bo_SVM, 1), media_valores(sh_SVM, 1), media_valores(cv_SVM, 1)]
# acuracias = [media_valores(gs_SVM, 0), media_valores(rs_SVM, 0), media_valores(bo_SVM, 0), media_valores(sh_SVM, 0), media_valores(cv_SVM, 0)]

# df = pd.DataFrame({
#     'Estrategia': estrategias,
#     'Tempo': tempos,
#     'Acc': acuracias
# })

# df.set_index('Estrategia', inplace=True)
# df.to_csv('./stats/SVMstats.csv')