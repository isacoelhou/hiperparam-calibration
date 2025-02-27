from hyperopt import fmin, tpe, hp, Trials
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from hyperopt import fmin, tpe, hp, Trials
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import optuna
from sklearn import svm
from sklearn.metrics import accuracy_score
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

import time
import os

def knn_ATPE():
    space = {
        'n_neighbors': hp.randint('n_neighbors', 1, 50),  
        'weights': hp.choice('weights', [0, 1]) 
    }

    weights_options = ['uniform', 'distance']  

    def objective(params):
        n_neighbors = params['n_neighbors']
        weights = weights_options[params['weights']] 

        KNN = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        KNN.fit(x_treino, y_treino)
        
        opiniao = KNN.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return -Acc  

    trials = Trials()

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,  
                max_evals=50,  
                trials=trials)

    best_params = {
        "n_neighbors": best["n_neighbors"],
        "weights": weights_options[best["weights"]] 
    }

    KNN = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
    
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    
    Acc = accuracy_score(y_teste, opiniao)
    save_params((best_params["n_neighbors"], best_params["weights"]), "KNN_PARAMS")
    
    return Acc

def dt_ATPE():
    space = {
        'splitter': hp.choice('splitter', ['best', 'random']),
        'criterion': hp.choice('criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 6, 8, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 4, 5, 6])
    }

    def objective(params):
        criterion = params['criterion']  
        splitter = params['splitter']  
        max_depth = params['max_depth']
        min_samples_split = params['min_samples_split']
        min_samples_leaf = params['min_samples_leaf']

        DT = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        DT.fit(x_treino, y_treino)
        opiniao = DT.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return -Acc
    
    trials = Trials()

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest, 
                max_evals=50,  
                trials=trials)

    best_params = {
        "criterion": ['gini', 'entropy'][best["criterion"]],
        "splitter": ['best', 'random'][best["splitter"]],
        "max_depth": [3, 4, 5, 6, 7][best["max_depth"]],
        "min_samples_split": [2, 5, 6, 8, 10][best["min_samples_split"]],
        "min_samples_leaf": [1, 3, 4, 5, 6][best["min_samples_leaf"]]
    }

    DT = tree.DecisionTreeClassifier(
        criterion=best_params["criterion"],
        splitter=best_params["splitter"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"]
    )
    
    DT.fit(x_treino, y_treino)
    opiniao = DT.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    save_params((best_params["criterion"], best_params["splitter"], best_params["max_depth"],
                 best_params["min_samples_split"], best_params["min_samples_leaf"]), "DT_PARAMS")
    
    return Acc

def mlp_ATPE():
    space = {
        'hidden_layer_size': hp.quniform('hidden_layer_size', numero_colunas, 2 * numero_colunas, 1),  # Tamanho da camada oculta
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),  # Taxa de aprendizado
        'max_iter': hp.choice('max_iter', [50, 100, 150, 300, 500, 1000]),  # Número máximo de iterações
        'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu'])  # Função de ativação
    }

    def objective(params):
        hidden_layer_size = int(params['hidden_layer_size']) 
        learning_rate = params['learning_rate']
        max_iter = params['max_iter']
        activation = params['activation']

        mlp = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size,),  
            learning_rate=learning_rate,
            max_iter=max_iter,
            activation=activation,
            random_state=42  
        )
        
        mlp.fit(x_treino, y_treino)
        opiniao = mlp.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return -Acc
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,  
                trials=trials)

    best_params = {
        "hidden_layer_size": int(best["hidden_layer_size"]),
        "learning_rate": ['constant', 'invscaling', 'adaptive'][best["learning_rate"]],
        "max_iter": [50, 100, 150, 300, 500, 1000][best["max_iter"]],
        "activation": ['identity', 'logistic', 'tanh', 'relu'][best["activation"]]
    }

    mlp = MLPClassifier(
        hidden_layer_sizes=(best_params["hidden_layer_size"],),
        learning_rate=best_params["learning_rate"],
        max_iter=best_params["max_iter"],
        activation=best_params["activation"],
        random_state=42
    )
    
    mlp.fit(x_treino, y_treino)
    opiniao = mlp.predict(x_teste)
    
    Acc = accuracy_score(y_teste, opiniao)

    save_params((best_params["hidden_layer_size"], best_params["learning_rate"],
                 best_params["max_iter"], best_params["activation"]), "MLP_PARAMS")
    
    return Acc

def rf_ATPE():
    space = {
        'n_estimators': hp.quniform('n_estimators', 10, 100, 1), 
        'criterion': hp.choice('criterion', ['gini', 'entropy']), 
        'max_depth': hp.quniform('max_depth', 3, 7, 1), 
        'min_samples_split': hp.quniform('min_samples_split', 5, 8, 1),  
        'min_samples_leaf': hp.quniform('min_samples_leaf', 3, 6, 1) 
    }

    def objective(params):
        n_estimators = int(params['n_estimators'])  
        criterion = params['criterion']
        max_depth = int(params['max_depth'])  
        min_samples_split = int(params['min_samples_split']) 
        min_samples_leaf = int(params['min_samples_leaf']) 

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42  
        )
        
        rf.fit(x_treino, y_treino)
        opiniao = rf.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return -Acc
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,  
                trials=trials)

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "criterion": ['gini', 'entropy'][best["criterion"]],
        "max_depth": int(best["max_depth"]),
        "min_samples_split": int(best["min_samples_split"]),
        "min_samples_leaf": int(best["min_samples_leaf"])
    }

    rf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )
    
    rf.fit(x_treino, y_treino)
    opiniao = rf.predict(x_teste)
    
    Acc = accuracy_score(y_teste, opiniao)

    save_params((best_params["n_estimators"], best_params["criterion"],
                 best_params["max_depth"], best_params["min_samples_split"],
                 best_params["min_samples_leaf"]), "RF_PARAMS")
    
    # Retorna a acurácia no conjunto de teste
    return Acc


def svm_ATPE():
    space = {
        'C': hp.choice('C', [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]),  
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    }

    def objective(params):
        C = params['C']
        kernel = params['kernel']

        svm = SVC(
            C=C,
            kernel=kernel,
            random_state=42 
        )
        
        svm.fit(x_treino, y_treino)
        opiniao = svm.predict(x_validacao)
        
        Acc = accuracy_score(y_validacao, opiniao)
        return -Acc
    
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50,  
                trials=trials)

    best_params = {
        "C": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1][best["C"]],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'][best["kernel"]]
    }

    svm = SVC(
        C=best_params["C"],
        kernel=best_params["kernel"],
        random_state=42
    )
    
    svm.fit(x_treino, y_treino)
    opiniao = svm.predict(x_teste)
    
    Acc = accuracy_score(y_teste, opiniao)
    save_params((best_params["C"], best_params["kernel"]), "SVM_PARAMS")
    
    return Acc

def save_params(params, filename):
    pasta_params = "Params"
    
    if not os.path.exists(pasta_params):
        os.makedirs(pasta_params)
    
    filename += ".txt"
    caminho_arquivo = os.path.join(pasta_params, filename)
    
    with open(caminho_arquivo, "a") as f:
        f.write(f"{params}\n")

for _ in range(10):

    dados = pd.read_csv("../datasets/Diabetes.csv")
    dados = shuffle(dados)
    
    df_dados = pd.DataFrame(dados)
    numero_colunas = df_dados.shape[1]

    df_dados = df_dados.iloc[:, :-1]
    
    Vetor_X = df_dados
    Vetor_Y = dados["Class"]
    
    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste = train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    dt_ATPE()
    mlp_ATPE()
    knn_ATPE()
    rf_ATPE()
    svm_ATPE()