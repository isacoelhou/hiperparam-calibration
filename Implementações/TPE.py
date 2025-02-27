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

def SVM_tpe_optimization():
    def objective(trial):
        C = trial.suggest_float("C", 0.1, 1.0, step=0.1)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        
        SVM_model = svm.SVC(C=C, kernel=kernel)
        SVM_model.fit(Vetor_X, Vetor_Y)
        opiniao = SVM_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    SVM_best = svm.SVC(C=best_params["C"], kernel=best_params["kernel"])
    SVM_best.fit(Vetor_X, Vetor_Y)
    opiniao = SVM_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["C"], best_params["kernel"]), "SVM_PARAMS")
    
    return Acc

def DT_tpe_optimization():              
    def objective(trial):
        splitter = trial.suggest_categorical("splitter", ["best", "random"])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 3, 7, step=1)
        min_samples_split = trial.suggest_categorical("min_samples_split", [5,6,8,10])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 6, step=1)

        DT_model = tree.DecisionTreeClassifier(splitter=splitter, criterion=criterion, 
                                          max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                          min_samples_split=min_samples_split)
        DT_model.fit(Vetor_X, Vetor_Y)
        opiniao = DT_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    DT_best = tree.DecisionTreeClassifier(splitter=best_params["splitter"], criterion=best_params["criterion"], 
                                     max_depth=best_params["max_depth"], min_samples_leaf=best_params["min_samples_leaf"],
                                     min_samples_split=best_params["min_samples_split"])
    DT_best.fit(Vetor_X, Vetor_Y)
    opiniao = DT_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["splitter"], best_params["criterion"], 
                 best_params["max_depth"], best_params["min_samples_leaf"],
                 best_params["min_samples_split"]), "DT_PARAMS")
    
    return Acc

def KNN_tpe_optimization():              
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])

        KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        KNN_model.fit(Vetor_X, Vetor_Y)
        opiniao = KNN_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    KNN_best = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
    KNN_best.fit(Vetor_X, Vetor_Y)
    opiniao = KNN_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["n_neighbors"], best_params["weights"]), "KNN_PARAMS")
    
    return Acc
    
def MLP_tpe_optimization():              
    def objective(trial):
        hidden_layer_size = trial.suggest_int("hidden_layer_size", numero_colunas, 2 * numero_colunas)
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        max_iter = trial.suggest_categorical("max_iter", [50, 100, 150, 300, 500, 1000])
        activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])

        MLP_model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size, hidden_layer_size, hidden_layer_size), 
                                  learning_rate=learning_rate, max_iter=max_iter, activation=activation)
        MLP_model.fit(x_treino, y_treino)
        opiniao = MLP_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    MLP_best = MLPClassifier(hidden_layer_sizes=(best_params["hidden_layer_size"], best_params["hidden_layer_size"], best_params["hidden_layer_size"]),
                              learning_rate=best_params["learning_rate"], max_iter=best_params["max_iter"],
                              activation=best_params["activation"])
    MLP_best.fit(x_treino, y_treino)
    opiniao = MLP_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["hidden_layer_size"], best_params["learning_rate"], 
                 best_params["max_iter"], best_params["activation"]), "MLP_PARAMS")
    
    return Acc

def RF_tpe_optimization():              
    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [10, 20, 30, 50, 75, 100])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 3, 7, step=1)
        min_samples_split = trial.suggest_categorical("min_samples_split", [5, 6, 8, 10])
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 3, 6, step=1)

        RF_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, 
                                          max_depth=max_depth, min_samples_split=min_samples_split, 
                                          min_samples_leaf=min_samples_leaf)
        RF_model.fit(x_treino, y_treino)
        opiniao = RF_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    RF_best = RandomForestClassifier(n_estimators=best_params["n_estimators"], criterion=best_params["criterion"],
                                     max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"],
                                     min_samples_leaf=best_params["min_samples_leaf"])
    RF_best.fit(x_treino, y_treino)
    opiniao = RF_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["n_estimators"], best_params["criterion"],
                 best_params["max_depth"], best_params["min_samples_split"],
                 best_params["min_samples_leaf"]), "RF_PARAMS")
    
    return Acc

def save_params(params, filename):
    pasta_params = "Params"
    
    if not os.path.exists(pasta_params):
        os.makedirs(pasta_params)
    
    filename += ".txt"
    caminho_arquivo = os.path.join(pasta_params, filename)
    
    with open(caminho_arquivo, "a") as f:
        f.write(f"{params}\n")

for _ in range(1):

    dados = pd.read_csv("../datasets/Diabetes.csv")
    dados = shuffle(dados)
    
    df_dados = pd.DataFrame(dados)
    numero_colunas = df_dados.shape[1]

    df_dados = df_dados.iloc[:, :-1]
    
    Vetor_X = df_dados
    Vetor_Y = dados["Class"]
    
    x_treino,x_temp,y_treino,y_temp = train_test_split(df_dados,dados["Class"],test_size=0.5,stratify=dados["Class"])
    x_validacao,x_teste,y_validacao,y_teste = train_test_split(x_temp,y_temp,test_size=0.5, stratify = y_temp)

    inicio = time.time()
    acc = RF_tpe_optimization()
    fim = time.time()

    RF_tpe_optimization()
    MLP_tpe_optimization()
    KNN_tpe_optimization()
    SVM_tpe_optimization()
    DT_tpe_optimization()