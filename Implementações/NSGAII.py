import optuna
import os
import pandas as pd

from sklearn import svm
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def knn_nsga2():
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        
        KNN = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        KNN.fit(x_treino, y_treino)
        
        opiniao = KNN.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return Acc
    
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.NSGAIISampler())
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    
    KNN = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
    KNN.fit(x_treino, y_treino)
    opiniao = KNN.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    save_params((best_params["n_neighbors"], best_params["weights"]), "KNN_PARAMS")
    
    return Acc

def mlp_nsga2():
    def objective(trial):
        hidden_size = trial.suggest_categorical("hidden_size", [numero_colunas, 2 * numero_colunas])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        max_iter = trial.suggest_categorical("max_iter", [50, 100, 150, 300, 500, 1000])
        activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
        
        MLP = MLPClassifier(hidden_layer_sizes=(hidden_size, hidden_size, hidden_size), 
                            learning_rate=learning_rate, 
                            max_iter=max_iter, 
                            activation=activation)
        MLP.fit(x_treino, y_treino)
        
        opiniao = MLP.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return Acc
    
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.NSGAIISampler())
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    
    MLP = MLPClassifier(hidden_layer_sizes=(best_params["hidden_size"], best_params["hidden_size"], best_params["hidden_size"]), 
                        learning_rate=best_params["learning_rate"], 
                        max_iter=best_params["max_iter"], 
                        activation=best_params["activation"])
    MLP.fit(x_treino, y_treino)
    opiniao = MLP.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["hidden_size"], best_params["learning_rate"], 
      best_params["max_iter"], best_params["activation"]), "MLP_PARAMS")

    
    return Acc

def rf_nsga2():
    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [10, 20, 30, 50, 75, 100])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7])
        min_samples_split = trial.suggest_categorical("min_samples_split", [5, 6, 8, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", [3, 4, 5, 6])
        
        RF = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                    max_depth=max_depth, min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
        RF.fit(x_treino, y_treino)
        
        opiniao = RF.predict(x_validacao)
        Acc = accuracy_score(y_validacao, opiniao)
        
        return Acc
    
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.NSGAIISampler())
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    
    # Treinar e avaliar no conjunto de teste com os melhores hiperparâmetros
    RF = RandomForestClassifier(n_estimators=best_params["n_estimators"], criterion=best_params["criterion"],
                                max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"],
                                min_samples_leaf=best_params["min_samples_leaf"])
    RF.fit(x_treino, y_treino)
    opiniao = RF.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)

    save_params((best_params["n_estimators"], best_params["criterion"],
                 best_params["max_depth"], best_params["min_samples_split"],
                 best_params["min_samples_leaf"]), "RF_PARAMS")
    return Acc

def svm_nsga2():

    def objective(trial):
        C = trial.suggest_float("C", 0.1, 1.0, step=0.1)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        
        SVM_model = svm.SVC(C=C, kernel=kernel)
        SVM_model.fit(Vetor_X, Vetor_Y)
        opiniao = SVM_model.predict(x_teste)
        Acc = accuracy_score(y_teste, opiniao)
        
        return Acc
    
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.NSGAIISampler())
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    
    SVM_best = svm.SVC(C=best_params["C"], kernel=best_params["kernel"])
    SVM_best.fit(Vetor_X, Vetor_Y)
    opiniao = SVM_best.predict(x_teste)
    Acc = accuracy_score(y_teste, opiniao)
    
    save_params((best_params["C"], best_params["kernel"]), "SVM_PARAMS")
    
    return Acc

def DT_nsga2():              
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
    
    study = optuna.create_study(directions=["maximize"], sampler=optuna.samplers.NSGAIISampler())
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

    knn_nsga2()
    rf_nsga2()
    DT_nsga2()
    mlp_nsga2()
    svm_nsga2()