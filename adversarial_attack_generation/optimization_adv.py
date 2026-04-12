import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, PartialFixedSampler, NSGAIISampler, QMCSampler
import os
import re, ast
from sklearn.metrics import classification_report

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    SaliencyMapMethod,
    BasicIterativeMethod,
    DeepFool,
    CarliniL2Method,
    ElasticNet,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_best_model(data, option):
    samplers = ['TPESampler']
    f1_scores = {}
    for sampler in samplers:
        file_name = f"results/{data}/classification_report_{data}_{option}_{sampler}.csv" 
        df = pd.read_csv(file_name, index_col=0)
        f1_score = df.loc['macro avg', 'f1-score']
        f1_scores[sampler] = f1_score

    best_sampler = max(f1_scores, key=f1_scores.get)

    file_path = f"results/{data}/trial_results_{data}_{option}_{best_sampler}.csv"
    df_t = pd.read_csv(file_path)
    max_f1_idx = df_t['f1'].idxmax()
    best_model = df_t.iloc[max_f1_idx]

    return best_model


def find_best_wb(data, option, attack_type):
    file_path = f"summary_WB_{data}_{option}.txt"
    s = open(file_path, "r", encoding="utf-8").read()
    pat = re.compile(rf"Best\s+trial_?\(?\s*{re.escape(attack_type)}\s*\)?\s*:\s*(\{{.*?\}})",
                     re.IGNORECASE | re.DOTALL)
    m = pat.search(s)
    return ast.literal_eval(m.group(1))

def wb_objective(trial, X, y, classifier, attack_type, attack_batch, model):
    if attack_type == "FGSM":
        eps = trial.suggest_categorical("eps", [0.01, 0.05, 0.1, 0.2, 0.3])
        attack = FastGradientMethod(classifier, eps=eps)
    elif attack_type == "JSMA":
        gamma = trial.suggest_categorical("gamma",[0.05, 0.1, 0.15, 0.2])
        attack = SaliencyMapMethod(classifier, theta=1.0, gamma=gamma)
    elif attack_type == "PGD":
        eps = trial.suggest_categorical("eps", [0.01, 0.05, 0.1, 0.2])
        eps_step = trial.suggest_categorical("eps_step", [0.001, 0.005, 0.01, 0.02])
        max_iter = trial.suggest_categorical("max_iter", [10, 20, 40, 50, 100])
        attack = ProjectedGradientDescent(classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=max_iter)
    elif attack_type == "BIM":
        eps = trial.suggest_categorical("eps", [0.01, 0.05, 0.1, 0.2])
        eps_step = trial.suggest_categorical("eps_step", [0.001, 0.005, 0.01, 0.02])
        max_iter = trial.suggest_categorical("max_iter", [5, 10, 20, 30])
        attack = BasicIterativeMethod(classifier, eps=eps, eps_step=eps_step, max_iter=max_iter)
    elif attack_type == "DeepFool":
        max_iter = trial.suggest_categorical("max_iter", [25, 50, 100, 200])
        attack = DeepFool(classifier, max_iter=max_iter)
    elif attack_type == "C&W":
        confidence = 0.1
        max_iter = 200
        learning_rate = 0.01
        binary_search_steps = 3
        batch_size = 8
        # confidence = trial.suggest_categorical("confidence", [0.0, 0.1, 0.5, 1.0])
        # max_iter = trial.suggest_categorical("max_iter", [100, 200, 500])
        # learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.01, 0.02])
        attack = CarliniL2Method(classifier, confidence=confidence, max_iter=max_iter, 
                binary_search_steps=binary_search_steps, learning_rate=learning_rate, batch_size=batch_size)
    elif attack_type == "EAD":
        beta = 1e-3
        max_iter = 500
        binary_search_steps = 10
        learning_rate = 0.01
        # beta = trial.suggest_categorical("beta", [1e-4, 1e-3, 1e-2, 1e-1])
        # max_iter = trial.suggest_categorical("max_iter", [100, 200, 500])
        # binary_search_steps = trial.suggest_categorical("binary_search_steps", [5, 10, 20])
        # learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.05])
        attack = ElasticNet(classifier, beta=beta, max_iter=max_iter, binary_search_steps=binary_search_steps, learning_rate=learning_rate)
    else:
        print("Unknown attack")
        exit()
    
    success_list = []
    N = len(y)

    for i in range(0, N, attack_batch):
        xb = X[i:i+attack_batch]
        yb = y[i:i+attack_batch]

        x_adv = attack.generate(x=xb)
        x_adv_t = torch.from_numpy(x_adv).to(device).float()
        with torch.no_grad():
            logits_adv = model(x_adv_t)
            _, preds_adv = torch.max(logits_adv, dim=1)
            preds_adv = preds_adv.cpu().numpy()
        batch_success = (preds_adv != yb).astype(bool)
        success_list.append(batch_success)
    
    success_all = np.concatenate(success_list)
    success_rate = success_all.mean()
    print(f"Trial {trial.number} : {success_rate*100:.2f}%")
    if success_rate >= 0.95:
        print(f"Success rate >= 95% ({success_rate*100:.2f}%), stopping the study.")
        trial.study.stop()
        return success_rate
        
    return success_rate
        