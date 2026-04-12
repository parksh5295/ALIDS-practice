import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    SaliencyMapMethod,
    BasicIterativeMethod,
    DeepFool,
    CarliniL2Method,
    ElasticNet,
)
import pdb
import inspect


def adv_attack_generation(attack_type, data, option, attack_batch, classifier, best_params, X, y, N, device):
    if attack_type == "FGSM":
        attack = FastGradientMethod(classifier, eps=best_params["eps"])
    
    elif attack_type == "JSMA":
        attack = SaliencyMapMethod(classifier, theta=1.0, gamma=best_params["gamma"])
    
    elif attack_type == "PGD":
        attack = ProjectedGradientDescent(classifier, 
                    norm=np.inf, eps=best_params["eps"], 
                    eps_step=best_params["eps_step"], max_iter=best_params["max_iter"])
    
    elif attack_type == "BIM":
        attack = BasicIterativeMethod(classifier, 
                    eps=best_params["eps"], eps_step=best_params["eps_step"],
                    max_iter=best_params["max_iter"])
    
    elif attack_type == "DeepFool":
        attack = DeepFool(classifier, max_iter=best_params["max_iter"])
    
    elif attack_type == "C&W":
        attack = CarliniL2Method(classifier, 
                    confidence=best_params["confidence"], max_iter=best_params["max_iter"], 
                    binary_search_steps=best_params["binary_search_steps"],
                    learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"])
    
    elif attack_type == "EAD":
        attack = ElasticNet(classifier, 
                    beta=best_params["beta"], max_iter=best_params["max_iter"], 
                    binary_search_steps=best_params["binary_search_steps"], 
                    learning_rate=best_params["learning_rate"])

    print(f"\n>>> Attack: {attack_type}")
    print(f"generate {N} samples")
    adv_list = []
    # cnt = 0
    for i in range(0, N, attack_batch):
        xb = X[i:i+attack_batch]
        yb = y[i:i+attack_batch]
        
        if option == 'binary':
            x_adv = attack.generate(x=xb)
        
        elif option == 'multi':
            if data.lower() == 'edgeiiot':
                # benign_label = 7
                benign_label = 4
                idx_benign = np.where(yb == benign_label)[0]
                idx_attack = np.where(yb != benign_label)[0]
            elif data.lower() == 'toniot':
                benign_label = 1
                idx_benign = np.where(yb == benign_label)[0]
                idx_attack = np.where(yb != benign_label)[0]
            elif data.lower() == 'xiiotid':
                benign_label = 5
                idx_benign = np.where(yb == benign_label)[0]
                idx_attack = np.where(yb != benign_label)[0]
            elif data.lower() == 'wustliiot':
                benign_label = 0
                idx_benign = np.where(yb == benign_label)[0]
                idx_attack = np.where(yb != benign_label)[0]

            x_adv = np.zeros_like(xb)
            if attack_type not in ("JSMA", "DeepFool"):
                attack.targeted = False
                if attack_type == "PGD" or attack_type == "BIM":
                    attack._attack.targeted = False
            xb_benign = xb[idx_benign]
            x_adv_benign = attack.generate(x=xb_benign)
            # pdb.set_trace()
            # print(f"{attack.targeted} benign")
            x_adv[idx_benign] = x_adv_benign

            xb_attack = xb[idx_attack]

            if attack_type in ("JSMA", "DeepFool"):
                yb_target = np.eye(len(np.unique(y)))[[benign_label] * xb_attack.shape[0]]
                x_adv_attack = attack.generate(x=xb_attack, y=yb_target)
            else:
                attack.targeted = True
                if attack_type == "PGD" or attack_type == "BIM":
                    attack._attack.targeted = True
                
                yb_target = np.full((xb_attack.shape[0],), benign_label, dtype=np.int64)
                x_adv_attack = attack.generate(x=xb_attack, y=yb_target, targeted=True)
                # print(f"{attack.targeted} attack")
            x_adv[idx_attack] = x_adv_attack
            # pdb.set_trace()
            
        else:
            exit()
        
        x_adv_t = torch.from_numpy(x_adv).to(device).float()
        adv_list.append(x_adv)
    
    return adv_list

