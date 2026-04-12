import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, PartialFixedSampler, NSGAIISampler, QMCSampler

import os
import argparse
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
# from art.estimators.classification import PyTorchClassifier


# from optimization_adv import find_best_model, find_best_wb, wb_objective
# from adv_training import adv_attack_generation
from dnn_adv import train, evaluate
from gan_adv import compute_class_ranges, train_gan
import pdb


def main(args):
    # if args.data.lower() == 'edgeiiot':
    #     filename = 'dataset/ML-EdgeIIoT-dataset.csv'
    #     scenario_option = '1' if args.option == 'binary' else '0'
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_edgeIIoTset(filename, scenario_option)

    # elif args.data.lower() == 'toniot':
    #     filename = 'dataset/NF-ToN-IoT-v3_processed_reduced.csv'
    #     scenario_option = '1' if args.option == 'binary' else '0'
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_nftoniot(filename, scenario_option)

    # elif args.data.lower() == 'xiiotid':
    #     filename = 'dataset/pre_XIIoTID.csv'
    #     scenario_option = '1' if args.option == 'binary' else '0'
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_xiiot(filename, scenario_option)

    # elif args.data.lower() == 'wustliiot':
    #     filename = 'dataset/wustl_iiot_2021.csv'
    #     scenario_option = '1' if args.option == 'binary' else '0'
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_wustlIIoT(filename, scenario_option)

    # else:
    #     print("Wrong Dataset")
    #     exit()

    # data preprocessing - NF-ToN-IoT-v3
    filename = 'dataset/NF-ToN-IoT-v3_processed_reduced.csv'
    data = pd.read_csv(filename)
    # data = data_full.sample(frac=0.1, random_state=42)

    features = data.drop(columns=['Label', 'Attack'])
    features_np = features.to_numpy(dtype=np.float32)

    label_series = data['Attack']
    le = LabelEncoder()
    my_label_encoded = le.fit_transform(label_series)

    X_train, X_test, y_train, y_test = train_test_split(
        features_np, my_label_encoded, test_size=0.2, random_state=42, stratify=my_label_encoded
    )
    # # X_train (64%) / X_train_adv (adversarial sample) (16%)
    # X_train, X_train_adv, y_train, y_train_adv = train_test_split(
    #     X_train_total, y_train_total, test_size=0.2, random_state=42, stratify=y_train_total
    # )

    columns = features.columns.tolist()
    categorical_cols = ["PROTOCOL","L7_PROTO","TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS"]


    # _, _, columns, categorical_cols, benign_label = column_info(args.data, scenario_option)
    cat_idx = [columns.index(col) for col in categorical_cols]
    num_idx = [i for i in range(len(columns)) if i not in cat_idx]
    # X_train, X_train_adv, X_test = X_train[:, cat_idx], X_val[:, cat_idx], X_test[:, cat_idx]
    

    X_train, y_train = X_train.astype(np.float32), y_train.astype(np.int64)
    X_test, y_test = X_test.astype(np.float32), y_test.astype(np.int64)
    

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clip_min = float(np.min(X_train_scaled))
    clip_max = float(np.max(X_train_scaled))
    num_classes = len(np.unique(y_train))

    # optuna-based hyper-parameter optimization
    # best_ids_params = find_best_model(args.data, args.option)
    best_ids_params = {
        'architecture': '256_128',
        'dropout_rate': 0.24,
        'lr': 0.001,
        'batch_size': 128
    }

    # base model performance
    base_model, test_loader = train(best_ids_params, X_train_scaled, X_test_scaled, y_train, y_test)
    all_labels_base, all_preds_base = evaluate(base_model, test_loader)

    report_dict = classification_report(all_labels_base, all_preds_base, target_names=le.classes_, output_dict=True, zero_division=0)

    print(f"Original Model: accuracy={report_dict['accuracy']*100:.2f}%, precision={report_dict['macro avg']['precision']*100:.2f}%, recall={report_dict['macro avg']['recall']*100:.2f}%, f1={report_dict['macro avg']['f1-score']*100:.2f}%")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper_optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # classifier = PyTorchClassifier(
    #     model=base_model,
    #     loss=criterion,
    #     optimizer=wrapper_optimizer,
    #     input_shape=(X_train_scaled.shape[1],),
    #     nb_classes=num_classes,
    #     clip_values=(clip_min, clip_max),
    #     device_type=device
    # )
    
    # "FGSM", "JSMA", "PGD", "BIM", "DeepFool", "C&W", "EAD"

    # attack_list = ["FGSM", "JSMA", "PGD"]
    # attack_list = ["DeepFool"]
    # dict(zip(le.classes_, range(len(le.classes_))))

    attack_batch = best_ids_params['batch_size']

    for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
        order = np.bincount(all_labels_base).argsort()[::-1]
        cm = confusion_matrix(all_labels_base, all_preds_base, normalize=normalize)
        cm = cm[order][:, order]
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
        disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

        plt.title(f"Confusion Matrix - Original")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs("cm", exist_ok=True)
        plt.savefig(f"cm/original{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()


    print("\n--- Starting Adversarial Attacks ---")

    with open(f'summary_WB.txt', 'w', buffering=1) as f:
        f.write(f"Original Model: accuracy={report_dict['accuracy']*100:.2f}%, precision={report_dict['macro avg']['precision']*100:.2f}%, recall={report_dict['macro avg']['recall']*100:.2f}%, f1={report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
        f.write("=== Summary ===\n")
        
        # # X_test_normal (10%) / X_test_adv (adversarial sample) (10%)
        # X_test_normal, X_test_adv, y_test_normal, y_test_adv = train_test_split(
        #     X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        # )
        # X_test_normal, y_test_normal = X_test_normal.astype(np.float32), y_test_normal.astype(np.int64)
        # X_test_adv, y_test_adv = X_test_adv.astype(np.float32), y_test_adv.astype(np.int64)


        mask = np.array([0 if col in categorical_cols else 1 for col in columns], dtype=np.float32)
        class_ranges = compute_class_ranges(X_train_scaled, y_train)
        benign_label = 1

        targeted_trainer, untargeted_trainer, history = train_gan(
                trainloader=DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                    torch.tensor(y_train, dtype=torch.long)),
                                    batch_size=256, shuffle=True),
                target_model=base_model,
                mask=mask,
                class_ranges=class_ranges,
                benign_label=benign_label,
                dim=len(columns),
                device=device,
                eps_bound=0.6,
                n_epochs=100
        )
        
        xb = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        yb = torch.tensor(y_test, dtype=torch.long).to(device)

        idx = torch.arange(len(yb)).to(device)

        attack_mask = (yb != benign_label)
        benign_mask = (yb == benign_label)

        attack_idx, benign_idx = idx[attack_mask], idx[benign_mask]

        xb_attack, yb_attack = xb[attack_mask], yb[attack_mask]

        with torch.no_grad():
            eps_bound = 0.6
            delta_t = targeted_trainer.enforce1_mask(targeted_trainer.G(xb_attack)) * eps_bound
            x_adv_attack = targeted_trainer.enforce2_clip(xb_attack, xb_attack + delta_t, yb_attack)

        xb_benign, yb_benign = xb[benign_mask], yb[benign_mask]
        
        with torch.no_grad():
            eps_bound = 0.6
            delta_u = untargeted_trainer.enforce1_mask(untargeted_trainer.G(xb_benign)) * eps_bound
            x_adv_benign = untargeted_trainer.enforce2_clip(xb_benign, xb_benign + delta_u, yb_benign)
            
        x_adv_test, y_adv_test = torch.zeros_like(xb), torch.zeros_like(yb)
        x_adv_test[attack_idx], y_adv_test[attack_idx] = x_adv_attack, yb_attack
        x_adv_test[benign_idx], y_adv_test[benign_idx] = x_adv_benign, yb_benign

        adv_test_df = pd.DataFrame(x_adv_test.detach().cpu().numpy(), columns=columns)
        adv_test_label = le.inverse_transform(y_test)
        adv_test_df['label'] = adv_test_label
        adv_test_df.to_csv(f"adversarial_samples.csv", index=False)

        adv_test_data = TensorDataset(x_adv_test, torch.from_numpy(y_test).long())
        adv_test_loader = DataLoader(adv_test_data, batch_size=attack_batch, shuffle=False)

        all_labels_adv, all_preds_adv = evaluate(base_model, adv_test_loader)
        report_dict_adv = classification_report(all_labels_adv, all_preds_adv, target_names=le.classes_, output_dict=True, zero_division=0)
        print(report_dict_adv)

        f.write(f"Adversarial Attack : accuracy={report_dict_adv['accuracy']*100:.2f}%, precision={report_dict_adv['macro avg']['precision']*100:.2f}%, recall={report_dict_adv['macro avg']['recall']*100:.2f}%, f1={report_dict_adv['macro avg']['f1-score']*100:.2f}%\n\n")

        all_preds_adv_np = np.array(all_preds_adv)
        all_labels_adv_np = np.array(all_labels_adv)
        benign_mask = (all_labels_adv_np == benign_label)
        attack_mask = (all_labels_adv_np != benign_label)
        benign_asr = np.mean(all_preds_adv_np[benign_mask] != benign_label) * 100
        attack_asr = np.mean(all_preds_adv_np[attack_mask] == benign_label) * 100
        f.write(f"Benign → Attack ASR = {benign_asr:.2f}%\n")
        f.write(f"Attack → Benign ASR = {attack_asr:.2f}%\n\n")

        for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
            order = np.bincount(all_labels_adv).argsort()[::-1]
            cm = confusion_matrix(all_labels_adv, all_preds_adv, normalize=normalize)
            cm = cm[order][:, order]
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
            disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

            plt.title(f"Confusion Matrix - Adversarial")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"cm/adversarial{suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, required=True, help="dataset: ['EdgeIIoT', 'ToNIoT', 'XIIoTID', 'WUSTLIIoT']")
    # parser.add_argument("--option", type=str, required=True, help="classification: ['binary', 'multi']")
    # # parser.add_argument("--adv", type=str2bool, nargs="?", const=True, default=False, help="adversarial training: [True, False]")
    args = parser.parse_args()
    main(args)

        # # x_scaled_arr = scaler.inverse_transform(x_adv_test.detach().cpu().numpy())
        # adv_test_df = pd.DataFrame(x_adv_test.detach().cpu().numpy(), columns=columns)
        # adv_test_label = le.inverse_transform(y_test[:N])
        # adv_test_df['label'] = adv_test_label
        # adv_test_df.to_csv(f"adv_samples/gan/adv_test_{data}_clip.csv", index=False)
        # # pdb.set_trace()
        
        # test_data = TensorDataset(x_adv_test, torch.from_numpy(y_test[:N]).long())
        # # test_data = TensorDataset(torch.from_numpy(X_test[N:]).float(), torch.from_numpy(y_test[N:]).long())
        # test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

        # adv_test_labels, adv_test_preds, _ = evaluate(model, test_loader)
        # c_report_dict = classification_report(adv_test_labels, adv_test_preds, target_names=le.classes_, output_dict=True, zero_division=0)
        # print(c_report_dict)
        # f.write(f"Adversarial Attack : accuracy={c_report_dict['accuracy']*100:.2f}%, precision={c_report_dict['macro avg']['precision']*100:.2f}%, recall={c_report_dict['macro avg']['recall']*100:.2f}%, f1={c_report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
        
        # combined_X_test = np.concatenate((x_adv_test.detach().cpu().numpy(), X_test[N:]), axis=0)
        # combined_y_test = np.concatenate((y_test[:N], y_test[N:]), axis=0)

        # combined_test_data = TensorDataset(torch.from_numpy(combined_X_test).float(), torch.from_numpy(combined_y_test).long())
        # combined_test_loader = DataLoader(combined_test_data, batch_size=256, shuffle=False)
        
        # adv_test_labels_1, adv_test_preds_1, _ = evaluate(model, combined_test_loader)
        # c_report_dict_1 = classification_report(adv_test_labels_1, adv_test_preds_1, target_names=le.classes_, output_dict=True, zero_division=0)
        # print(c_report_dict_1)
        # f.write(f"Adversarial Attack : accuracy={c_report_dict_1['accuracy']*100:.2f}%, precision={c_report_dict_1['macro avg']['precision']*100:.2f}%, recall={c_report_dict_1['macro avg']['recall']*100:.2f}%, f1={c_report_dict_1['macro avg']['f1-score']*100:.2f}%\n\n")

        # adv_test_preds_np = np.array(adv_test_preds)
        # adv_test_labels_np = np.array(adv_test_labels)
        # benign_mask = (adv_test_labels_np == benign_label)
        # attack_mask = (adv_test_labels_np != benign_label)
        # benign_asr = np.mean(adv_test_preds_np[benign_mask] != benign_label) * 100
        # attack_asr = np.mean(adv_test_preds_np[attack_mask] == benign_label) * 100
        # f.write(f"Benign → Attack ASR = {benign_asr:.2f}%\n")
        # f.write(f"Attack → Benign ASR = {attack_asr:.2f}%\n\n")

        # # pdb.set_trace()
        
        # for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
        #     order = np.bincount(adv_test_labels).argsort()[::-1]
        #     cm = confusion_matrix(adv_test_labels, adv_test_preds, normalize=normalize)
        #     # benign_idx = np.where
        #     cm = cm[order][:, order]
        #     fig, ax = plt.subplots(figsize=(10, 8))

        #     if data.lower() == 'wustliiot':
        #         class_names = ["Normal", "DoS", "Reconnaissance", "Backdoor", "Command Injection"]
        #         display_labels = [class_names[i] for i in order]
        #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        #     else:
        #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
        #     disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

        #     plt.title(f"Confusion Matrix - GAN (After Adversarial Attack)")
        #     plt.xlabel("Predicted Label")
        #     plt.ylabel("True Label")

        #     plt.xticks(rotation=45, ha='right')
        #     plt.tight_layout()
        #     plt.savefig(f"cm/{data}_GAN_adv_attack{suffix}_clip.png", dpi=300, bbox_inches='tight')
        #     plt.close()


        # # x_adv_train = gan_adv_generation(data, option, X_train, X_val, y_train, y_val, model, wb=True, adv=True)
        # # eps_bound = 0.3
        # xt = torch.tensor(X_val, dtype=torch.float32).to(device)
        # yt = torch.tensor(y_val, dtype=torch.long).to(device)
        # idx1 = torch.arange(len(yt)).to(device)

        # attack_mask = (yt != benign_label)
        # benign_mask = (yt == benign_label)

        # attack_idx1 = idx1[attack_mask]
        # benign_idx1 = idx1[benign_mask]

        # xt_attack = xt[attack_mask]
        # yt_attack = yt[attack_mask]

        # with torch.no_grad():
        #     eps_bound = 0.5
        #     delta_t = targeted_trainer.enforce1_mask(targeted_trainer.G(xt_attack)) * eps_bound
        #     x_adv_attack_1 = targeted_trainer.enforce2_clip(
        #         xt_attack, xt_attack + delta_t, yt_attack)
        #     #     clip_to_target=True,
        #     #     # target_labels=torch.ones_like(yt_attack) * benign_label
        #     #     target_labels=yt_attack
        #     # )
        
        # xt_benign = xt[benign_mask]
        # yt_benign = yt[benign_mask]

        # with torch.no_grad():
        #     eps_bound = 0.6
        #     delta_u = untargeted_trainer.enforce1_mask(untargeted_trainer.G(xt_benign)) * eps_bound
        #     x_adv_benign_1 = untargeted_trainer.enforce2_clip(xt_benign, xt_benign + delta_u, yt_benign)
            
        # x_adv_train = torch.zeros_like(xt)
        # y_adv_train = torch.zeros_like(yt)

        # x_adv_train[attack_idx1] = x_adv_attack_1
        # y_adv_train[attack_idx1] = yt_attack

        # x_adv_train[benign_idx1] = x_adv_benign_1
        # y_adv_train[benign_idx1] = yt_benign
        
        # # pdb.set_trace()

        # # with torch.no_grad():
        # #     x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        # #     y_t = torch.tensor(y_val, dtype=torch.long).to(device)

        # #     delta_t = G(x_t)
        # #     delta_t = trainer.enforce1_mask(delta_t) * eps_bound 
        # #     x_adv_train = trainer.enforce2_clip(x_t, x_t + delta_t, y_t, clip_to_target=True, target_labels=torch.ones_like(y_t) * benign_label)
        
        # x_adv_train_np = x_adv_train.detach().cpu().numpy()
        # adv_train_df = pd.DataFrame(x_adv_train_np, columns=columns)
        # adv_train_label = le.inverse_transform(y_val)
        # adv_train_df['label'] = adv_train_label
        # adv_train_df.to_csv(f"adv_samples/gan/adv_train_{data}_clip.csv", index=False)

        # combined_X_train = np.concatenate((x_adv_train_np, X_train), axis=0)
        # combined_y_train = np.concatenate((y_val, y_train), axis=0)

        # adv_model, _ = train(best_ids_params, combined_X_train, X_test, combined_y_train, y_test)
        # adv_train_labels, adv_train_preds, _ = evaluate(adv_model, test_loader)
        # adv_report_dict = classification_report(adv_train_labels, adv_train_preds, target_names=le.classes_, output_dict=True, zero_division=0)
        # print(adv_report_dict)
        # f.write(f"Adversarial Training : accuracy={adv_report_dict['accuracy']*100:.2f}%, precision={adv_report_dict['macro avg']['precision']*100:.2f}%, recall={adv_report_dict['macro avg']['recall']*100:.2f}%, f1={adv_report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
        
        # adv_train_labels_1, adv_train_preds_1, _ = evaluate(adv_model, combined_test_loader)
        # adv_report_dict_1 = classification_report(adv_train_labels_1, adv_train_preds_1, target_names=le.classes_, output_dict=True, zero_division=0)
        # print(adv_report_dict_1)
        # f.write(f"Adversarial Training : accuracy={adv_report_dict_1['accuracy']*100:.2f}%, precision={adv_report_dict_1['macro avg']['precision']*100:.2f}%, recall={adv_report_dict_1['macro avg']['recall']*100:.2f}%, f1={adv_report_dict_1['macro avg']['f1-score']*100:.2f}%\n\n")
        # adv_train_preds_np = np.array(adv_train_preds)
        # adv_train_labels_np = np.array(adv_train_labels)
        # benign_mask = (adv_train_labels_np == benign_label)
        # attack_mask = (adv_train_labels_np != benign_label)
        # benign_asr = np.mean(adv_train_preds_np[benign_mask] != benign_label) * 100
        # attack_asr = np.mean(adv_train_preds_np[attack_mask] == benign_label) * 100
        # f.write(f"Benign → Attack ASR = {benign_asr:.2f}%\n")
        # f.write(f"Attack → Benign ASR = {attack_asr:.2f}%\n\n")

        # for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
        #     order = np.bincount(adv_train_labels).argsort()[::-1]
        #     cm = confusion_matrix(adv_train_labels, adv_train_preds, normalize=normalize)
        #     cm = cm[order][:, order]
        #     fig, ax = plt.subplots(figsize=(10, 8))

        #     if data.lower() == 'wustliiot':
        #         class_names = ["Normal", "DoS", "Reconnaissance", "Backdoor", "Command Injection"]
        #         display_labels = [class_names[i] for i in order]
        #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        #     else:
        #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
        #     disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

        #     plt.title(f"Confusion Matrix - GAN (After Adversarial Training)")
        #     plt.xlabel("Predicted Label")
        #     plt.ylabel("True Label")

        #     plt.xticks(rotation=45, ha='right')
        #     plt.tight_layout()
        #     plt.savefig(f"cm/{data}_GAN_adv_training{suffix}_clip.png", dpi=300, bbox_inches='tight')
        #     plt.close()


        # for attack_type in attack_list:
            # study = optuna.create_study(direction="maximize", sampler=TPESampler())
            # study.optimize(lambda trial: wb_objective(trial, X_test[:N], y_test[:N], classifier, attack_type, attack_batch, model), n_trials=20)
            # print(f"Best trial_({attack_type}):", study.best_trial.params)
            # line = f"Best trial_({attack_type}): {study.best_trial.params}\n"
            # f.write(f"Attack Type: {attack_type}\n")
            
            # best_params = study.best_trial.params
            # best_params = find_best_wb(args.data, args.option, attack_type)
            # adv_test_list = adv_attack_generation(attack_type, args.data, args.option, attack_batch, classifier, best_params, X_test[:N], y_test[:N], N, device)
            # adv_test = np.concatenate(adv_test_list, axis=0)

            # X_test_re = np.zeros_like(X_test[:N])
            # X_test_re[:, cat_idx] = X_test_cat[:N]
            # X_test_re[:, num_idx] = adv_test[:, num_idx]
            # # pdb.set_trace()
            # test_data = TensorDataset(torch.from_numpy(X_test_re).float(), torch.from_numpy(y_test[:N]).long())

            # # test_data = TensorDataset(torch.from_numpy(adv_test).float(), torch.from_numpy(y_test[:N]).long())
            # test_loader = DataLoader(test_data, batch_size=int(best_ids_params['batch_size']), shuffle=False)

            # adv_labels, adv_preds, _ = evaluate(model, test_loader)
            
            # c_report_dict = classification_report(adv_labels, adv_preds, target_names=le.classes_, output_dict=True, zero_division=0)
            # print(c_report_dict)
            # pdb.set_trace()

            # combined_X_test = np.concatenate((adv_test, X_test[N:]), axis=0)
            # combined_y_test = np.concatenate((y_test[:N], y_test[N:]), axis=0)

            # combined_test_data = TensorDataset(torch.from_numpy(combined_X_test).float(), torch.from_numpy(combined_y_test).long())
            # combined_test_loader = DataLoader(combined_test_data, batch_size=int(best_ids_params['batch_size']), shuffle=False)

            # combined_test_labels, combined_test_preds, _ = evaluate(model, combined_test_loader)

            # c1_report_dict = classification_report(combined_test_labels, combined_test_preds, target_names=le.classes_, output_dict=True, zero_division=0)
            # print(c1_report_dict)
            # # pdb.set_trace()
            # report1 = f"After Adversarial Attack_full(adversarial training X) {len(X_train)}/{len(combined_X_test)}: accuracy={c1_report_dict['accuracy']*100:.2f}%, precision={c1_report_dict['macro avg']['precision']*100:.2f}%, recall={c1_report_dict['macro avg']['recall']*100:.2f}%, f1={c1_report_dict['macro avg']['f1-score']*100:.2f}%\n"
            # print(report1)
            # f.write(report1)
            
            # for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
            #     order = np.bincount(adv_labels).argsort()[::-1]
            #     cm = confusion_matrix(adv_labels, adv_preds, normalize=normalize)
            #     cm = cm[order][:, order]
            #     fig, ax = plt.subplots(figsize=(10, 8))
            #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
            #     disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

            #     plt.title(f"Confusion Matrix - {attack_type} (After Adversarial Attack)")
            #     plt.xlabel("Predicted Label")
            #     plt.ylabel("True Label")

            #     plt.xticks(rotation=45, ha='right')
            #     plt.tight_layout()
            #     plt.savefig(f"cm/{args.data}_{attack_type}_adv_attack{suffix}_recheck.png", dpi=300, bbox_inches='tight')
            #     plt.close()
            
            

            # report = f"After Adversarial Attack(adversarial training X){len(X_train)}/{len(adv_test)}: accuracy={c_report_dict['accuracy']*100:.2f}%, precision={c_report_dict['macro avg']['precision']*100:.2f}%, recall={c_report_dict['macro avg']['recall']*100:.2f}%, f1={c_report_dict['macro avg']['f1-score']*100:.2f}%\n"
            # print(report)
            # f.write(report)
            # # df_report = pd.DataFrame(c_report_dict).transpose()
            # # df_report.to_csv(f"WB/classification_report_{attack_type}_{N}.csv")
            # # pdb.set_trace()
            # if args.adv == True:
            #     print("True")
            #     adv_train_list = adv_attack_generation(attack_type, args.data, args.option, attack_batch, classifier, best_params, X_val, y_val, len(y_val), device)
            #     adv_train = np.concatenate(adv_train_list, axis=0)

            #     X_val_re = np.zeros_like(X_val)
            #     X_val_re[:, cat_idx] = X_val_cat
            #     X_val_re[:, num_idx] = adv_train[:, num_idx]
            #     combined_X_train = np.concatenate((X_val_re, X_train), axis=0)
            #     # combined_X_train = np.concatenate((adv_train, X_train), axis=0)
            #     combined_y_train = np.concatenate((y_val, y_train), axis=0)

            #     # pdb.set_trace()
            #     adv_model, _ = train(best_ids_params, combined_X_train, adv_test, combined_y_train, y_test[:N])
            #     adv_test_labels, adv_test_preds, _ = evaluate(adv_model, test_loader)
                
            #     adv_report_dict = classification_report(adv_test_labels, adv_test_preds, target_names=le.classes_, output_dict=True, zero_division=0)
            #     print(adv_report_dict)
                
            #     # test_dataset = TensorDataset(torch.tensor(combined_X_test, dtype=torch.float32), torch.tensor(combined_y_test))
            #     # test_loader = DataLoader(test_dataset, batch_size=int(best_params["batch_size"]), shuffle=False)
                
            #     # adv_test_labels1, adv_test_preds1, _ = evaluate(adv_model, test_loader)
                
            #     # adv_report_dict1 = classification_report(adv_test_labels1, adv_test_preds1, target_names=le.classes_, output_dict=True, zero_division=0)
            #     # print(adv_report_dict1)
            #     # adv_report1 = f"After Adversarial Attack_full(adversarial training O){len(combined_X_train)}/{len(combined_X_test)}: accuracy={adv_report_dict1['accuracy']*100:.2f}%, precision={adv_report_dict1['macro avg']['precision']*100:.2f}%, recall={adv_report_dict1['macro avg']['recall']*100:.2f}%, f1={adv_report_dict1['macro avg']['f1-score']*100:.2f}%\n"
            #     # print(adv_report1)
            #     # f.write(adv_report1)
            #     for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
            #         order = np.bincount(adv_test_labels).argsort()[::-1]
            #         cm = confusion_matrix(adv_test_labels, adv_test_preds, normalize=normalize)
            #         cm = cm[order][:, order]
            #         fig, ax = plt.subplots(figsize=(10, 8))
            #         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
            #         disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

            #         plt.title(f"Confusion Matrix - {attack_type} (After Adversarial Training)")
            #         plt.xlabel("Predicted Label")
            #         plt.ylabel("True Label")
                    
            #         plt.xticks(rotation=45, ha='right')
            #         plt.tight_layout()
            #         plt.savefig(f"cm/{args.data}_{attack_type}_adv_training{suffix}_recheck.png", dpi=300, bbox_inches='tight')
            #         plt.close()
            #     # cm3 = confusion_matrix(adv_test_labels, adv_test_preds)
            #     # disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=le.classes_)
            #     # disp.plot(cmap='Blues', values_format='d')

            #     # plt.title(f"Confusion Matrix - {attack_type} (After Adversarial Training)")
            #     # plt.xlabel("Predicted Label")
            #     # plt.ylabel("True Label")
            #     # plt.savefig(f"cm_{attack_type}_adv_training.png", dpi=300, bbox_inches='tight')
            #     adv_report = f"After Adversarial Attack(adversarial training O){len(combined_X_train)}/{len(adv_test)}: accuracy={adv_report_dict['accuracy']*100:.2f}%, precision={adv_report_dict['macro avg']['precision']*100:.2f}%, recall={adv_report_dict['macro avg']['recall']*100:.2f}%, f1={adv_report_dict['macro avg']['f1-score']*100:.2f}%\n"
            #     print(adv_report)
            #     f.write(adv_report)

                # df_adv_report = pd.DataFrame(adv_report_dict).transpose()
                # df_adv_report.to_csv(f"WB/classification_report_adv_{attack_type}_{N}.csv")
            # else:
            #     print("False")
            #     pass
