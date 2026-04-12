import os
import math
import copy
import random
from typing import Optional, Dict, Tuple
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from preprocess import preprocess_wustlIIoT, preprocess_xiiot, preprocess_edgeIIoTset, preprocess_nftoniot
# from wb_optimize import find_best_model, wb_objective
# from dnn import DNN, train, evaluate
import pdb

def compute_class_ranges(X, y):
    classes = np.unique(y)
    ranges = {}
    for c in classes:
        Xc = X[y==c]
        mins = Xc.min(axis=0)
        maxs = Xc.max(axis=0)
        ranges[int(c)] = (mins, maxs)
    return ranges

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, dim, hidden=512, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Tanh() 
        )
        self.apply(weights_init)

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.size(0), self.z_dim, device=x.device)
        inp = torch.cat([x, z], dim=1)
        return self.net(inp)

class Critic(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1)
        )
        self.apply(weights_init)
    def forward(self, x):
        return self.net(x).view(-1)

class SurrogateClassifier(nn.Module):
    def __init__(self, dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
        self.apply(weights_init)
    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake, device="cpu", lambda_gp=10.0):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, device=device)
    eps = eps.expand_as(real)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)
    scores = critic(x_hat)
    grads = torch.autograd.grad(outputs=scores, inputs=x_hat,
                                grad_outputs=torch.ones_like(scores),
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp

def cw_margin_loss(logits, targets, targeted=True, kappa=0.0):
    one_hot = F.one_hot(targets, num_classes=logits.size(1)).float().to(logits.device)
    target_logit = torch.sum(one_hot * logits, dim=1)
    other_mask = (1 - one_hot)
    masked_logits = logits * other_mask + (one_hot * (-1e4))

    other_logit = torch.max(masked_logits, dim=1)[0]
    if targeted:
        loss = torch.clamp(other_logit - target_logit + kappa, min=0)
    else:
        loss = torch.clamp(target_logit - other_logit + kappa, min=0)
    
    return loss.mean()

class GANTrainerBase:
    def __init__(self, dim, mask, class_ranges, benign_label, device):
        self.dim = dim
        self.mask = torch.tensor(mask.astype(np.float32)).to(device)
        self.class_ranges = {int(k): (torch.tensor(v[0], dtype=torch.float32),
                                       torch.tensor(v[1], dtype=torch.float32))
                             for k, v in class_ranges.items()}
        self.benign_label = int(benign_label)
        self.device = device

        self.G = Generator(dim).to(device)
        self.C = Critic(dim).to(device)
    
    def enforce1_mask(self, delta):
        return delta * self.mask

    def enforce2_clip(self, x, x_star, labels, clip_to_target=False, target_labels=None):
        mins = []
        maxs = []

        clip_labels = target_labels if clip_to_target and target_labels is not None else labels

        for label in clip_labels.cpu().numpy():
            if label in self.class_ranges:
                mn, mx = self.class_ranges[int(label)]
            else:
                mn = torch.zeros_like(x[0])
                mx = torch.ones_like(x[0])
            mins.append(mn.unsqueeze(0))
            maxs.append(mx.unsqueeze(0))
        mins = torch.cat(mins).to(self.device)
        maxs = torch.cat(maxs).to(self.device)

        return torch.max(torch.min(x_star, maxs), mins)

    def train_critic(self, bx, by, eps_bound=0.5, critic_iters=5, lambda_gp=10.0, opt_C=None):
        for _ in range(critic_iters):
            with torch.no_grad():
                delta = self.G(bx)
                delta = self.enforce1_mask(delta) * eps_bound
                x_star = self.enforce2_clip(bx, bx + delta, by)
                                            # , clip_to_target=True,
                                            # # target_labels=torch.ones_like(by) * self.benign_label)
                                            # target_labels=by)
            real_scores = self.C(bx)
            fake_scores = self.C(x_star)
            c_loss = -(real_scores.mean() - fake_scores.mean())
            gp = gradient_penalty(self.C, bx, x_star, device=self.device, lambda_gp=lambda_gp)
            total_c_loss = c_loss + gp

            opt_C.zero_grad()
            total_c_loss.backward()
            opt_C.step()
        return total_c_loss


class TargetedGANTrainer(GANTrainerBase):
    def compute_adv_loss(self, logits, by):
        attack_mask = (by != self.benign_label)
        if attack_mask.any():
            targeted_logits = logits[attack_mask]
            targeted_targets = torch.full_like(by[attack_mask], self.benign_label)
            return cw_margin_loss(targeted_logits, targeted_targets, targeted=True, kappa=0.5)
        else:
            return torch.tensor(0.0, device=self.device)

class UntargetedGANTrainer(GANTrainerBase):
    def compute_adv_loss(self, logits, by):
        benign_mask = (by == self.benign_label)
        if benign_mask.any():
            untargeted_logits = logits[benign_mask]
            untargeted_targets = by[benign_mask]
            return cw_margin_loss(untargeted_logits, untargeted_targets, targeted=False, kappa=0.5)
        else:
            return torch.tensor(0.0, device=self.device)

def train_gan(trainloader, target_model, mask, class_ranges, benign_label, dim, device,
                                  eps_bound=0.5, critic_iters=5, lambda_adv=5.0, alpha=0.05, beta=0.001,
                                  lambda_gp=10.0, n_epochs=50):

    targeted_trainer = TargetedGANTrainer(dim, mask, class_ranges, benign_label, device)
    untargeted_trainer = UntargetedGANTrainer(dim, mask, class_ranges, benign_label, device)

    opt_G_targeted = torch.optim.Adam(targeted_trainer.G.parameters(), lr=1e-4, betas=(0.5,0.9))
    opt_C_targeted = torch.optim.Adam(targeted_trainer.C.parameters(), lr=1e-4, betas=(0.5,0.9))
    opt_G_untargeted = torch.optim.Adam(untargeted_trainer.G.parameters(), lr=1e-4, betas=(0.5,0.9))
    opt_C_untargeted = torch.optim.Adam(untargeted_trainer.C.parameters(), lr=1e-4, betas=(0.5,0.9))

    target_model.to(device).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    history = {"gen_loss_targeted": [], "gen_loss_untargeted": [], "asr_a": [], "asr_b": []}
    for epoch in range(n_epochs):
        epoch_gen_loss_targeted, epoch_gen_loss_untargeted = [], []
        epoch_asr_a, epoch_asr_b = [], []
        for bx, by in trainloader:
            bx, by = bx.to(device), by.to(device)

            attack_mask = (by != benign_label)
            if attack_mask.sum() > 0:
                bx_att = bx[attack_mask]
                by_att = by[attack_mask]

                targeted_trainer.train_critic(bx_att, by_att, eps_bound, critic_iters, lambda_gp, opt_C_targeted)

                delta_t = targeted_trainer.enforce1_mask(targeted_trainer.G(bx_att)) * eps_bound
                x_star_t = targeted_trainer.enforce2_clip(bx_att, bx_att + delta_t, by_att)
                                                        #   clip_to_target=True,
                                                        #   target_labels=torch.ones_like(by_att) * benign_label)
                                                        # target_labels=by_att)
                logits_t = target_model(x_star_t)
                logits_t = torch.clamp(logits_t, -10, 10)

                adv_loss_t = targeted_trainer.compute_adv_loss(logits_t, by_att)
                wgan_term_t = - targeted_trainer.C(x_star_t).mean()
                delta_norm_t = delta_t.view(delta_t.size(0), -1).norm(2, dim=1)
                pert_loss_t = (F.relu(delta_norm_t - eps_bound) ** 2).mean()
                gen_loss_t = lambda_adv * adv_loss_t + alpha * wgan_term_t + beta * pert_loss_t

                opt_G_targeted.zero_grad(); gen_loss_t.backward(); opt_G_targeted.step()
                epoch_gen_loss_targeted.append(gen_loss_t.item())

                with torch.no_grad():
                    preds_t = target_model(x_star_t).argmax(dim=1)
                    attack_asr_batch = (preds_t == benign_label).float().mean().item()
                    epoch_asr_a.append(attack_asr_batch * 100)
            else:
                pass

            benign_mask = (by == benign_label)
            if benign_mask.sum() > 0:
                bx_ben = bx[benign_mask]
                by_ben = by[benign_mask]
                eps_bound = 0.6
                untargeted_trainer.train_critic(bx_ben, by_ben, eps_bound, critic_iters, lambda_gp, opt_C_untargeted)

                delta_u = untargeted_trainer.enforce1_mask(untargeted_trainer.G(bx_ben)) * eps_bound
                x_star_u = untargeted_trainer.enforce2_clip(bx_ben, bx_ben + delta_u, by_ben,
                                                            clip_to_target=False)
                logits_u = target_model(x_star_u)
                logits_u = torch.clamp(logits_u, -10, 10)

                adv_loss_u = untargeted_trainer.compute_adv_loss(logits_u, by_ben)
                wgan_term_u = - untargeted_trainer.C(x_star_u).mean()
                delta_norm_u = delta_u.view(delta_u.size(0), -1).norm(2, dim=1)
                pert_loss_u = (F.relu(delta_norm_u - eps_bound) ** 2).mean()
                gen_loss_u = lambda_adv * adv_loss_u + alpha * wgan_term_u + beta * pert_loss_u

                opt_G_untargeted.zero_grad(); gen_loss_u.backward(); opt_G_untargeted.step()
                epoch_gen_loss_untargeted.append(gen_loss_u.item())

                with torch.no_grad():
                    preds_u = target_model(x_star_u).argmax(dim=1)
                    benign_asr_batch = (preds_u != benign_label).float().mean().item()
                    epoch_asr_b.append(benign_asr_batch * 100)
            else:
                pass

        avg_t = np.mean(epoch_gen_loss_targeted) if len(epoch_gen_loss_targeted)>0 else 0.0
        avg_u = np.mean(epoch_gen_loss_untargeted) if len(epoch_gen_loss_untargeted)>0 else 0.0
        avg_asr_a = np.mean(epoch_asr_a) if len(epoch_asr_a)>0 else 0.0
        avg_asr_b = np.mean(epoch_asr_b) if len(epoch_asr_b)>0 else 0.0

        history["gen_loss_targeted"].append(avg_t)
        history["gen_loss_untargeted"].append(avg_u)
        history["asr_a"].append(avg_asr_a)
        history["asr_b"].append(avg_asr_b)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | GenLoss_T: {avg_t:.4f} | GenLoss_U: {avg_u:.4f} | ASR_Attack: {avg_asr_a:.2f}% | ASR_Benign: {avg_asr_b:.2f}%")

    return targeted_trainer, untargeted_trainer, history



# if __name__ == "__main__":

#     torch.manual_seed(0)
#     np.random.seed(0)
#     random.seed(0)
#     # data = "ToNIoT"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     scenario_option = '0'
#     option = 'multi'
#     #'EdgeIIoT', 'ToNIoT',
#     data_list = ['EdgeIIoT', 'ToNIoT', 'WUSTLIIoT', 'XIIoTID']
#     for data in data_list:
#         # if data != 'EdgeIIoT':
#         #     exit()
        
#         features, my_label, columns, cat_cols, benign_label = column_info(data, scenario_option)

#         print(data)
#         print(cat_cols)
#         # continue

#         scaler = MinMaxScaler()
#         le = LabelEncoder()
#         my_label_encoded = le.fit_transform(my_label)
#         X_train_f, X_test, y_train_f, y_test = train_test_split(
#             features, my_label_encoded, test_size=0.3, random_state=42, stratify=my_label_encoded
#         )
#         X_train, X_val, y_train, y_val = train_test_split(X_train_f, y_train_f, test_size=0.2, random_state=42, stratify=y_train_f)

#         X_train = scaler.fit_transform(X_train)
#         X_val = scaler.transform(X_val)
#         X_test = scaler.transform(X_test)

#         best_ids_params = find_best_model(data, option)
#         final_model, testloader = train(best_ids_params, X_train, X_test, y_train, y_test)
#         # final_model.to(device).eval()
#         # for p in final_model.parameters():
#         #     p.requires_grad = False
        
#         all_labels, all_preds, model = evaluate(final_model, testloader)
#         report_dict = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True, zero_division=0)
        
#         with open(f'summary_WB_{data}_gan_clip.txt', 'w', buffering=1) as f:
#             f.write(f"Original Model ({data}) : accuracy={report_dict['accuracy']*100:.2f}%, precision={report_dict['macro avg']['precision']*100:.2f}%, recall={report_dict['macro avg']['recall']*100:.2f}%, f1={report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
#             f.write("=== Summary ===\n")
            
#             class_ranges = compute_class_ranges(X_train, y_train)
#             mask = np.array([0 if col in cat_cols else 1 for col in columns], dtype=np.float32)

#             if data == "EdgeIIoT":
#                 n_epochs = 100
#             elif data == "WUSTLIIoT":
#                 n_epochs = 100
#             elif data == "XIIoTID":
#                 n_epochs = 100
#             elif data == "ToNIoT":
#                 n_epochs = 100

#             targeted_trainer, untargeted_trainer, history = train_gan(
#                 trainloader=DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
#                                                     torch.tensor(y_train, dtype=torch.long)),
#                                     batch_size=256, shuffle=True),
#                 target_model=model,
#                 mask=mask,
#                 class_ranges=class_ranges,
#                 benign_label=benign_label,
#                 dim=len(columns),
#                 device=device,
#                 eps_bound=0.6,
#                 n_epochs=n_epochs
#             )

#             # if data == "ToNIoT":
#             #     eps_bound = 0.5
#             # else:
#             #     eps_bound = 0.5

#             N = int(len(y_test)/5)
#             xb = torch.tensor(X_test[:N], dtype=torch.float32).to(device)
#             yb = torch.tensor(y_test[:N], dtype=torch.long).to(device)

#             idx = torch.arange(len(yb)).to(device)

#             attack_mask = (yb != benign_label)
#             benign_mask = (yb == benign_label)

#             attack_idx = idx[attack_mask]
#             benign_idx = idx[benign_mask]
#             # G, trainer = gan_adv_generation(data, scenario_option, X_train, X_val, y_train, y_val, model, wb=True)
#             # G.eval()
            
#             xb_attack = xb[attack_mask]
#             yb_attack = yb[attack_mask]

#             with torch.no_grad():
#                 eps_bound = 0.6
#                 delta_t = targeted_trainer.enforce1_mask(targeted_trainer.G(xb_attack)) * eps_bound
#                 x_adv_attack = targeted_trainer.enforce2_clip(
#                     xb_attack, xb_attack + delta_t, yb_attack)
#                 #     clip_to_target=True,
#                 #     # target_labels=torch.ones_like(yb_attack) * benign_label
#                 #     target_labels=yb_attack
#                 # )
            
#             xb_benign = xb[benign_mask]
#             yb_benign = yb[benign_mask]
            
#             with torch.no_grad():
#                 eps_bound = 0.6
#                 delta_u = untargeted_trainer.enforce1_mask(untargeted_trainer.G(xb_benign)) * eps_bound
#                 x_adv_benign = untargeted_trainer.enforce2_clip(xb_benign, xb_benign + delta_u, yb_benign)
                
#             x_adv_test = torch.zeros_like(xb)
#             y_adv_test = torch.zeros_like(yb)

#             x_adv_test[attack_idx] = x_adv_attack
#             y_adv_test[attack_idx] = yb_attack

#             x_adv_test[benign_idx] = x_adv_benign
#             y_adv_test[benign_idx] = yb_benign

#             # pdb.set_trace()
#             # with torch.no_grad():

#             #     delta = G(xb)
#             #     delta = trainer.enforce1_mask(delta) * eps_bound 
#             #     x_adv_test = trainer.enforce2_clip(xb, xb + delta, yb, clip_to_target=True, target_labels=torch.ones_like(yb) * benign_label)
            
#             # x_adv_test = gan_adv_generation(data, option, X_train, X_test, y_train, y_test, model, wb=True, adv=False)
#             # x_adv_test = gan_adv_generation(data, option, X_train, X_val, y_train, y_val, model, wb=True, adv=False)
#             # x_adv_arr = np.concatenate(x_adv_test, axis=0)
            
#             # x_scaled_arr = scaler.inverse_transform(x_adv_test.detach().cpu().numpy())
#             adv_test_df = pd.DataFrame(x_adv_test.detach().cpu().numpy(), columns=columns)
#             adv_test_label = le.inverse_transform(y_test[:N])
#             adv_test_df['label'] = adv_test_label
#             adv_test_df.to_csv(f"adv_samples/gan/adv_test_{data}_clip.csv", index=False)
#             # pdb.set_trace()
            
#             test_data = TensorDataset(x_adv_test, torch.from_numpy(y_test[:N]).long())
#             # test_data = TensorDataset(torch.from_numpy(X_test[N:]).float(), torch.from_numpy(y_test[N:]).long())
#             test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

#             adv_test_labels, adv_test_preds, _ = evaluate(model, test_loader)
#             c_report_dict = classification_report(adv_test_labels, adv_test_preds, target_names=le.classes_, output_dict=True, zero_division=0)
#             print(c_report_dict)
#             f.write(f"Adversarial Attack : accuracy={c_report_dict['accuracy']*100:.2f}%, precision={c_report_dict['macro avg']['precision']*100:.2f}%, recall={c_report_dict['macro avg']['recall']*100:.2f}%, f1={c_report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
            
#             combined_X_test = np.concatenate((x_adv_test.detach().cpu().numpy(), X_test[N:]), axis=0)
#             combined_y_test = np.concatenate((y_test[:N], y_test[N:]), axis=0)

#             combined_test_data = TensorDataset(torch.from_numpy(combined_X_test).float(), torch.from_numpy(combined_y_test).long())
#             combined_test_loader = DataLoader(combined_test_data, batch_size=256, shuffle=False)
            
#             adv_test_labels_1, adv_test_preds_1, _ = evaluate(model, combined_test_loader)
#             c_report_dict_1 = classification_report(adv_test_labels_1, adv_test_preds_1, target_names=le.classes_, output_dict=True, zero_division=0)
#             print(c_report_dict_1)
#             f.write(f"Adversarial Attack : accuracy={c_report_dict_1['accuracy']*100:.2f}%, precision={c_report_dict_1['macro avg']['precision']*100:.2f}%, recall={c_report_dict_1['macro avg']['recall']*100:.2f}%, f1={c_report_dict_1['macro avg']['f1-score']*100:.2f}%\n\n")

#             adv_test_preds_np = np.array(adv_test_preds)
#             adv_test_labels_np = np.array(adv_test_labels)
#             benign_mask = (adv_test_labels_np == benign_label)
#             attack_mask = (adv_test_labels_np != benign_label)
#             benign_asr = np.mean(adv_test_preds_np[benign_mask] != benign_label) * 100
#             attack_asr = np.mean(adv_test_preds_np[attack_mask] == benign_label) * 100
#             f.write(f"Benign → Attack ASR = {benign_asr:.2f}%\n")
#             f.write(f"Attack → Benign ASR = {attack_asr:.2f}%\n\n")

#             # pdb.set_trace()
            
#             for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
#                 order = np.bincount(adv_test_labels).argsort()[::-1]
#                 cm = confusion_matrix(adv_test_labels, adv_test_preds, normalize=normalize)
#                 # benign_idx = np.where
#                 cm = cm[order][:, order]
#                 fig, ax = plt.subplots(figsize=(10, 8))

#                 if data.lower() == 'wustliiot':
#                     class_names = ["Normal", "DoS", "Reconnaissance", "Backdoor", "Command Injection"]
#                     display_labels = [class_names[i] for i in order]
#                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
#                 else:
#                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
#                 disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

#                 plt.title(f"Confusion Matrix - GAN (After Adversarial Attack)")
#                 plt.xlabel("Predicted Label")
#                 plt.ylabel("True Label")

#                 plt.xticks(rotation=45, ha='right')
#                 plt.tight_layout()
#                 plt.savefig(f"cm/{data}_GAN_adv_attack{suffix}_clip.png", dpi=300, bbox_inches='tight')
#                 plt.close()


#             # x_adv_train = gan_adv_generation(data, option, X_train, X_val, y_train, y_val, model, wb=True, adv=True)
#             # eps_bound = 0.3
#             xt = torch.tensor(X_val, dtype=torch.float32).to(device)
#             yt = torch.tensor(y_val, dtype=torch.long).to(device)
#             idx1 = torch.arange(len(yt)).to(device)

#             attack_mask = (yt != benign_label)
#             benign_mask = (yt == benign_label)

#             attack_idx1 = idx1[attack_mask]
#             benign_idx1 = idx1[benign_mask]

#             xt_attack = xt[attack_mask]
#             yt_attack = yt[attack_mask]

#             with torch.no_grad():
#                 eps_bound = 0.5
#                 delta_t = targeted_trainer.enforce1_mask(targeted_trainer.G(xt_attack)) * eps_bound
#                 x_adv_attack_1 = targeted_trainer.enforce2_clip(
#                     xt_attack, xt_attack + delta_t, yt_attack)
#                 #     clip_to_target=True,
#                 #     # target_labels=torch.ones_like(yt_attack) * benign_label
#                 #     target_labels=yt_attack
#                 # )
            
#             xt_benign = xt[benign_mask]
#             yt_benign = yt[benign_mask]

#             with torch.no_grad():
#                 eps_bound = 0.6
#                 delta_u = untargeted_trainer.enforce1_mask(untargeted_trainer.G(xt_benign)) * eps_bound
#                 x_adv_benign_1 = untargeted_trainer.enforce2_clip(xt_benign, xt_benign + delta_u, yt_benign)
                
#             x_adv_train = torch.zeros_like(xt)
#             y_adv_train = torch.zeros_like(yt)

#             x_adv_train[attack_idx1] = x_adv_attack_1
#             y_adv_train[attack_idx1] = yt_attack

#             x_adv_train[benign_idx1] = x_adv_benign_1
#             y_adv_train[benign_idx1] = yt_benign
            
#             # pdb.set_trace()

#             # with torch.no_grad():
#             #     x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
#             #     y_t = torch.tensor(y_val, dtype=torch.long).to(device)

#             #     delta_t = G(x_t)
#             #     delta_t = trainer.enforce1_mask(delta_t) * eps_bound 
#             #     x_adv_train = trainer.enforce2_clip(x_t, x_t + delta_t, y_t, clip_to_target=True, target_labels=torch.ones_like(y_t) * benign_label)
            
#             x_adv_train_np = x_adv_train.detach().cpu().numpy()
#             adv_train_df = pd.DataFrame(x_adv_train_np, columns=columns)
#             adv_train_label = le.inverse_transform(y_val)
#             adv_train_df['label'] = adv_train_label
#             adv_train_df.to_csv(f"adv_samples/gan/adv_train_{data}_clip.csv", index=False)

#             combined_X_train = np.concatenate((x_adv_train_np, X_train), axis=0)
#             combined_y_train = np.concatenate((y_val, y_train), axis=0)

#             adv_model, _ = train(best_ids_params, combined_X_train, X_test, combined_y_train, y_test)
#             adv_train_labels, adv_train_preds, _ = evaluate(adv_model, test_loader)
#             adv_report_dict = classification_report(adv_train_labels, adv_train_preds, target_names=le.classes_, output_dict=True, zero_division=0)
#             print(adv_report_dict)
#             f.write(f"Adversarial Training : accuracy={adv_report_dict['accuracy']*100:.2f}%, precision={adv_report_dict['macro avg']['precision']*100:.2f}%, recall={adv_report_dict['macro avg']['recall']*100:.2f}%, f1={adv_report_dict['macro avg']['f1-score']*100:.2f}%\n\n")
            
#             adv_train_labels_1, adv_train_preds_1, _ = evaluate(adv_model, combined_test_loader)
#             adv_report_dict_1 = classification_report(adv_train_labels_1, adv_train_preds_1, target_names=le.classes_, output_dict=True, zero_division=0)
#             print(adv_report_dict_1)
#             f.write(f"Adversarial Training : accuracy={adv_report_dict_1['accuracy']*100:.2f}%, precision={adv_report_dict_1['macro avg']['precision']*100:.2f}%, recall={adv_report_dict_1['macro avg']['recall']*100:.2f}%, f1={adv_report_dict_1['macro avg']['f1-score']*100:.2f}%\n\n")
#             adv_train_preds_np = np.array(adv_train_preds)
#             adv_train_labels_np = np.array(adv_train_labels)
#             benign_mask = (adv_train_labels_np == benign_label)
#             attack_mask = (adv_train_labels_np != benign_label)
#             benign_asr = np.mean(adv_train_preds_np[benign_mask] != benign_label) * 100
#             attack_asr = np.mean(adv_train_preds_np[attack_mask] == benign_label) * 100
#             f.write(f"Benign → Attack ASR = {benign_asr:.2f}%\n")
#             f.write(f"Attack → Benign ASR = {attack_asr:.2f}%\n\n")

#             for normalize, suffix, fmt in [(None, "", "d"), ("true", "_normalized", "0.2f")]:
#                 order = np.bincount(adv_train_labels).argsort()[::-1]
#                 cm = confusion_matrix(adv_train_labels, adv_train_preds, normalize=normalize)
#                 cm = cm[order][:, order]
#                 fig, ax = plt.subplots(figsize=(10, 8))

#                 if data.lower() == 'wustliiot':
#                     class_names = ["Normal", "DoS", "Reconnaissance", "Backdoor", "Command Injection"]
#                     display_labels = [class_names[i] for i in order]
#                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
#                 else:
#                     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_[order])
#                 disp.plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)

#                 plt.title(f"Confusion Matrix - GAN (After Adversarial Training)")
#                 plt.xlabel("Predicted Label")
#                 plt.ylabel("True Label")

#                 plt.xticks(rotation=45, ha='right')
#                 plt.tight_layout()
#                 plt.savefig(f"cm/{data}_GAN_adv_training{suffix}_clip.png", dpi=300, bbox_inches='tight')
#                 plt.close()
    
    
#     # pdb.set_trace()
#     exit()





    # if data.lower() == 'toniot':
    #     filename = 'dataset/NF-ToN-IoT-v3_processed_reduced.csv'
    #     scenario_option = '0'
    #     benign_label = 1
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_nftoniot(filename, scenario_option)
        
    #     df = pd.read_csv(filename)
    #     df_f = df.drop(columns=['Label', 'Attack'])
    #     columns = df_f.columns.tolist()
        
    #     categorical_cols = [
    #     "L4_SRC_PORT","L4_DST_PORT",
    #     "PROTOCOL","L7_PROTO",
    #     "TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS",
    #     "ICMP_TYPE","ICMP_IPV4_TYPE",
    #     "DNS_QUERY_ID","DNS_QUERY_TYPE","FTP_COMMAND_RET_CODE"
    #     ]

    # elif data.lower() == 'wustliiot':
    #     filename = 'dataset/wustl_iiot_2021.csv'
    #     scenario_option = '0'
    #     benign_label = 0
    #     X_train, X_val, X_test, y_train, y_val, y_test, le = preprocess_wustlIIoT(filename, scenario_option)

    #     df = pd.read_csv(filename)
    #     drop_columns = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Traffic', 'Target']
    #     df_f = df.drop(drop_columns, axis=1, inplace=True)
    #     columns = df_f.columns.tolist()

    #     categorical_cols = ["Sport", "Dport", "Proto"]
    
    # elif data.lower()== 'xiiotid':
    #     pass
    # elif data.lower()== 'edgeiiot':
    #     pass


    # num_columns= list(set(columns) - set(categorical_cols))
    # print(len(num_columns))

    # class_ranges = compute_class_ranges(X_test, y_test)
    # D = len(columns)
    # mask = np.array([0 if col in categorical_cols else 1 for col in columns], dtype=np.float32)

    # trainer = NIDSGANTrainer(dim=D, mask=mask, class_ranges=class_ranges, benign_label=benign_label, device=device)
    # # pdb.set_trace()


    # wb = True
    # if wb:
    #     best_ids_params = find_best_model(data, '0')
    #     final_model, test_loader = train(best_ids_params, X_train, X_test, y_train, y_test)
    #     _, _, model = evaluate(final_model, test_loader)
    #     history, G, C = trainer.train(X_test, y_test, target_model=model, num_classes=len(np.unique(y_test)), batch_size=256, n_epochs=30, critic_iters=5, eps_bound=0.2)
        
    # else:
    #     history, G, C = trainer.train(X_test, y_test, target_model=None, num_classes=len(np.unique(y_test)), batch_size=256, n_epochs=30, critic_iters=5, eps_bound=0.2)        
    
    # device = trainer.device
    # G.eval()
    # idx_attack = np.where(y_test != benign_label)[0][:10]
    # xb = torch.tensor(X_test[idx_attack], dtype=torch.float32).to(device)
    # yb = torch.tensor(y_test[idx_attack], dtype=torch.long).to(device)

    # delta = G(xb)
    # delta = trainer.enforce1_mask(delta) * 0.2
    # x_star = trainer.enforce2_clip(xb, xb + delta, yb)

    # combined_X_train = np.concatenate((x_star, X_train), axis=0)
    # combined_y_train = np.concatenate((y_val, y_train), axis=0)

    
    # adv_model, adv_test_loader = train(best_ids_params, combined_X_train, adv_test, combined_y_train, y_test[:N])
    # adv_test_labels, adv_test_preds, _ = evaluate(adv_model, adv_test_loader)
                    

    # print("Original X (first sample):", xb[0].cpu().numpy())
    # print("Perturbed X* (first sample):", x_star[0].detach().cpu().numpy())
    # print("Done demo. Check saved checkpoints in ./checkpoints/")
