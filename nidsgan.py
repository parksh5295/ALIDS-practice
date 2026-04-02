"""
NIDSGAN: Generating Practical Adversarial Network Traffic Flows (WGAN-GP + surrogate NIDS loss + domain constraints).
"""

import itertools
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter

def _build_surrogate_mlp(input_size, hidden_size=128, dropout_rate=0.25):
    return nn.Sequential(
        OrderedDict([
            ('hidden1', nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)),
            ('hidden1_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden1_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden1_activation', nn.ReLU()),
            ('hidden2', nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)),
            ('hidden2_batchnorm', nn.BatchNorm1d(hidden_size)),
            ('hidden2_dropout', nn.Dropout(p=dropout_rate)),
            ('hidden2_activation', nn.ReLU()),
            ('output', nn.Linear(in_features=hidden_size, out_features=1, bias=True)),
        ])
    )


def load_surrogate_ids(path, input_size, device, hidden_size=128, dropout_rate=0.25):
    """Load PyTorch MLP surrogate (same architecture as ids.MultiLayerPerceptron)."""
    net = _build_surrogate_mlp(input_size, hidden_size, dropout_rate)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def reassemble_torch(attack_type, adv_nff, adv_ff):
    """Differentiable full feature vector from preprocessed ff / nff (matches test_wgan.reassemble)."""
    if attack_type == 'DoS':
        intrinsic = adv_ff[:, :6]
        content = adv_nff[:, :13]
        time_based = adv_ff[:, 6:15]
        host_based = adv_nff[:, 13:]
        categorical = adv_ff[:, 15:]
        return torch.cat((intrinsic, content, time_based, host_based, categorical), dim=1)
    if attack_type == 'Probe':
        intrinsic = adv_ff[:, :6]
        content = adv_nff[:, :13]
        time_based = adv_ff[:, 6:15]
        host_based = adv_ff[:, 15:25]
        categorical = adv_ff[:, 25:]
        return torch.cat((intrinsic, content, time_based, host_based, categorical), dim=1)
    raise ValueError(f'Unsupported attack_type for reassemble: {attack_type}')


def project_l2_ball(delta, epsilon):
    """Project each row so ||delta||_2 <= epsilon (NIDSGAN perturbation budget)."""
    norms = delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    scale = (epsilon / norms).clamp(max=1.0)
    return delta * scale


class PerturbationGenerator(nn.Module):
    """Maps (malicious flow + noise) -> raw perturbation before masking and projection."""

    def __init__(self, input_size, output_size):
        super().__init__()

        def block(input_dim, output_dim):
            return [nn.Linear(input_dim, output_dim), nn.ReLU(inplace=False)]

        self.model = nn.Sequential(
            *block(input_size, 128),
            *block(128, 128),
            *block(128, 128),
            *block(128, 128),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    """WGAN-GP critic on non-functional (perturbed) feature space."""

    def __init__(self, input_size):
        super().__init__()

        def block(input_dim, output_dim):
            return [nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2, inplace=False)]

        self.model = nn.Sequential(
            *block(input_size, 128),
            *block(128, 128),
            *block(128, 128),
            *block(128, 128),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


def gradient_penalty(critic, real, fake):
    """WGAN-GP gradient penalty (random interpolation between real and fake)."""
    alpha = torch.rand(real.size(0), 1, device=real.device, dtype=real.dtype)
    interp = alpha * real + (1.0 - alpha) * fake
    interp.requires_grad_(True)
    out = critic(interp)
    grad_out = grad(
        outputs=out.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grad_out.view(grad_out.size(0), -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


class NIDSGAN(object):
    """
    NIDSGAN: adversarial loss (surrogate NIDS -> benign), L2 perturbation budget,
    WGAN-GP realism loss, domain mask and valid-range clipping.
    """

    def __init__(self, options, n_attributes_nff, full_input_size, attack_type, perturb_mask, feat_min, feat_max):
        self.n_attributes = n_attributes_nff
        self.full_input_size = full_input_size
        self.attack_type = attack_type
        self.noise_dim = options.noise_dim
        self.epsilon = options.epsilon
        self.lambda_adv = options.lambda_adv
        self.lambda_pert = options.lambda_pert
        self.lambda_gp = options.lambda_gp

        self.generator = PerturbationGenerator(self.n_attributes + self.noise_dim, self.n_attributes)
        self.critic = Critic(self.n_attributes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.critic.to(self.device)

        self.perturb_mask = torch.tensor(perturb_mask, dtype=torch.float32, device=self.device).view(1, -1)
        self.feat_min = torch.tensor(feat_min, dtype=torch.float32, device=self.device).view(1, -1)
        self.feat_max = torch.tensor(feat_max, dtype=torch.float32, device=self.device).view(1, -1)

        spath = getattr(options, 'surrogate_path', None)
        if spath:
            self.surrogate = load_surrogate_ids(
                spath,
                full_input_size,
                self.device,
                hidden_size=options.surrogate_hidden_size,
                dropout_rate=options.surrogate_dropout,
            )
        else:
            self.surrogate = None

        self.epochs = options.epochs
        self.batch_size = options.batch_size
        self.learning_rate = options.learning_rate
        self.critic_iter = options.critic_iter
        self.evaluate = options.evaluate

        self.optim_G = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        self.optim_D = optim.Adam(self.critic.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))

        self.writer_train = SummaryWriter(log_dir=f'runs/{options.name}/train')
        self.writer_val = SummaryWriter(log_dir=f'runs/{options.name}/val')

        self.start_epoch = 0
        self.checkpoint_directory = os.path.join(options.checkpoint_directory, options.name)
        self.checkpoint_interval_s = options.checkpoint_interval_s
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        self.previous_checkpoint_time = time.time()
        if options.checkpoint is not None:
            self.load_checkpoint(options.checkpoint)

    def apply_constraints(self, x, delta):
        """Mask perturbation, L2 ball, then clip to valid feature ranges."""
        delta = delta * self.perturb_mask
        delta = project_l2_ball(delta, self.epsilon)
        x_adv = x + delta
        x_adv = torch.max(torch.min(x_adv, self.feat_max), self.feat_min)
        return x_adv, delta

    def forward_adv_full(self, x_ff, adv_nff):
        """Full tensor for surrogate IDS (differentiable)."""
        return reassemble_torch(self.attack_type, adv_nff, x_ff)

    def train(self, trainingset, validationset):
        if self.surrogate is None:
            raise ValueError('NIDSGAN.train requires options.surrogate_path to a trained surrogate MLP.')
        (
            normal_nff, mal_nff, mal_ff, normal_ff,
            labels_nor, labels_mal,
        ) = trainingset

        (
            val_normal_nff, val_mal_nff, val_mal_ff, val_normal_ff,
            val_labels_nor, val_labels_mal,
        ) = validationset

        mal_nff = torch.tensor(mal_nff, dtype=torch.float32, device=self.device)
        mal_ff = torch.tensor(mal_ff, dtype=torch.float32, device=self.device)
        normal_nff = torch.tensor(normal_nff, dtype=torch.float32, device=self.device)

        val_mal_nff = torch.tensor(val_mal_nff, dtype=torch.float32, device=self.device)
        val_mal_ff = torch.tensor(val_mal_ff, dtype=torch.float32, device=self.device)
        val_normal_nff = torch.tensor(val_normal_nff, dtype=torch.float32, device=self.device)
        val_normal_ff = torch.tensor(val_normal_ff, dtype=torch.float32, device=self.device)

        normal_ff = torch.tensor(normal_ff, dtype=torch.float32, device=self.device)

        self.generator.train()
        self.critic.train()

        epoch_iterator = self._get_epoch_iterator()
        for epoch in epoch_iterator:
            n_mal = len(mal_nff)
            order = np.random.permutation(n_mal)
            for start in range(0, n_mal, self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:
                    continue
                x = mal_nff[idx]
                ff = mal_ff[idx]

                for _ in range(self.critic_iter):
                    noise = torch.rand(len(idx), self.noise_dim, device=self.device)
                    g_in = torch.cat((x, noise), dim=1)
                    delta = self.generator(g_in)
                    x_adv, _ = self.apply_constraints(x, delta)
                    fake = x_adv.detach()
                    real_idx = np.random.randint(0, n_mal, size=len(idx))
                    real = mal_nff[real_idx]

                    d_real = self.critic(real).mean()
                    d_fake = self.critic(fake).mean()
                    gp = gradient_penalty(self.critic, real, fake)
                    loss_D = d_fake - d_real + self.lambda_gp * gp

                    self.optim_D.zero_grad()
                    loss_D.backward()
                    self.optim_D.step()

                noise = torch.rand(len(idx), self.noise_dim, device=self.device)
                g_in = torch.cat((x, noise), dim=1)
                delta = self.generator(g_in)
                x_adv, delta_used = self.apply_constraints(x, delta)

                full_adv = self.forward_adv_full(x_nff=None, x_ff=ff, adv_nff=x_adv)
                logits = self.surrogate(full_adv).squeeze(-1)
                loss_adv = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(logits), reduction='mean'
                )
                loss_pert = (delta_used ** 2).mean()
                loss_gan = -self.critic(x_adv).mean()
                loss_G = (
                    self.lambda_adv * loss_adv
                    + self.lambda_pert * loss_pert
                    + loss_gan
                )

                self.optim_G.zero_grad()
                loss_G.backward()
                self.optim_G.step()

            if epoch % self.evaluate == 0:
                self._log_epoch(
                    epoch,
                    mal_nff, mal_ff, normal_nff, normal_ff, labels_mal, labels_nor,
                    val_mal_nff, val_mal_ff, val_normal_nff, val_normal_ff, val_labels_mal, val_labels_nor,
                )

            current_time = time.time()
            if self.checkpoint_interval_s <= current_time - self.previous_checkpoint_time:
                self.save_checkpoint(epoch + 1)
                self.previous_checkpoint_time = time.time()

    def _log_epoch(
        self,
        epoch,
        mal_nff, mal_ff, normal_nff, normal_ff, labels_mal, labels_nor,
        val_mal_nff, val_mal_ff, val_normal_nff, val_normal_ff, val_labels_mal, val_labels_nor,
    ):
        self.generator.eval()
        self.critic.eval()
        with torch.no_grad():
            self._log_split(self.writer_train, epoch, mal_nff, mal_ff, normal_nff, normal_ff, labels_mal, labels_nor)
            self._log_split(self.writer_val, epoch, val_mal_nff, val_mal_ff, val_normal_nff, val_normal_ff, val_labels_mal, val_labels_nor)
        self.generator.train()
        self.critic.train()

    def _log_split(self, writer, epoch, mal_nff, mal_ff, normal_nff, normal_ff, labels_mal, labels_nor):
        noise = torch.rand(len(mal_nff), self.noise_dim, device=self.device)
        g_in = torch.cat((mal_nff, noise), dim=1)
        delta = self.generator(g_in)
        x_adv, _ = self.apply_constraints(mal_nff, delta)
        full_adv = self.forward_adv_full(mal_ff, x_adv)
        logits_adv = self.surrogate(full_adv).squeeze(-1)
        evasion = (logits_adv < 0).float().mean().item()
        writer.add_scalar('nidsgan/surrogate_evasion_rate_malicious', evasion, epoch)

        noise_n = torch.rand(len(normal_nff), self.noise_dim, device=self.device)
        g_in_n = torch.cat((normal_nff, noise_n), dim=1)
        delta_n = self.generator(g_in_n)
        x_n_adv, _ = self.apply_constraints(normal_nff, delta_n)
        full_n = self.forward_adv_full(normal_ff, x_n_adv)
        logits_n = self.surrogate(full_n).squeeze(-1)
        false_alarm = (logits_n >= 0).float().mean().item()
        writer.add_scalar('nidsgan/surrogate_false_alarm_on_perturbed_normal', false_alarm, epoch)

    def _get_epoch_iterator(self):
        if self.epochs < 0:
            return itertools.count(self.start_epoch)
        return range(self.start_epoch, self.epochs)

    def generate(self, malicious_nff, malicious_ff):
        """Generate adversarial non-functional features (evaluation)."""
        self.generator.eval()
        self.critic.eval()
        n = len(malicious_nff)
        x = torch.tensor(malicious_nff, dtype=torch.float32, device=self.device)
        ff = torch.tensor(malicious_ff, dtype=torch.float32, device=self.device)
        noise = torch.rand(n, self.noise_dim, device=self.device)
        g_in = torch.cat((x, noise), dim=1)
        delta = self.generator(g_in)
        x_adv, _ = self.apply_constraints(x, delta)
        return x_adv

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pt'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pt'))

    def load(self, path):
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pt'), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pt'), map_location=self.device))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optim_G.load_state_dict(checkpoint['generator_optimizer'])
        self.optim_D.load_state_dict(checkpoint['discriminator_optimizer'])
        self.start_epoch = checkpoint['epoch']

    def save_checkpoint(self, epoch):
        checkpoint = {
            'generator': self.generator.state_dict(),
            'critic': self.critic.state_dict(),
            'generator_optimizer': self.optim_G.state_dict(),
            'discriminator_optimizer': self.optim_D.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_directory, f'epoch_{epoch}.pt'))
