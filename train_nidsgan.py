"""
Train NIDSGAN (WGAN-GP + surrogate NIDS loss + domain constraints).
Requires a trained surrogate MLP on full NSL-KDD features (see train_ids.py / configs/multi_layer_perceptron.yaml).
"""

import os

import configargparse
import numpy as np
import torch
import yaml

from data import load_train, load_val, preprocess, split_features, preprocess_with_domain_info
from nidsgan import NIDSGAN, reassemble_torch


def main():
    options = parse_arguments()
    if not options.surrogate_path:
        raise SystemExit('train_nidsgan: --surrogate_path is required (trained MLP on full NSL-KDD features).')
    attack = options.attack

    train_ff, train_nff, train_nor_ff, train_nor_nff = _load_split_attack_normal(
        load_train(), attack, options.normalize
    )
    val_ff, val_nff, val_nor_ff, val_nor_nff = _load_split_attack_normal(
        load_val(), attack, options.normalize
    )

    mal_nff, labels_mal, pmask, feat_min, feat_max = preprocess_with_domain_info(train_nff, normalize=options.normalize)
    nor_nff, labels_nor, _, _, _ = preprocess_with_domain_info(train_nor_nff, normalize=options.normalize, feat_min=feat_min, feat_max=feat_max)

    val_mal_nff, val_labels_mal, _, _, _ = preprocess_with_domain_info(val_nff, normalize=options.normalize, feat_min=feat_min, feat_max=feat_max)
    val_nor_nff, val_labels_nor, _, _, _ = preprocess_with_domain_info(val_nor_nff, normalize=options.normalize, feat_min=feat_min, feat_max=feat_max)

    mal_ff, _ = preprocess(train_ff, normalize=options.normalize)
    nor_ff, _ = preprocess(train_nor_ff, normalize=options.normalize)
    val_mal_ff, _ = preprocess(val_ff, normalize=options.normalize)
    val_nor_ff, _ = preprocess(val_nor_ff, normalize=options.normalize)

    n_attributes = mal_nff.shape[1]
    full_input_size = _infer_full_dim(attack, mal_nff, mal_ff)

    trainingset = (nor_nff, mal_nff, mal_ff, nor_ff, labels_nor, labels_mal)
    validationset = (val_nor_nff, val_mal_nff, val_mal_ff, val_nor_ff, val_labels_nor, val_labels_mal)

    model = NIDSGAN(options, n_attributes, full_input_size, attack, pmask, feat_min, feat_max)
    model.train(trainingset, validationset)

    if options.save_model is not None:
        save_model_directory = os.path.join(options.save_model, options.name)
        os.makedirs(save_model_directory, exist_ok=True)
        model.save(save_model_directory)


def _load_split_attack_normal(dataframe, attack, normalize):
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(dataframe, selected_attack_class=attack)
    return functional_features, non_functional_features, normal_ff, normal_nff


def _infer_full_dim(attack, mal_nff, mal_ff):
    nff = torch.tensor(mal_nff[:1], dtype=torch.float32)
    ff = torch.tensor(mal_ff[:1], dtype=torch.float32)
    return reassemble_torch(attack, nff, ff).shape[1]


def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--normalize', required=False, action='store_true', default=False, help='normalize data (default false)')
    parser.add('--attack', required=True, default='Probe', help='selected attack (DoS or Probe)')
    parser.add('--name', required=True, type=str, help='Unique name of the experiment.')
    parser.add('--checkpoint', required=False, type=str, default=None, help='path to load checkpoint from')
    parser.add('--checkpoint_directory', required=False, type=str, default='checkpoints/', help='path to checkpoints directory (default: checkpoints/)')
    parser.add('--checkpoint_interval_s', required=False, type=int, default=1800, help='seconds between saving checkpoints (default: 1800)')
    parser.add('--evaluate', required=False, type=int, default=50, help='epochs between tensorboard eval (default: 50)')
    parser.add('--surrogate_path', required=False, type=str, default=None, help='path to trained PyTorch MLP (.pt) on full features')
    parse_nidsgan_arguments(parser)
    options = parser.parse_args()

    save_config = options.save_config
    del options.config
    del options.save_config

    if save_config is not None:
        with open(save_config, 'w') as config_file:
            yaml.dump(vars(options), config_file)

    return options


def parse_nidsgan_arguments(parser):
    g = parser.add_argument_group('nidsgan')
    g.add('--epochs', required=False, default=200, type=int, help='epochs (default 200), -1 for infinite')
    g.add('--batch_size', required=False, default=64, type=int, help='batch size (default 64)')
    g.add('--learning_rate', required=False, default=0.0001, type=float, help='learning rate (default 0.0001)')
    g.add('--noise_dim', required=False, default=9, type=int, help='noise dimension (default 9)')
    g.add('--critic_iter', required=False, default=5, type=int, help='critic steps per generator step (default 5)')
    g.add('--epsilon', required=False, default=0.3, type=float, help='L2 perturbation radius (default 0.3, paper)')
    g.add('--lambda_adv', required=False, default=1.0, type=float, help='weight for surrogate adversarial loss')
    g.add('--lambda_pert', required=False, default=0.01, type=float, help='weight for L2 perturbation magnitude penalty')
    g.add('--lambda_gp', required=False, default=10.0, type=float, help='WGAN-GP coefficient (default 10)')
    g.add('--surrogate_hidden_size', required=False, default=128, type=int, help='must match trained MLP')
    g.add('--surrogate_dropout', required=False, default=0.25, type=float, help='must match trained MLP dropout')
    g.add(
        '--eval_ids',
        required=False,
        default='all',
        type=str,
        help='test_nidsgan only: comma-separated evaluator keys (see test_wgan.IDS_CONFIGS), or "all". '
        'Example: multi_layer_perceptron only (requires only MLP .pt).',
    )


if __name__ == '__main__':
    main()
