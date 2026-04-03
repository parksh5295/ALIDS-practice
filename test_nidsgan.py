"""Evaluate NIDSGAN-generated traffic against IDS models (same protocol as test_wgan.py)."""

import os

import numpy as np
import pandas as pd
import torch

from data import load_test, preprocess, split_features, preprocess_with_domain_info
from nidsgan import NIDSGAN, reassemble_torch
from test_wgan import IDS_CONFIGS, get_tester, print_results
from train_nidsgan import parse_arguments


def _ids_configs_for_eval(eval_ids: str):
    """Subset of IDS_CONFIGS; each entry needs a trained file under configs/*.yaml save_model."""
    keys = [n for n, _ in IDS_CONFIGS]
    s = (eval_ids or 'all').strip().lower()
    if s == 'all':
        return IDS_CONFIGS
    wanted = {x.strip() for x in eval_ids.split(',') if x.strip()}
    out = [(n, p) for n, p in IDS_CONFIGS if n in wanted]
    if not out:
        raise SystemExit(f'--eval_ids: no match for {eval_ids!r}. Valid keys: {keys}')
    missing = wanted - {n for n, _ in out}
    if missing:
        raise SystemExit(f'--eval_ids: unknown keys {missing}. Valid keys: {keys}')
    return out


def main():
    options = parse_arguments()
    functional_features, non_functional_features, normal_ff, normal_nff = split_features(
        load_test(), selected_attack_class=options.attack
    )
    adversarial_ff, _ = preprocess(functional_features, normalize=options.normalize)
    adversarial_nff, labels_mal = preprocess(non_functional_features, normalize=options.normalize)
    nor_nff, labels_nor = preprocess(normal_nff, normalize=options.normalize)
    nor_ff, _ = preprocess(normal_ff, normalize=options.normalize)

    n_attributes = adversarial_nff.shape[1]
    full_input_size = reassemble_torch(
        options.attack,
        torch.tensor(adversarial_nff[:1], dtype=torch.float32),
        torch.tensor(adversarial_ff[:1], dtype=torch.float32),
    ).shape[1]

    _, _, pmask, feat_min, feat_max = preprocess_with_domain_info(
        non_functional_features, normalize=options.normalize
    )

    model = NIDSGAN(options, n_attributes, full_input_size, options.attack, pmask, feat_min, feat_max)
    save_model_directory = os.path.join(options.save_model, options.name)
    model.load(save_model_directory)

    adversarial = model.generate(adversarial_nff, adversarial_ff).detach().cpu().numpy()
    data = reassemble(options.attack, adversarial, adversarial_ff, nor_nff, nor_ff)
    labels = np.concatenate((labels_mal, labels_nor), axis=0)
    ids_configs = _ids_configs_for_eval(getattr(options, 'eval_ids', 'all'))
    tester = get_tester(options.attack, data, labels)
    results = list(map(tester, ids_configs))
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'{options.name}_nidsgan.csv'), 'w') as result_file:
        pd.DataFrame(results, columns=['algorithm', 'accuracy', 'f1', 'precision', 'recall', 'detection_rate']).to_csv(result_file, index=False)
    print_results(results)


def reassemble(attack_type, adversarial_nff, adversarial_ff, normal_nff, normal_ff):
    if attack_type == 'DoS':
        intrinsic = adversarial_ff[:, :6]
        content = adversarial_nff[:, :13]
        time_based = adversarial_ff[:, 6:15]
        host_based = adversarial_nff[:, 13:]
        categorical = adversarial_ff[:, 15:]
        adversarial_traffic = np.concatenate((intrinsic, content, time_based, host_based, categorical), axis=1)

        intrinsic_normal = normal_ff[:, :6]
        content_normal = normal_nff[:, :13]
        time_based_normal = normal_ff[:, 6:15]
        host_based_normal = normal_nff[:, 13:]
        categorical_normal = normal_ff[:, 15:]
        normal_traffic = np.concatenate((intrinsic_normal, content_normal, time_based_normal, host_based_normal, categorical_normal), axis=1)
    elif attack_type == 'Probe':
        intrinsic = adversarial_ff[:, :6]
        content = adversarial_nff[:, :13]
        time_based = adversarial_ff[:, 6:15]
        host_based = adversarial_ff[:, 15:25]
        categorical = adversarial_ff[:, 25:]
        adversarial_traffic = np.concatenate((intrinsic, content, time_based, host_based, categorical), axis=1)

        intrinsic_normal = normal_ff[:, :6]
        content_normal = normal_nff[:, :13]
        time_based_normal = normal_ff[:, 6:15]
        host_based_normal = normal_ff[:, 15:25]
        categorical_normal = normal_ff[:, 25:]
        normal_traffic = np.concatenate((intrinsic_normal, content_normal, time_based_normal, host_based_normal, categorical_normal), axis=1)
    else:
        raise ValueError(attack_type)

    return np.concatenate((adversarial_traffic, normal_traffic), axis=0)


if __name__ == '__main__':
    main()
