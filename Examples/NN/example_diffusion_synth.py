from generate_synthetic_2d_data import AnalyticalFunctionDataset

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pyLOM.NN import Diffusion, DiT, RegressionEvaluator
import pyLOM

data_resolution = (16, 16)
generator = AnalyticalFunctionDataset(nx=data_resolution[0], ny=data_resolution[1], x_range=(0, 2*np.pi), y_range=(0, 2*np.pi))

solutions_random, parameters_random = generator.generate_dataset(
    n_samples=1000,
    alpha1_range=(-2.0, 2.0),
    alpha2_range=(-2.0, 2.0)
)
pyLOM.pprint(0, f"Generated {len(solutions_random)} samples with random parameter sampling with shape {solutions_random.shape}.")
# add channel dimension to solutions
solutions_random = solutions_random[:, None, :, :]
n_train = int(0.8 * len(solutions_random))
# standardize the data
solutions_train = torch.from_numpy(solutions_random[:n_train]).float()
solutions_test = torch.from_numpy(solutions_random[n_train:]).float()
train_mean, train_std = solutions_train.mean(), solutions_train.std()
solutions_train = (solutions_train - train_mean) / train_std
solutions_test = (solutions_test - train_mean) / train_std
train_data = TensorDataset(solutions_train, torch.from_numpy(parameters_random[:n_train]).float())
test_data = TensorDataset(solutions_test, torch.from_numpy(parameters_random[n_train:]).float())

model = DiT(
    depth=4,
    hidden_size=128,
    patch_size=1,
    num_heads=4,
    input_size=data_resolution, # dataset grid size
    cond_dim=2, # number of parameters (alpha1, alpha2)
    class_dropout_prob=0.2,
    in_channels=1,
    use_swiglu=True,
    # qk_norm=True, # when bf16 training
    attn_type="vanilla",  # window, linear, vanilla
    mlp_ratio=2.5,
)

results_folder = 'synthetic_data_experiment'

diffusion = Diffusion(
    model,
    input_size=data_resolution,
    cond_scale=2.0,
    sampling_method="euler",
    num_sampling_steps=400,
    results_folder=results_folder,
    use_cpu=True, 
)

diffusion.fit(
    train_data,
    dataset_test=test_data, # small_val_dataset is to avoid timeout when training on 2 GPUs
    train_batch_size=64,
    train_lr=2e-4,
    train_num_steps=100,  # total training steps
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp=True,     # turn on mixed precision
    # mixed_precision_type='bf16',
    save_and_sample_every=20000,
    eta_min_scheduler=1e-6,
    max_grad_norm=1.0,
    compile_model=True,
    split_batches=True
)

diffusion.save(f"{results_folder}/final_model.pt")

diffusion.load(f"{results_folder}/final_model.pt")

samples = diffusion.predict(test_data, batch_size=16)

evaluator = RegressionEvaluator()
eval_metrics = evaluator(samples, solutions_test.numpy())
evaluator.print_metrics()