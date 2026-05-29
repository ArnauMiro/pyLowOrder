import numpy as np
import torch
from torch.utils.data import TensorDataset

from pyLOM.NN import Diffusion, DiT, RegressionEvaluator
import pyLOM


class AnalyticalFunctionDataset:
    def __init__(self, nx=64, ny=64, x_range=(0, 2*np.pi), y_range=(0, 2*np.pi)):
        """
        Initialize the analytical function dataset generator.
        
        Parameters:
        - nx, ny: Number of grid points in x and y directions
        - x_range, y_range: Domain ranges for x and y coordinates
        """
        self.nx = nx
        self.ny = ny
        self.x_range = x_range
        self.y_range = y_range
        
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        print(f"Grid initialized: {nx}x{ny}")
        print(f"X domain: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
        print(f"Y domain: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
    
    def evaluate_function(self, alpha1, alpha2):
        """
        Evaluate a complex nonlinear function with chaotic behavior:
        f(x,y) = α₁*sin(α₂*x + y)*cos(x*y/α₁) + α₂*exp(-((x-π)² + (y-π)²)/(2*α₁²))*sin(α₁*x*y)
        
        Parameters:
        - alpha1, alpha2: Function parameters
        
        Returns:
        - result: 2D array with function values
        - parameters: dict with parameter values
        """
        # Avoid division by zero
        alpha1_safe = alpha1 if abs(alpha1) > 0.1 else 0.1 * np.sign(alpha1) if alpha1 != 0 else 0.1
        
        # Term 1: Coupled oscillations with frequency modulation
        term1 = alpha1 * np.sin(alpha2 * self.X + self.Y) * np.cos(self.X * self.Y / alpha1_safe)
        
        # Term 2: Gaussian-modulated nonlinear interaction
        gaussian = np.exp(-((self.X - np.pi)**2 + (self.Y - np.pi)**2) / (2 * alpha1_safe**2))
        term2 = alpha2 * gaussian * np.sin(alpha1 * self.X * self.Y / 5.0)  # Scale down the product to avoid extreme oscillations
        
        # Term 3: Add some chaos with a nonlinear combination
        term3 = 0.3 * np.tanh(alpha1 * np.sin(2*self.X) + alpha2 * np.cos(3*self.Y))
        
        # Term 4: Interference patterns
        term4 = 0.2 * alpha1 * alpha2 * np.sin(self.X + alpha2) * np.sin(self.Y + alpha1) / (1 + 0.1 * (self.X**2 + self.Y**2))
        
        result = term1 + term2 + term3 + term4
        
        parameters = {'alpha1': alpha1, 'alpha2': alpha2}
        
        return result, parameters
    
    def generate_dataset(self, n_samples, alpha1_range=(-2.0, 2.0), alpha2_range=(-2.0, 2.0), 
                        random_seed=42):
        """
        Generate a dataset of function evaluations.
        
        Parameters:
        - n_samples: Number of samples to generate
        - alpha1_range, alpha2_range: Ranges for parameter sampling
        - random_seed: Random seed for reproducibility
        
        Returns:
        - solutions: Array of shape (n_samples, ny, nx)
        - parameters: Array of shape (n_samples, 2) with [alpha1, alpha2]
        """
        np.random.seed(random_seed)
        
        solutions = []
        parameters = []
        
        print(f"Generating {n_samples} samples...")
        
        for i in range(n_samples):
            # Sample parameters uniformly
            alpha1 = np.random.uniform(*alpha1_range)
            alpha2 = np.random.uniform(*alpha2_range)
            
            # Evaluate function
            solution, params = self.evaluate_function(alpha1, alpha2)
            
            solutions.append(solution)
            parameters.append([alpha1, alpha2])
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        solutions = np.array(solutions)
        parameters = np.array(parameters)
        
        print(f"Dataset generation complete!")
        
        return solutions, parameters
    
    def generate_grid_dataset(self, alpha1_values, alpha2_values):
        """
        Generate a dataset on a regular grid of parameter values.
        
        Parameters:
        - alpha1_values: Array of alpha1 values
        - alpha2_values: Array of alpha2 values
        
        Returns:
        - solutions: Array of shape (n_alpha1 * n_alpha2, ny, nx)
        - parameters: Array of shape (n_alpha1 * n_alpha2, 2)
        """
        solutions = []
        parameters = []
        
        total_samples = len(alpha1_values) * len(alpha2_values)
        print(f"Generating {total_samples} samples on regular grid...")
        
        count = 0
        for alpha1 in alpha1_values:
            for alpha2 in alpha2_values:
                solution, params = self.evaluate_function(alpha1, alpha2)
                solutions.append(solution)
                parameters.append([alpha1, alpha2])
                
                count += 1
                if count % 100 == 0:
                    print(f"Generated {count}/{total_samples} samples")
        
        solutions = np.array(solutions)
        parameters = np.array(parameters)
        
        return solutions, parameters
    

data_resolution = (16, 16)
generator = AnalyticalFunctionDataset(nx=data_resolution[0], ny=data_resolution[1], x_range=(0, 2*np.pi), y_range=(0, 2*np.pi))

solutions_random, parameters_random = generator.generate_dataset(
    n_samples=1000,
    alpha1_range=(-2.0, 2.0),
    alpha2_range=(-2.0, 2.0)
)
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
    train_num_steps=10,  # total training steps
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
# evaluator.print_metrics()
pyLOM.pprint(0, 'TSUITE y            =',solutions_test.min().item(), solutions_test.max().item(), solutions_test.mean().item())
pyLOM.pprint(0,'End of output')