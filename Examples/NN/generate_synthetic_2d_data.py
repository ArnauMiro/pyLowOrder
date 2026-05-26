import numpy as np
import matplotlib.pyplot as plt

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
    
    def save_dataset(self, solutions, parameters, filename_prefix="analytical_dataset"):
        """Save the dataset to numpy files."""
        np.save(f"{filename_prefix}_solutions.npy", solutions)
        np.save(f"{filename_prefix}_parameters.npy", parameters)
        
        print(f"\nDataset saved as {filename_prefix}_solutions.npy and {filename_prefix}_parameters.npy")
        print(f"Solutions shape: {solutions.shape}")
        print(f"Parameters shape: {parameters.shape}")
        print(f"Solutions range: [{solutions.min():.4f}, {solutions.max():.4f}]")
    
    def visualize_samples(self, solutions, parameters, n_samples=4, figsize=(12, 10)):
        """Visualize some samples from the dataset."""
        n_samples = min(n_samples, len(solutions))
        indices = np.random.choice(len(solutions), n_samples, replace=False)
        
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.ravel() if n_samples > 1 else axes
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            im = axes[i].contourf(self.X, self.Y, solutions[idx], levels=50, cmap='RdBu_r')
            axes[i].set_title(f'α₁={parameters[idx][0]:.2f}, α₂={parameters[idx][1]:.2f}')
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_aspect('equal')
            plt.colorbar(im, ax=axes[i])
        
        # Hide empty subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_dataset(self, solutions, parameters):
        """Analyze the generated dataset."""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        print(f"Function: Complex nonlinear function with multiple interacting terms")
        print(f"f(x,y) = α₁*sin(α₂*x + y)*cos(x*y/α₁) + α₂*exp(-((x-π)² + (y-π)²)/(2*α₁²))*sin(α₁*x*y/5)")
        print(f"       + 0.3*tanh(α₁*sin(2x) + α₂*cos(3y)) + 0.2*α₁*α₂*sin(x+α₂)*sin(y+α₁)/(1+0.1*(x²+y²))")
        print(f"Number of samples: {len(solutions)}")
        print(f"Grid size: {self.nx} × {self.ny}")
        print(f"Domain: x ∈ [{self.x_range[0]:.2f}, {self.x_range[1]:.2f}], y ∈ [{self.y_range[0]:.2f}, {self.y_range[1]:.2f}]")
        
        print(f"\nParameter Statistics:")
        print(f"α₁: min={parameters[:, 0].min():.3f}, max={parameters[:, 0].max():.3f}, mean={parameters[:, 0].mean():.3f}")
        print(f"α₂: min={parameters[:, 1].min():.3f}, max={parameters[:, 1].max():.3f}, mean={parameters[:, 1].mean():.3f}")
        
        print(f"\nFunction Values:")
        print(f"f(x,y): min={solutions.min():.3f}, max={solutions.max():.3f}, mean={solutions.mean():.3f}")
        print(f"Standard deviation: {solutions.std():.3f}")
        
        # Theoretical maximum (when both sin and cos are at their peaks)
        max_alpha1 = max(abs(parameters[:, 0].min()), abs(parameters[:, 0].max()))
        max_alpha2 = max(abs(parameters[:, 1].min()), abs(parameters[:, 1].max()))
        theoretical_max = max_alpha1 * max_alpha2
        print(f"Theoretical maximum: ±{theoretical_max:.3f}")

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = AnalyticalFunctionDataset(nx=64, ny=64, x_range=(0, 2*np.pi), y_range=(0, 2*np.pi))
    
    # Method 1: Random sampling
    print("\n" + "="*50)
    print("METHOD 1: Random Parameter Sampling")
    print("="*50)
    
    solutions_random, parameters_random = generator.generate_dataset(
        n_samples=1000,
        alpha1_range=(-2.0, 2.0),
        alpha2_range=(-2.0, 2.0)
    )
    
    # Save random dataset
    generator.save_dataset(solutions_random, parameters_random, "train_medium_resolution")
    
    # Analyze dataset
    generator.analyze_dataset(solutions_random, parameters_random)
    
    # Visualize samples
    generator.visualize_samples(solutions_random, parameters_random, n_samples=4)
    
    # Method 2: Grid sampling (optional)
    # print("\n" + "="*50)
    # print("METHOD 2: Regular Grid Sampling")
    # print("="*50)
    
    # alpha1_grid = np.linspace(-2, 2, 10)
    # alpha2_grid = np.linspace(-2, 2, 10)
    
    # solutions_grid, parameters_grid = generator.generate_grid_dataset(alpha1_grid, alpha2_grid)
    
    # # Save grid dataset
    # generator.save_dataset(solutions_grid, parameters_grid, "analytical_grid")
    
    # # Analyze grid dataset
    # generator.analyze_dataset(solutions_grid, parameters_grid)
    
    # print("\n" + "="*50)
    # print("DATASETS READY FOR CNN TRAINING!")
    # print("="*50)