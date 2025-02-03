import numpy as np
from scoring import cost_function, find_optimal_weights
import math

# Set random seed for reproducibility
np.random.seed(42)

def generate_samples(true_logits, n_comparisons, noise_std=0.5):
    """Generate noisy pairwise comparisons from true logits"""
    samples = []
    n = len(true_logits)
    for _ in range(n_comparisons):
        i, j = np.random.choice(n, 2, replace=False)
        # True difference plus Gaussian noise
        c = true_logits[j] - true_logits[i] + np.random.normal(0, noise_std)
        samples.append((i, j, c))
    return samples

def generate_samples_partial(true_logits, n_comparisons, visible_nodes, noise_std=0.5):
    """Generate noisy pairwise comparisons from true logits, using only visible nodes"""
    samples = []
    for _ in range(n_comparisons):
        # Only choose from visible nodes for comparisons
        i, j = np.random.choice(visible_nodes, 2, replace=False)
        # True difference plus Gaussian noise
        c = true_logits[j] - true_logits[i] + np.random.normal(0, noise_std)
        samples.append((i, j, c))
    return samples

def recover_logits(samples, n_nodes):
    """Recover logits using optimization"""
    # Create a single "model" that we'll optimize directly
    initial_logits = np.zeros(n_nodes)
    
    def objective(x):
        return cost_function(x, samples)
    
    # Use scipy's minimize with the initial guess
    from scipy.optimize import minimize
    result = minimize(objective, initial_logits, method='BFGS')
    return result.x

def calculate_error(true_logits, recovered_logits):
    """Calculate relative error between true and recovered logits"""
    # Normalize both to have zero mean to handle translation invariance
    true_centered = true_logits - np.mean(true_logits)
    recovered_centered = recovered_logits - np.mean(recovered_logits)
    
    # Scale to handle scale invariance
    true_scaled = true_centered / np.std(true_centered)
    recovered_scaled = recovered_centered / np.std(recovered_centered)
    
    return np.mean(np.abs(true_scaled - recovered_scaled))

def run_experiment(n_nodes=34, n_comparisons=3400, noise_std=0.5, n_trials=5):
    """Run multiple trials and average the results"""
    errors = []
    
    for trial in range(n_trials):
        # print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Generate true logits
        true_logits = np.random.normal(0, 1, n_nodes)
        
        # Generate noisy samples
        samples = generate_samples(true_logits, n_comparisons, noise_std)
        
        # Recover logits
        recovered_logits = recover_logits(samples, n_nodes)
        
        # Calculate error
        error = calculate_error(true_logits, recovered_logits)
        errors.append(error)
        # print(f"Trial error: {error:.2%}")
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"\nResults over {n_trials} trials:")
    print(f"Mean error: {mean_error:.2%}")
    print(f"Standard deviation: {std_error:.2%}")
    
    return mean_error, std_error

def run_partial_experiment(n_nodes=34, n_comparisons=3400, noise_std=0.5, n_trials=5, visible_fraction=0.5):
    """Run experiment where only a fraction of nodes are visible to judges"""
    errors = []
    
    n_visible = int(n_nodes * visible_fraction)
    
    for trial in range(n_trials):
        # print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Generate true logits for all nodes
        true_logits = np.random.normal(0, 1, n_nodes)
        
        # Randomly select visible nodes
        visible_nodes = np.random.choice(n_nodes, n_visible, replace=False)
        
        # Generate noisy samples using only visible nodes
        samples = generate_samples_partial(true_logits, n_comparisons, visible_nodes, noise_std)
        
        # Recover logits
        # print("Recovering logits...")
        recovered_logits = recover_logits(samples, n_nodes)
        
        # Calculate error
        error = calculate_error(true_logits, recovered_logits)
        errors.append(error)
        # print(f"Trial error: {error:.2%}")
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"\nResults over {n_trials} trials (with {visible_fraction*100:.0f}% visible nodes):")
    print(f"Mean error: {mean_error:.2%}")
    print(f"Standard deviation: {std_error:.2%}")
    
    return mean_error, std_error

if __name__ == "__main__":
    # Run experiments with different numbers of comparisons
    comparison_counts = [100, 500, 1000, 2000, 3000]
    
    print("Running experiments with 34 seed nodes")
    mean_errors_34 = []
    std_errors_34 = []
    for n_comparisons in comparison_counts:
        print(f"\n=== Testing with {n_comparisons} comparisons ===")
        mean_error, std_error = run_experiment(
            n_nodes=34,
            n_comparisons=n_comparisons,
            noise_std=0.5,
            n_trials=5
        )
        mean_errors_34.append(mean_error)
        std_errors_34.append(std_error)

    print(f'Running experiments with 120 dependencies')
    mean_errors_120 = []
    std_errors_120 = []
    for n_comparisons in comparison_counts:
        print(f"\n=== Testing with {n_comparisons} comparisons ===")
        mean_error, std_error = run_experiment(
            n_nodes=120,
            n_comparisons=n_comparisons,
            noise_std=0.5,
            n_trials=5
        )
        mean_errors_120.append(mean_error)
        std_errors_120.append(std_error)

    print("\n=== Testing with 50% visible nodes ===")
    comparison_counts_partial = [100, 1000, 3000, 5000]
    
    # Results for 34 nodes with 50% visibility
    print("\nRunning experiments with 34 seed nodes (50% visible)")
    mean_errors_partial_34 = []
    std_errors_partial_34 = []
    for n_comparisons in comparison_counts_partial:
        print(f"\n=== Testing with {n_comparisons} comparisons ===")
        mean_error, std_error = run_partial_experiment(
            n_nodes=34,
            n_comparisons=n_comparisons,
            noise_std=0.5,
            n_trials=5,
            visible_fraction=0.5
        )
        mean_errors_partial_34.append(mean_error)
        std_errors_partial_34.append(std_error)

    # Results for 120 nodes with 50% visibility
    print("\nRunning experiments with 120 nodes (50% visible)")
    mean_errors_partial_120 = []
    std_errors_partial_120 = []
    for n_comparisons in comparison_counts_partial:
        print(f"\n=== Testing with {n_comparisons} comparisons ===")
        mean_error, std_error = run_partial_experiment(
            n_nodes=120,
            n_comparisons=n_comparisons,
            noise_std=0.5,
            n_trials=5,
            visible_fraction=0.5
        )
        mean_errors_partial_120.append(mean_error)
        std_errors_partial_120.append(std_error)
