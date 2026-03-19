import torch
import gpytorch
import numpy as np
import os
import pandas as pd
from gpytorch.settings import max_cg_iterations, cg_tolerance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.optimize import minimize
    
class FullFieldGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(FullFieldGPModel, self).__init__(train_x, train_y, likelihood)
        
        # Simple mean function for 4D input (G, K, x, y)
        self.mean_module = gpytorch.means.LinearMean(4)
        
        # RBF kernel with ARD (automatic relevance determination)
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=4)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Use regular MultivariateNormal
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_model(train_x, train_y, num_epochs=50):
    """Train the GP model with learning rate scheduler"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = FullFieldGPModel(train_x, train_y, likelihood)

    # Training mode
    model.train()
    likelihood.train()

    # Collect all parameters
    params = list(model.parameters())
    
    # Initial learning rate
    initial_lr = 0.1  # Reduced initial learning rate
    
    # Modified optimizer settings with single parameter group
    optimizer = torch.optim.Adam(params, lr=initial_lr)
    
    # Learning rate scheduler - StepLR instead of ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,  # Reduce LR every 10 epochs
        gamma=0.5      # Multiply LR by 0.5
    )

    # Loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("\nTraining model...")
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_state = None
    
    # Store loss history
    loss_history = []
    lr_history = []

    try:
        with max_cg_iterations(4000), cg_tolerance(0.2):
            for epoch in range(num_epochs):
                optimizer.zero_grad()

                # Forward pass using the stored training data
                output = model(train_x)
                loss = -mll(output, train_y)
                
                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss at epoch {epoch+1}. Stopping training.")
                    break

                loss.backward()
                optimizer.step()
                
                # Step the scheduler
                scheduler.step()
                
                # Store loss and learning rate
                loss_history.append(loss.item())
                lr_history.append(optimizer.param_groups[0]['lr'])

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_state = {
                        'model': model.state_dict(),
                        'likelihood': likelihood.state_dict()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

                if (epoch + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.3f} - LR: {current_lr:.6f}")

        if best_state is not None:
            model.load_state_dict(best_state['model'])
            likelihood.load_state_dict(best_state['likelihood'])

    except RuntimeError as e:
        print(f"Error during training: {e}")
        return None, None, None, None

    return model, likelihood, loss_history, lr_history


def load_model(model_path, likelihood_path, train_x, train_y):
    """Load model and likelihood from saved files"""
    try:
        # Initialize model and likelihood with the training data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = FullFieldGPModel(train_x, train_y, likelihood)
        
        # Load state dictionaries
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=True))
        likelihood.load_state_dict(torch.load(likelihood_path, map_location=torch.device('cpu'),weights_only=True))
        
        return model, likelihood
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


####################### Prediction Function ##########################
def predict_field(model_ux, model_uy, likelihood, test_G, test_K, coordinates, norm_params):
    """
    Predict full field displacements for given G, K values at all coordinate points.
    """
    if model_ux is None or model_uy is None:
        return None, None, None, None

    # Set models to evaluation mode
    model_ux.eval()
    model_uy.eval()
    likelihood.eval()

    try:
        # Prepare input for all coordinate points
        n_points = len(coordinates)
        test_inputs = np.zeros((n_points, 4))
        test_inputs[:, 0] = test_G
        test_inputs[:, 1] = test_K
        test_inputs[:, 2:] = coordinates

        # Normalize inputs
        test_inputs_transformed = test_inputs.copy()
        test_inputs_transformed[:,:2] = np.log10(test_inputs[:,:2])
        test_inputs_norm = (test_inputs_transformed - norm_params['X_mean']) / norm_params['X_std']
        
        # Convert to tensor
        test_x = torch.tensor(test_inputs_norm, dtype=torch.float32)

        # Make predictions in batches
        batch_size = 1000
        ux_means = []
        uy_means = []
        ux_stds = []
        uy_stds = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(test_x), batch_size):
                batch_x = test_x[i:i+batch_size]
                
                # Predict ux
                ux_output = likelihood(model_ux(batch_x))
                ux_means.append(ux_output.mean.numpy())
                ux_stds.append(ux_output.variance.sqrt().numpy())
                
                # Predict uy
                uy_output = likelihood(model_uy(batch_x))
                uy_means.append(uy_output.mean.numpy())
                uy_stds.append(uy_output.variance.sqrt().numpy())

        # Combine batches
        ux_mean = np.concatenate(ux_means)
        uy_mean = np.concatenate(uy_means)
        ux_std = np.concatenate(ux_stds)
        uy_std = np.concatenate(uy_stds)

        # Denormalize predictions
        ux_pred = ux_mean * norm_params['Y_std'][0] + norm_params['Y_mean'][0]
        uy_pred = uy_mean * norm_params['Y_std'][1] + norm_params['Y_mean'][1]

        return ux_pred, uy_pred, ux_std, uy_std

    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        return None, None, None, None

    
def combine_snapshots_to_csv(snapshot_folder, output_file):
    """
    Combine all snapshots into a single CSV file without subsampling
    """
    all_data = []
    print("Processing snapshots...")
    
    for file in os.listdir(snapshot_folder):
        if file.startswith("snapshot_G") and file.endswith(".csv"):
            # Extract G and K from filename
            parts = file.replace("snapshot_G", "").replace(".csv", "").split("_K")
            G, K = map(float, parts)
            
            # Load displacement data
            data = pd.read_csv(os.path.join(snapshot_folder, file))
            
            # Create a new DataFrame with all data
            full_data = pd.DataFrame({
                'G': [G] * len(data),
                'K': [K] * len(data),
                'x-coordinate [m]': data['x-coordinate [m]'],
                'y-coordinate [m]': data['y-coordinate [m]'],
                'x-displacement [m]': data['x-displacement [m]'],
                'y-displacement [m]': data['y-displacement [m]']
            })
            
            all_data.append(full_data)
            print(f"Processed {file}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    print(f"Total number of points: {len(combined_df)}")
    
    return combined_df

def load_and_normalize_data(data_file):
    """
    Load and normalize the combined data
    """
    # Load data
    data = pd.read_csv(data_file)
    
    # Split into inputs and outputs
    X = data[['G', 'K', 'x-coordinate [m]', 'y-coordinate [m]']].values
    Y = data[['x-displacement [m]', 'y-displacement [m]']].values
    
    # Log transform G and K
    X_transformed = X.copy()
    X_transformed[:,:2] = np.log10(X[:,:2])
    
    # Calculate normalization parameters
    X_mean = X_transformed.mean(axis=0)
    X_std = X_transformed.std(axis=0)
    Y_mean = Y.mean(axis=0)
    Y_std = Y.std(axis=0)
    
    # Normalize
    X_normalized = (X_transformed - X_mean) / X_std
    Y_normalized = (Y - Y_mean) / Y_std
    
    # Store normalization parameters
    norm_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }
    
    return X_normalized, Y_normalized, norm_params

######################### Surrogate MSE Loss #######################################
def surrogate_mse_loss(beta, model_ux, model_uy, exp_coords, exp_disps, norm_params):
    """Calculate MSE loss using surrogate models with improved weighting and filtering"""
    G, K = beta
    
    # Physical constraints - more relaxed but still physically meaningful
    # For most engineering materials, K > G is expected
    if (G < 5e10 or G > 1e11 or K < 6e10 or K > 1.2e11):
        print(f"G: {G:.2e}, K: {K:.2e} - Out of bounds")
        return 1e10

    try:
        # Prepare input
        n_points = len(exp_coords)
        test_inputs = np.zeros((n_points, 4))
        test_inputs[:, 0] = G
        test_inputs[:, 1] = K
        test_inputs[:, 2:] = exp_coords

        # Normalize inputs
        test_inputs_transformed = test_inputs.copy()
        test_inputs_transformed[:,:2] = np.log10(test_inputs[:,:2])
        test_inputs_norm = (test_inputs_transformed - norm_params['X_mean']) / norm_params['X_std']
        
        test_x = torch.tensor(test_inputs_norm, dtype=torch.float32)

        # Filter points near the hole to reduce noise
        center_x, center_y = 40e-3, 10e-3
        radius = 4e-3
        distance_from_center = np.sqrt((exp_coords[:, 0] - center_x)**2 + 
                                      (exp_coords[:, 1] - center_y)**2)
        mask = distance_from_center >= radius
        
        # Process only points away from the hole
        masked_test_x = test_x[mask]
        
        # Get experimental values (only for points away from hole)
        exp_ux = exp_disps[mask, 0]
        exp_uy = exp_disps[mask, 1]

        # Calculate mean displacements for normalization
        mean_abs_ux = np.mean(np.abs(exp_ux))
        mean_abs_uy = np.mean(np.abs(exp_uy))
        
        # Compute weights based on mean displacements
        wx = 1.0 / max(mean_abs_ux, 1e-10)
        wy = 1.0 / max(mean_abs_uy, 1e-10)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions in batches to avoid memory issues
            batch_size = 500
            pred_ux_all = []
            pred_uy_all = []
            
            for i in range(0, len(masked_test_x), batch_size):
                batch_x = masked_test_x[i:i+batch_size]
                
                # Get predictions
                ux_output = model_ux(batch_x)
                uy_output = model_uy(batch_x)
                
                # Extract means
                pred_ux_batch = ux_output.mean.numpy()
                pred_uy_batch = uy_output.mean.numpy()
                
                # Collect predictions
                pred_ux_all.append(pred_ux_batch)
                pred_uy_all.append(pred_uy_batch)
            
            # Combine batches
            pred_ux_norm = np.concatenate(pred_ux_all)
            pred_uy_norm = np.concatenate(pred_uy_all)
        
        # Denormalize predictions
        pred_ux = pred_ux_norm * norm_params['Y_std'][0] + norm_params['Y_mean'][0]
        pred_uy = pred_uy_norm * norm_params['Y_std'][1] + norm_params['Y_mean'][1]
        
        # Calculate weighted MSE components
        mse_x = np.mean(wx**2 * (exp_ux - pred_ux)**2)
        mse_y = np.mean(wy**2 * (exp_uy - pred_uy)**2)
        
        # Calculate L2 norm errors for reporting
        l2_error_x = np.linalg.norm(wx * (exp_ux - pred_ux))
        l2_error_y = np.linalg.norm(wy * (exp_uy - pred_uy))
        
        # Combined weighted MSE loss
        total_loss = (mse_x + mse_y) / 2.0
        
        # Calculate additional metrics for debugging
        max_abs_error_x = np.max(np.abs(exp_ux - pred_ux))
        max_abs_error_y = np.max(np.abs(exp_uy - pred_uy))
        mean_abs_error_x = np.mean(np.abs(exp_ux - pred_ux))
        mean_abs_error_y = np.mean(np.abs(exp_uy - pred_uy))

        # Print diagnostics
        print(f"G: {G:.2e}, K: {K:.2e}, K/G: {K/G:.2f}")
        print(f"MSE X: {mse_x:.2e}, MSE Y: {mse_y:.2e}")
        print(f"L2 Error X: {l2_error_x:.2e}, L2 Error Y: {l2_error_y:.2e}")
        print(f"Mean Abs Error X: {mean_abs_error_x:.2e}, Mean Abs Error Y: {mean_abs_error_y:.2e}")
        print(f"Max Abs Error X: {max_abs_error_x:.2e}, Max Abs Error Y: {max_abs_error_y:.2e}")
        print(f"Total Loss: {total_loss:.6e}")
        print("-" * 60)
        
        return total_loss

    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        return 1e10
    
##########################################################################################################
def plot_absolute_error(surrogate_results_file, experimental_data_file):
    """Plot absolute error between surrogate predictions and experimental data with improved interpolation"""
    # Load the data
    surrogate_df = pd.read_csv(surrogate_results_file)
    exp_df = pd.read_csv(experimental_data_file)

    # Convert experimental data from mm to m if needed
    exp_df['x'] = exp_df['x-coordinate [mm]'].values/1000
    exp_df['y'] = exp_df['y-coordinate [mm]'].values/1000
    exp_df['ux'] = exp_df['x-displacement [mm]'].values/1000
    exp_df['uy'] = exp_df['y-displacement [mm]'].values/1000

    # Define a regular grid for visualization
    xi = np.linspace(min(surrogate_df['x-coordinate [m]']), max(surrogate_df['x-coordinate [m]']), 100)
    yi = np.linspace(min(surrogate_df['y-coordinate [m]']), max(surrogate_df['y-coordinate [m]']), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Create points for interpolation
    surr_points = np.column_stack((surrogate_df['x-coordinate [m]'], surrogate_df['y-coordinate [m]']))
    surr_ux = surrogate_df['x-displacement [m]'].values
    surr_uy = surrogate_df['y-displacement [m]'].values
    
    exp_points = np.column_stack((exp_df['x'], exp_df['y']))
    exp_ux = exp_df['ux']
    exp_uy = exp_df['uy']
    
    # Interpolate both datasets to the regular grid
    from scipy.interpolate import griddata
    
    # Use 'cubic' or 'linear' for smoother interpolation, 'nearest' to avoid NaNs
    method = 'cubic'
    
    # Interpolate surrogate data to grid
    surr_ux_grid = griddata(surr_points, surr_ux, (xi, yi), method=method)
    surr_uy_grid = griddata(surr_points, surr_uy, (xi, yi), method=method)
    
    # Interpolate experimental data to grid
    exp_ux_grid = griddata(exp_points, exp_ux, (xi, yi), method=method)
    exp_uy_grid = griddata(exp_points, exp_uy, (xi, yi), method=method)
    
    # Calculate error fields
    error_field_ux = np.abs(surr_ux_grid - exp_ux_grid)
    error_field_uy = np.abs(surr_uy_grid - exp_uy_grid)
    
    # Create mask for the hole
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3
    hole_mask = ((xi - center_x)**2 + (yi - center_y)**2) < radius**2
    
    # Apply mask
    error_field_ux = np.ma.masked_where(hole_mask | np.isnan(error_field_ux), error_field_ux)
    error_field_uy = np.ma.masked_where(hole_mask | np.isnan(error_field_uy), error_field_uy)
    
    # Print mean errors (excluding NaN values)
    print(f"The mean error for displacement in x is {np.nanmean(error_field_ux):.2e} m")
    print(f"The mean error for displacement in y is {np.nanmean(error_field_uy):.2e} m")
    
    # Create plot
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    
    # Plot X displacement error
    contour_x = ax[0].pcolormesh(xi, yi, error_field_ux, cmap='turbo', shading='auto')
    cbar_x = fig.colorbar(contour_x, ax=ax[0])
    cbar_x.set_label(label='Absolute Error X (m)', size='18')
    cbar_x.ax.tick_params(labelsize=18)
    ax[0].set_title('Absolute Error in X Displacement', size='20')
    ax[0].set_xlabel('X Coordinate [m]', size='18')
    ax[0].set_ylabel('Y Coordinate [m]', size='18')
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].grid(True)
    
    # Plot Y displacement error
    contour_y = ax[1].pcolormesh(xi, yi, error_field_uy, cmap='turbo', shading='auto')
    cbar_y = fig.colorbar(contour_y, ax=ax[1])
    cbar_y.set_label(label='Absolute Error Y (m)', size='18')
    cbar_y.ax.tick_params(labelsize=18)
    ax[1].set_title('Absolute Error in Y Displacement', size='20')
    ax[1].set_xlabel('X Coordinate [m]', size='18')
    ax[1].set_ylabel('Y Coordinate [m]', size='18')
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)
    
    # Add hole outline (rather than filled)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    ax[0].plot(circle_x, circle_y, 'k-', lw=2)
    ax[1].plot(circle_x, circle_y, 'k-', lw=2)
    
    plt.tight_layout()
    plt.savefig('surrogate_error.pdf', format="pdf", bbox_inches='tight', dpi=300)
    plt.close()

# Main execution
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Paths
    training_folder = r"C:\Users\z004wesy\Desktop\surrogate_modelling\snapshots"
    combined_data_file = "combined_snapshots.csv"
    test_folder = r"C:\Users\z004wesy\Desktop\surrogate_modelling\test_snapshot"
    model_folder = "model_testing"  # New folder for storing/loading models
    
    # Create model folder if it doesn't exist
    os.makedirs(model_folder, exist_ok=True)

    # Combine snapshots if the combined file doesn't exist
    if not os.path.exists(combined_data_file):
        combine_snapshots_to_csv(training_folder, combined_data_file)
    
    # Load and normalize data
    X_normalized, Y_normalized, norm_params = load_and_normalize_data(combined_data_file)
    
    # Convert to PyTorch tensors
    train_x = torch.tensor(X_normalized, dtype=torch.float32)
    train_y = torch.tensor(Y_normalized, dtype=torch.float32)
    train_y_ux = train_y[:, 0] 
    train_y_uy = train_y[:, 1]
    
    # Define model file paths
    model_ux_path = os.path.join(model_folder, 'model_ux.pth')
    model_uy_path = os.path.join(model_folder, 'model_uy.pth')
    likelihood_ux_path = os.path.join(model_folder, 'likelihood_ux.pth')
    likelihood_uy_path = os.path.join(model_folder, 'likelihood_uy.pth')
    history_path = os.path.join(model_folder, 'training_history.pkl')
    norm_params_path = os.path.join(model_folder, 'norm_params.pkl')
    
    # Check if models exist in model_testing folder, if so, load them
    if (os.path.exists(model_ux_path) and os.path.exists(model_uy_path) and 
        os.path.exists(likelihood_ux_path) and os.path.exists(likelihood_uy_path)):
        print("Loading existing models from model_testing folder...")
        
        model_ux, likelihood = load_model(model_ux_path, likelihood_ux_path, train_x, train_y_ux)
        model_uy, _ = load_model(model_uy_path, likelihood_uy_path, train_x, train_y_uy)
        
        # Load training history if available
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history_data = pickle.load(f)
                loss_history_ux = history_data['loss_ux']
                loss_history_uy = history_data['loss_uy']
                lr_history_ux = history_data['lr_ux']
                lr_history_uy = history_data['lr_uy']
                print("Loaded training history")
        else:
            # Initialize with empty lists if history not available
            loss_history_ux = []
            loss_history_uy = []
            lr_history_ux = []
            lr_history_uy = []
        
        # Load normalization parameters if available
        if os.path.exists(norm_params_path):
            with open(norm_params_path, 'rb') as f:
                norm_params = pickle.load(f)
                print("Loaded normalization parameters")
    else:
        print("No existing models found. Training new models...")
        
        # Train models and get histories
        model_ux, likelihood, loss_history_ux, lr_history_ux = train_model(train_x, train_y_ux)
        print('################### Done Training for u_x ###########################')
        model_uy, likelihood, loss_history_uy, lr_history_uy = train_model(train_x, train_y_uy)
        print('################### Done Training for u_y ###########################')

        # Save models to the model_testing folder
        print(f"Saving models to {model_folder}...")
        torch.save(model_ux.state_dict(), model_ux_path)
        torch.save(model_uy.state_dict(), model_uy_path)
        torch.save(likelihood.state_dict(), likelihood_ux_path)
        torch.save(likelihood.state_dict(), likelihood_uy_path)
        
        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump({
                'loss_ux': loss_history_ux,
                'loss_uy': loss_history_uy,
                'lr_ux': lr_history_ux, 
                'lr_uy': lr_history_uy
            }, f)
        
        # Save normalization parameters
        with open(norm_params_path, 'wb') as f:
            pickle.dump(norm_params, f)

    # Plot training losses and learning rates (if we have history data)
    if loss_history_ux and loss_history_uy:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot losses
        ax1.plot(loss_history_ux, label='ux model')
        ax1.plot(loss_history_uy, label='uy model')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss History')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rates
        ax2.plot(lr_history_ux, label='ux model')
        ax2.plot(lr_history_uy, label='uy model')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig('training_history.pdf')
        plt.close()
        
    ########### Prediction phase ############
    # Test values
    test_G = 7.17e10  # 7.35e^10 Pa
    test_K = 8.91e10  # 9.17e^10 Pa

    # Load test data with full coordinates
    fem_file = f"snapshot_G{test_G:.2e}_K{test_K:.2e}.csv"
    fem_path = os.path.join(test_folder, fem_file)
    fem_data = pd.read_csv(fem_path)
    
    # Get full coordinates
    full_coordinates = np.column_stack((
        fem_data['x-coordinate [m]'].values,
        fem_data['y-coordinate [m]'].values
    ))

    # Make predictions on full field
    print("Making predictions...")
    ux_pred, uy_pred, ux_std, uy_std = predict_field(
        model_ux, model_uy, likelihood, 
        test_G, test_K, 
        full_coordinates,
        norm_params
    )

    if ux_pred is not None:
        # Create DataFrame with predictions using full coordinates
        results_df = pd.DataFrame({
            'x-coordinate [m]': full_coordinates[:, 0],
            'y-coordinate [m]': full_coordinates[:, 1],
            'x-displacement [m]': ux_pred,
            'y-displacement [m]': uy_pred,
            'ux_std': ux_std,
            'uy_std': uy_std
        })

        # Save predictions to CSV
        output_file = f'predictions_G{test_G:.2e}_K{test_K:.2e}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        # Load FEM solution for comparison
        fem_file = f"snapshot_G{test_G:.2e}_K{test_K:.2e}.csv"
        fem_path = os.path.join(test_folder, fem_file)
        
        if os.path.exists(fem_path):
            fem_data = pd.read_csv(fem_path)
            # Calculate relative errors
            rel_error_ux = np.abs(ux_pred - fem_data['x-displacement [m]'].values) / np.abs(fem_data['x-displacement [m]'].values) * 100
            rel_error_uy = np.abs(uy_pred - fem_data['y-displacement [m]'].values) / np.abs(fem_data['y-displacement [m]'].values) * 100
            
            print("\nError Statistics:")
            print(f"Mean relative error ux: {np.mean(rel_error_ux):.2f}%")
            print(f"Mean relative error uy: {np.mean(rel_error_uy):.2f}%")
            print(f"Max relative error ux: {np.max(rel_error_ux):.2f}%")
            print(f"Max relative error uy: {np.max(rel_error_uy):.2f}%")

    ########################### Calculate Loss ###################################
    # Try to load experimental data
    exp_file = r'C:\Users\z004wesy\Desktop\surrogate_modelling\20231116_displacements_interpolated.csv'
    if os.path.exists(exp_file):
        exp_data = pd.read_csv(exp_file)	

        exp_x = exp_data['x-coordinate [mm]'].values/1000  # Convert mm to m
        exp_y = exp_data['y-coordinate [mm]'].values/1000  # Convert mm to m
        exp_ux = exp_data['x-displacement [mm]'].values/1000  # Convert mm to m
        exp_uy = exp_data['y-displacement [mm]'].values/1000  # Convert mm to m

        # Prepare data for optimization
        exp_coords = np.column_stack((exp_x, exp_y))
        exp_disps = np.column_stack((exp_ux, exp_uy))

        # Multiple initial guesses to avoid local minima
        initial_guesses = [
            [6e10, 8e10],  # K > G
            [6.5e10, 9e10],  # K >> G
            [5.5e10, 7e10]   # Different ratio
        ]

        best_result = None
        best_loss = float('inf')

        for i, initial_guess in enumerate(initial_guesses):
            print(f"\nStarting optimization #{i+1} with initial G={initial_guess[0]:.2e}, K={initial_guess[1]:.2e}")
            
            result = minimize(
                fun=lambda x: surrogate_mse_loss(x, model_ux, model_uy, exp_coords, exp_disps, norm_params),
                x0=initial_guess,
                method='Nelder-Mead',
                options={'atol': 1e-15, 'maxiter': 100,'adaptive': True},
                bounds=[(5e10, 7.5e10), (6e10, 1e11)]
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
                print(f"New best solution: G={result.x[0]:.2e}, K={result.x[1]:.2e}, Loss={result.fun:.6e}")

        G_sol, K_sol = best_result.x
        print(f"\nFinal value for Shear Modulus is {G_sol:.2e} & Final value of Bulk Modulus is {K_sol:.2e}")
        print(f"K/G ratio: {K_sol/G_sol:.2f}")
        print(f"Total number of iterations: {best_result.nit}")

        # Save results to model_testing folder
        results_file = os.path.join(model_folder, f'optimal_parameters.csv')
        with open(results_file, 'w') as f:
            f.write(f"G,K\n{G_sol},{K_sol}")
        print(f"Optimal parameters saved to {results_file}")

        print("Saving Displacements with final G and K values for comparison...")
        ux_pred, uy_pred, ux_std, uy_std = predict_field(
            model_ux, model_uy, likelihood, 
            G_sol, K_sol, 
            full_coordinates, 
            norm_params
        )
        
        if ux_pred is not None:
            # Create results dataframe
            results_df = pd.DataFrame({
                'x-coordinate [m]': full_coordinates[:, 0],
                'y-coordinate [m]': full_coordinates[:, 1],
                'x-displacement [m]': ux_pred,
                'y-displacement [m]': uy_pred,
                'ux_std': ux_std,
                'uy_std': uy_std
            })
            
            # Save predictions to CSV
            output_file = f'Final_G{G_sol:.2e}_K{K_sol:.2e}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

        print("Plotting error fields...")
        exp_file_raw = r'C:\Users\z004wesy\Desktop\surrogate_modelling\20231116_displacements_raw.csv'
        if os.path.exists(exp_file_raw):
            plot_absolute_error(
                surrogate_results_file=f'Final_G{G_sol:.2e}_K{K_sol:.2e}.csv',
                experimental_data_file=exp_file_raw
            )
        else:
            print(f"Error: Raw experimental data file not found")
    else:
        print(f"Error: Experimental data file not found: {exp_file}")