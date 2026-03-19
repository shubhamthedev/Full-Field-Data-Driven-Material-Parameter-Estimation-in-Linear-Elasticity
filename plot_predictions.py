import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(predictions_file):
    # Load predictions
    df = pd.read_csv(predictions_file)
    x = df['x-coordinate [m]']
    y = df['y-coordinate [m]']
    ux = df['x-displacement [m]']
    uy = df['y-displacement [m]']

    # Plotting the mesh
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Add hole
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)

    # Subplot for displacement_x
    contour_x = ax[0].tricontourf(x, y, ux, levels=100, cmap='turbo')  # Removed c=
    cbar_x = fig.colorbar(contour_x, ax=ax[0])
    cbar_x.set_label(label='Displacement X (m)', size='18')
    cbar_x.ax.tick_params(labelsize=18)
    ax[0].set_title('Computed X Displacement(Surrogate)',size='20')
    ax[0].set_xlabel('X Coordinate [m]',size='18')
    ax[0].set_ylabel('Y Coordinate [m]', size='18')
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].grid(True)
    # Add hole
    ax[0].fill(circle_x, circle_y, 'w-', lw=2)

    # Subplot for Y displacement
    contour_y = ax[1].tricontourf(x, y, uy, levels=100, cmap='turbo')  # Removed c=
    cbar_y = fig.colorbar(contour_y, ax=ax[1])
    cbar_y.set_label(label='Displacement Y (m)', size='18')
    cbar_y.ax.tick_params(labelsize=18)
    ax[1].set_title('Computed Y Displacement(Surrogate)', size='20')
    ax[1].set_xlabel('X Coordinate [m]', size='18')
    ax[1].set_ylabel('Y Coordinate [m]', size='18')
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)
    # Add hole
    ax[1].fill(circle_x, circle_y, 'w-', lw=2)

    plt.tight_layout()
    plt.savefig('gp_predictions_displacement_field.pdf', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Replace these values with your actual test values
    test_G = 6.35e10
    test_K = 8.90e10
    
    # predictions_file = f'predictions_G{test_G:.2e}_K{test_K:.2e}.csv'
    predictions_file = f'Final_G{test_G:.2e}_K{test_K:.2e}.csv'
    plot_predictions(predictions_file)
    print(f"Plots saved as gp_predictions_displacement_field.pdf")