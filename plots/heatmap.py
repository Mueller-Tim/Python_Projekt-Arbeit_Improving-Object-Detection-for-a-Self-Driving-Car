import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import cm

# Parameters
csv_file = "../nn/dist_net/test_predictions.csv"
classes_to_exclude = []  # e.g. [1.0, 3.0] to exclude classes 1 and 3
grid_size = 100  # Number of bins along each axis for the heatmap
x_range = (-18, 18)  # Range for X axis in meters
y_range = (0, 28)    # Range for Y axis in meters

# Read CSV
df = pd.read_csv(csv_file)

# Exclude classes if specified
if classes_to_exclude:
    df = df[~df['class'].isin(classes_to_exclude)]

# Convert angles to radians
df['Actual_Angle_rad'] = np.deg2rad((df['Actual_Angle'] + 180) % 360)
df['Predicted_Angle_rad'] = np.deg2rad((df['Predicted_Angle'] + 180) % 360)

# Polar to Cartesian conversion
df['Actual_X'] = df['Actual_Distance'] * np.sin(df['Actual_Angle_rad'])
df['Actual_Y'] = df['Actual_Distance'] * np.cos(df['Actual_Angle_rad'])

df['Predicted_X'] = df['Predicted_Distance'] * np.sin(df['Predicted_Angle_rad'])
df['Predicted_Y'] = df['Predicted_Distance'] * np.cos(df['Predicted_Angle_rad'])

# Compute error vectors
df['Error_X'] = df['Predicted_X'] - df['Actual_X']
df['Error_Y'] = df['Predicted_Y'] - df['Actual_Y']

# Bin the data
x_bins = np.linspace(x_range[0], x_range[1], grid_size+1)
y_bins = np.linspace(y_range[0], y_range[1], grid_size+1)

df['x_bin'] = np.digitize(df['Actual_X'], x_bins) - 1
df['y_bin'] = np.digitize(df['Actual_Y'], y_bins) - 1

# Filter points outside the bin range
df = df[(df['x_bin'] >= 0) & (df['x_bin'] < grid_size) &
        (df['y_bin'] >= 0) & (df['y_bin'] < grid_size)]

# Initialize arrays for counts and error sums using groupby
grouped = df.groupby(['y_bin', 'x_bin'])

# Aggregate counts and error sums
heatmap_counts_grouped = grouped.size().unstack(fill_value=0)
error_sum_x_grouped = grouped['Error_X'].sum().unstack(fill_value=0)
error_sum_y_grouped = grouped['Error_Y'].sum().unstack(fill_value=0)

# Reindex to ensure all bin indices are present
heatmap_counts_grouped = heatmap_counts_grouped.reindex(index=range(grid_size), columns=range(grid_size), fill_value=0)
error_sum_x_grouped = error_sum_x_grouped.reindex(index=range(grid_size), columns=range(grid_size), fill_value=0)
error_sum_y_grouped = error_sum_y_grouped.reindex(index=range(grid_size), columns=range(grid_size), fill_value=0)

# Convert to NumPy arrays
heatmap_counts = heatmap_counts_grouped.values
error_sum_x = error_sum_x_grouped.values
error_sum_y = error_sum_y_grouped.values

# Debugging: Print shapes to ensure consistency
print(f"heatmap_counts shape: {heatmap_counts.shape}")
print(f"error_sum_x shape: {error_sum_x.shape}")
print(f"error_sum_y shape: {error_sum_y.shape}")

# Compute average error vectors
with np.errstate(divide='ignore', invalid='ignore'):
    avg_error_x = np.where(heatmap_counts > 0, error_sum_x / heatmap_counts, 0)
    avg_error_y = np.where(heatmap_counts > 0, error_sum_y / heatmap_counts, 0)

# Compute average error magnitude
avg_error_magnitude = np.sqrt(avg_error_x**2 + avg_error_y**2)

# Mask bins with no data
avg_error_magnitude_masked = np.where(heatmap_counts > 0, avg_error_magnitude, np.nan)

# Define custom colormap
# 0 to 1: 'viridis' colormap
# >1: red
viridis = plt.get_cmap('viridis', 256)

# Extract colors from 'viridis'
colors = viridis(np.linspace(0, 1, 256))

# Append red color for values >1
colors = np.vstack([colors, [1, 0, 0, 1]])  # RGBA for red

# Create a new colormap that includes 'viridis' and red
custom_cmap = mcolors.LinearSegmentedColormap.from_list('vir_plus_red', colors)

# Define normalization: map 0-1 to 0-1, and values >1 use the 'over' color
norm = mcolors.Normalize(vmin=0, vmax=1)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the average error magnitude heatmap with custom normalization and colormap
im2 = ax.imshow(avg_error_magnitude_masked, origin='lower', extent=x_range + y_range,
                aspect='auto', cmap=custom_cmap, norm=norm, clim=(0,1))

# Set the 'over' color to red
im2.cmap.set_over('red')

# Add colorbar with extend for values >1
cbar2 = plt.colorbar(im2, ax=ax, extend='max')
cbar2.set_label('Average Error Magnitude [m]')

# Create a grid for quiver
X_centers = 0.5*(x_bins[:-1] + x_bins[1:])
Y_centers = 0.5*(y_bins[:-1] + y_bins[1:])
X_grid, Y_grid = np.meshgrid(X_centers, Y_centers)

# Verify shapes
print(f"X_grid shape: {X_grid.shape}")
print(f"Y_grid shape: {Y_grid.shape}")

# Create mask
mask = heatmap_counts > 0

# Verify mask shape
print(f"mask shape: {mask.shape}")

# Plot quiver arrows only where we have data
# To improve visibility, you might want to scale the quivers appropriately
quiver_scale = np.nanmax(avg_error_magnitude) / 1  # Adjust scaling factor as needed

ax.quiver(X_grid[mask],
          Y_grid[mask],
          avg_error_x[mask],
          avg_error_y[mask],
          avg_error_magnitude[mask],  # Use magnitude for coloring arrows if desired
          color='white', angles='xy', scale_units='xy', scale=quiver_scale, width=0.002,
          alpha=0.7, pivot='middle')

# Optional: Add a separate quiver key for reference
# This helps interpret the scale of the arrows
# Adjust the parameters as needed
ax.quiverkey(ax.quiver([], [], [], [], [], color='white'),
             X=0.92, Y=1.05, U=1,
             label='Average Error Vector', labelpos='E')

# Set labels and title
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_title('Heatmap of Average Error Magnitude with Error Directions')

plt.tight_layout()
plt.show()

# Save the combined figure
fig.savefig("combined_heatmap_error.png", dpi=600, bbox_inches='tight')
