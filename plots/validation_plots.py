import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

# Configure seaborn aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})


def load_data(csv_path):
    """
    Loads the CSV data into a pandas DataFrame.

    Parameters:
    - csv_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"File {csv_path} not found.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise


def save_plot_directory(directory='plots'):
    """
    Creates a directory for saving plots if it doesn't exist.

    Parameters:
    - directory (str): Name of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory '{directory}' for saving plots.")
    else:
        print(f"Directory '{directory}' already exists.")


def get_class_name(cls):
    """
    Maps numeric class IDs to descriptive class names.

    Parameters:
    - cls (int or float): Numeric class ID.

    Returns:
    - str: Descriptive class name.
    """
    cls_name = ""
    num = round(cls)

    if num == 0:
        cls_name = "blue_cone"
    elif num == 1:
        cls_name = "yellow_cone"
    elif num == 2:
        cls_name = "orange_cone"
    elif num == 3:
        cls_name = "big_orange_cone"
    elif num == 4:
        cls_name = "knocked_over_cone"
    else:
        cls_name = f"unknown_class_{num}"

    return cls_name


def plot_actual_vs_predicted_distance(df, classes, palette, save_dir='plots'):
    """
    Plots Actual Distance vs. Predicted Distance and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - classes (list): List of unique classes.
    - palette (dict): Mapping of class names to colors.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Actual_Distance',
        y='Predicted_Distance',
        hue='class_name',  # Use class_name for the legend
        palette=palette,
        alpha=0.7
    )
    plt.plot([df['Actual_Distance'].min(), df['Actual_Distance'].max()],
             [df['Actual_Distance'].min(), df['Actual_Distance'].max()],
             color='red', linestyle='--', label='Ideal')
    plt.title('Actual Distance vs. Predicted Distance')
    plt.xlabel('Actual Distance (m)')
    plt.ylabel('Predicted Distance (m)')
    plt.legend(title='Class')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'actual_vs_predicted_distance.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_actual_vs_predicted_angle(df, classes, palette, save_dir='plots'):
    """
    Plots Actual Angle vs. Predicted Angle and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - classes (list): List of unique classes.
    - palette (dict): Mapping of class names to colors.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Actual_Angle',
        y='Predicted_Angle',
        hue='class_name',  # Use class_name for the legend
        palette=palette,
        alpha=0.7
    )
    plt.plot([df['Actual_Angle'].min(), df['Actual_Angle'].max()],
             [df['Actual_Angle'].min(), df['Actual_Angle'].max()],
             color='red', linestyle='--', label='Ideal')
    plt.title('Actual Angle vs. Predicted Angle')
    plt.xlabel('Actual Angle (degrees)')
    plt.ylabel('Predicted Angle (degrees)')
    plt.legend(title='Class')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'actual_vs_predicted_angle.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def calculate_polar_error(row):
    """
    Calculates the Euclidean distance between actual and predicted points in polar coordinates.

    Parameters:
    - row (pd.Series): A row from the DataFrame.

    Returns:
    - float: Euclidean distance error.
    """
    # Extract actual and predicted distances and angles
    r1, theta1 = row['Actual_Distance'], row['Actual_Angle']
    r2, theta2 = row['Predicted_Distance'], row['Predicted_Angle']

    # Convert angles from degrees to radians
    theta1_rad = math.radians(theta1)
    theta2_rad = math.radians(theta2)

    # Convert polar to Cartesian coordinates
    x1, y1 = r1 * math.cos(theta1_rad), r1 * math.sin(theta1_rad)
    x2, y2 = r2 * math.cos(theta2_rad), r2 * math.sin(theta2_rad)

    # Calculate Euclidean distance
    distance_error = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance_error


def plot_error_over_distance(df, classes, palette, save_dir='plots'):
    """
    Plots the Euclidean distance error over Actual Distance and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data with error calculated.
    - classes (list): List of unique classes.
    - palette (dict): Mapping of class names to colors.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Actual_Distance',
        y='Polar_Error',
        hue='class_name',  # Use class_name for the legend
        palette=palette,
        alpha=0.7
    )
    plt.title('Polar Coordinate Error Over Actual Distance')
    plt.xlabel('Actual Distance (m)')
    plt.ylabel('Euclidean Distance Error (m)')
    plt.legend(title='Class')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'polar_error_over_actual_distance.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_error_over_distance_per_class(df, classes, palette, save_dir='plots'):
    """
    Plots the Euclidean distance error over Actual Distance for each class individually and saves each plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data with error calculated.
    - classes (list): List of unique classes.
    - palette (dict): Mapping of class names to colors.
    - save_dir (str): Directory to save the plots.
    """
    for cls in classes:
        name = get_class_name(cls)

        plt.figure(figsize=(10, 6))
        class_df = df[df['class_name'] == name]
        sns.scatterplot(
            data=class_df,
            x='Actual_Distance',
            y='Polar_Error',
            color=palette[name],
            alpha=0.7
        )
        plt.title(f'Polar Coordinate Error Over Actual Distance for Class: {name}')
        plt.xlabel('Actual Distance (m)')
        plt.ylabel('Euclidean Distance Error (m)')
        plt.tight_layout()
        # Replace spaces or special characters in filename if necessary
        safe_name = name.replace(" ", "_")
        filename = os.path.join(save_dir, f'polar_error_over_actual_distance_class_{safe_name}.png')
        plt.savefig(filename)
        print(f"Saved plot for class '{name}': {filename}")
        plt.close()


def plot_roll_over_index(df, save_dir='plots'):
    """
    Plots Roll over Data Frame Index and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['roll'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Roll over Data Frame Number')
    plt.xlabel('Data Frame Index')
    plt.ylabel('Roll')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'roll_over_index.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_pitch_over_index(df, save_dir='plots'):
    """
    Plots Pitch over Data Frame Index and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - save_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['pitch'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Pitch over Data Frame Number')
    plt.xlabel('Data Frame Index')
    plt.ylabel('Pitch')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'pitch_over_index.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()


def plot_average_polar_error_per_meter(df, classes, palette, save_dir='plots'):
    """
    Plots a column chart showing the average Polar Error of each class per 1 meter interval and saves the plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data with Polar_Error calculated.
    - classes (list): List of unique classes.
    - palette (dict): Mapping of class names to colors.
    - save_dir (str): Directory to save the plot.
    """

    # Define 1-meter bins
    bin_width = 2
    max_distance = df['Actual_Distance'].max()
    bins = np.arange(0, math.ceil(max_distance) + bin_width, bin_width)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}m" for i in range(len(bins)-1)]
    df['Distance_Bin'] = pd.cut(df['Actual_Distance'], bins=bins, labels=labels, include_lowest=True)

    # Group by class and distance bin to calculate average Polar_Error
    error_df = df.groupby(['class_name', 'Distance_Bin'])['Polar_Error'].mean().reset_index()

    # Pivot the DataFrame for better plotting
    error_pivot = error_df.pivot(index='Distance_Bin', columns='class_name', values='Polar_Error')

    # Reorder the columns of the pivoted DataFrame to match the classes
    error_pivot = error_pivot[classes]  # Ensure column order matches classes
    error_pivot = error_pivot.reindex(labels)  # Reorder rows based on distance bins

    # Plotting with aligned colors
    plt.figure(figsize=(14, 8))
    colors = [palette[cls] for cls in classes]  # Get colors in the correct order
    error_pivot.plot(kind='bar', ax=plt.gca(), color=colors)

    plt.title(f'Average Polar Error per {bin_width} Meter Interval by Class')
    plt.xlabel(f'Actual Distance ({bin_width}m bins)')
    plt.ylabel('Average Polar Error (m)')
    plt.legend(title='Class')
    plt.tight_layout()
    filename = os.path.join(save_dir, 'average_polar_error_per_meter.png')
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def calculate_mean_accuracy_per_bin(df, classes, bins, labels):
    """
    Calculates and prints the mean accuracy for each bin for each class.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data with Polar_Error calculated.
    - classes (list): List of unique classes.
    - bins (list): Binning intervals for Actual_Distance.
    - labels (list): Labels for each bin.

    Returns:
    - pd.DataFrame: DataFrame containing mean accuracy per bin for each class.
    """
    # Add Distance_Bin column
    df['Distance_Bin'] = pd.cut(df['Actual_Distance'], bins=bins, labels=labels, include_lowest=True)

    # Calculate mean accuracy per bin per class
    accuracy_df = (
        df.groupby(['class_name', 'Distance_Bin'])['Polar_Error']
        .mean()
        .reset_index()
        .rename(columns={'Polar_Error': 'Mean_Accuracy'})
    )

    # Print the mean accuracy for each bin for each class
    print("\nMean accuracy for each bin for each class:")
    for cls in classes:
        class_name = get_class_name(cls)
        print(f"\nClass: {class_name}")
        class_accuracy = accuracy_df[accuracy_df['class_name'] == class_name]
        print(class_accuracy[['Distance_Bin', 'Mean_Accuracy']])

    return accuracy_df


def main():
    # Path to your CSV file
    csv_path = '../nn/dist_net/test_predictions.csv'  # Replace with your actual CSV file path

    # Define the maximum allowable actual distance
    max_distance = 28.0  # Example value; adjust as needed

    # Directory to save plots
    save_dir = 'dist_net'

    # Create plots directory if it doesn't exist
    save_plot_directory(save_dir)

    # Load the data
    df = load_data(csv_path)

    # Ensure that the necessary columns exist
    required_columns = [
        'class', 'Actual_Distance', 'Actual_Angle',
        'Predicted_Distance', 'Predicted_Angle',
        'roll', 'pitch'  # Added 'roll' and 'pitch' to required columns
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV.")

    # Map numeric classes to names
    df['class_name'] = df['class'].apply(get_class_name)

    # Identify unique classes based on class_name to ensure correct palette mapping
    classes = sorted(df['class'].unique())
    class_names = [get_class_name(cls) for cls in classes]

    # Define specific colors for certain classes
    fixed_colors = {
        "blue_cone": "blue",
        "yellow_cone": "yellow",
        "orange_cone": "orange",
        "big_orange_cone": "red",
        "knocked_over_cone": "black",
    }

    # Assign colors to other classes
    other_classes = [cls for cls in class_names if cls not in fixed_colors]
    # Generate distinct colors for other classes
    other_palette = sns.color_palette("hsv", len(other_classes))
    other_palette_dict = {cls: other_palette[i] for i, cls in enumerate(other_classes)}

    # Combine fixed and other palettes
    palette = {**fixed_colors, **other_palette_dict}

    # Debugging: Print the palette
    print("\nAssigned color palette:")
    for cls, color in palette.items():
        print(f"{cls}: {color}")

    # Filter out data points where Actual_Distance exceeds max_distance
    initial_count = len(df)
    df = df[df['Actual_Distance'] <= max_distance]
    filtered_count = len(df)
    excluded_count = initial_count - filtered_count
    print(f"\nFiltered out {excluded_count} data points where Actual_Distance > {max_distance}m")

    if filtered_count == 0:
        print("No data points left after filtering. Exiting.")
        return

    # Reset index after filtering for plotting purposes
    df = df.reset_index(drop=True)

    # Calculate Polar Error and add it to the DataFrame
    print("\nCalculating polar coordinate errors...")
    df['Polar_Error'] = df.apply(calculate_polar_error, axis=1)
    print("Polar coordinate errors calculated.")

    # Plot Actual vs Predicted Distance
    print("\nPlotting Actual vs Predicted Distance...")
    plot_actual_vs_predicted_distance(df, classes, palette, save_dir)

    # Plot Actual vs Predicted Angle
    print("\nPlotting Actual vs Predicted Angle...")
    plot_actual_vs_predicted_angle(df, classes, palette, save_dir)

    # Plot Polar Error Over Actual Distance
    print("\nPlotting Polar Error Over Actual Distance...")
    plot_error_over_distance(df, classes, palette, save_dir)

    # Plot Polar Error Over Actual Distance for Each Class
    print("\nPlotting Polar Error Over Actual Distance for Each Class...")
    plot_error_over_distance_per_class(df, classes, palette, save_dir)

    # Plot Roll over Data Frame Number
    print("\nPlotting Roll over Data Frame Number...")
    plot_roll_over_index(df, save_dir)

    # Plot Pitch over Data Frame Number
    print("\nPlotting Pitch over Data Frame Number...")
    plot_pitch_over_index(df, save_dir)

    # Plot Average Polar Error per 1 Meter Interval by Class
    print("\nPlotting Average Polar Error per 1 Meter Interval by Class...")
    plot_average_polar_error_per_meter(df, class_names, palette, save_dir)

    # Define bins for calculating mean accuracy
    bin_width = 2
    max_distance = df['Actual_Distance'].max()
    bins = np.arange(0, math.ceil(max_distance) + bin_width, bin_width)
    labels = [f"{int(bins[i])}-{int(bins[i + 1])}m" for i in range(len(bins) - 1)]

    # Calculate and print mean accuracy per bin per class
    calculate_mean_accuracy_per_bin(df, classes, bins, labels)

    print("\nAll plots have been generated and saved successfully.")


if __name__ == "__main__":
    main()
