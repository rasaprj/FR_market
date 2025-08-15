import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_regulation(csv_file_path):
    """
    Plot frequency regulation data for 5 continuous days
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing frequency regulation data
    """
    
    # Read CSV file without headers since data doesn't have proper column names
    print("Loading frequency regulation data...")
    df = pd.read_csv(csv_file_path, header=None)
    
    print(f"Data shape: {df.shape}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Number of rows: {df.shape[0]}")
    
    # Prepare data for 5 continuous days
    hours_list = []
    values_list = []
    
    # Combine all 5 days into one continuous timeline
    for day in range(5):  # First 5 days (columns 0-4)
        for row in range(len(df)):
            # Calculate continuous hours across all 5 days
            day_hours = (row * 2) / 3600  # Hours within this day (2-second intervals)
            total_hours = day * 24 + day_hours  # Total hours from start
            
            # Only include data up to 24 hours per day
            if day_hours <= 24:
                hours_list.append(total_hours)
                values_list.append(df.iloc[row, day])  # Get data from the corresponding day column
    
    print(f"Total data points: {len(hours_list):,}")
    
    # Create the plot with even wider and narrower aspect ratio
    plt.figure(figsize=(24, 5))  # Even wider and narrower (24:5 ratio)
    
    # Plot the continuous frequency regulation data
    plt.plot(hours_list, values_list, color='#2563eb', linewidth=1.5)
    
    # Set up the axes
    plt.xlim(0, 120)  # 0 to 120 hours (5 * 24)
    plt.ylim(-1, 1)   # Fixed Y-axis range
    
    # Set X-axis ticks at day boundaries with Times New Roman font and bigger size
    plt.xticks([0, 24, 48, 72, 96, 120], ['0', '24', '48', '72', '96', '120'], 
               fontfamily='Times New Roman', fontsize=18)
    
    # Set Y-axis ticks at specific values with Times New Roman font and bigger size
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontfamily='Times New Roman', fontsize=18)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    
    # Labels and title with bigger fonts
    plt.xlabel('Time (hour)', fontsize=20, color='black', fontfamily='Times New Roman')
    plt.ylabel('FR signal', fontsize=20, color='black', fontfamily='Times New Roman')
    
    # Adjust layout to prevent text cutoff and add more space for Y-axis label
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.08)
    
    # Show the plot
    plt.show()
    
    return hours_list, values_list

def analyze_data_structure(csv_file_path):
    """
    Analyze the structure of the frequency regulation data
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    
    # Read and examine the data structure
    df = pd.read_csv(csv_file_path, header=None)
    
    print("=== Data Structure Analysis ===")
    print(f"Total columns: {df.shape[1]}")
    print(f"Total rows: {df.shape[0]}")
    print(f"Expected rows for 24h at 2s intervals: {24 * 60 * 60 / 2}")
    
    print("\nFirst 5 rows, first 10 columns:")
    print(df.iloc[:5, :10])
    
    print("\nData types:")
    print(df.dtypes.head())
    
    print("\nBasic statistics for first 5 columns (days):")
    print(df.iloc[:, :5].describe())
    
    return df

# Example usage:
if __name__ == "__main__":
    # Your specific file path
    csv_file = r"C:\Users\ASUS\Desktop\My Hell\FR\01_2017_Dynamic.csv"
    
    try:
        # Analyze data structure first
        df = analyze_data_structure(csv_file)
        
        # Create the visualization
        hours, values = plot_frequency_regulation(csv_file)
        
        print(f"\nVisualization completed successfully!")
        print(f"Plotted {len(values):,} data points over {max(hours):.1f} hours")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")

# Additional utility function for interactive analysis
def get_day_statistics(csv_file_path):
    """
    Get statistics for each of the first 5 days
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    
    df = pd.read_csv(csv_file_path, header=None)
    
    print("=== Daily Statistics ===")
    for day in range(5):
        day_data = df.iloc[:, day]
        print(f"\nDay {day + 1}:")
        print(f"  Mean: {day_data.mean():.6f}")
        print(f"  Std:  {day_data.std():.6f}")
        print(f"  Min:  {day_data.min():.6f}")
        print(f"  Max:  {day_data.max():.6f}")
        print(f"  Range: {day_data.max() - day_data.min():.6f}")

# Function to save the plot
def save_frequency_plot(csv_file_path, output_path="frequency_regulation_plot.png", dpi=300):
    """
    Save the frequency regulation plot to a file
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    output_path (str): Path where to save the plot
    dpi (int): Resolution for the saved image
    """
    
    # Read and process data
    df = pd.read_csv(csv_file_path, header=None)
    
    hours_list = []
    values_list = []
    
    for day in range(5):
        for row in range(len(df)):
            day_hours = (row * 2) / 3600
            total_hours = day * 24 + day_hours
            
            if day_hours <= 24:
                hours_list.append(total_hours)
                values_list.append(df.iloc[row, day])
    
    # Create and save plot
    plt.figure(figsize=(16, 9))
    plt.plot(hours_list, values_list, color='#2563eb', linewidth=1.5)
    
    plt.xlim(0, 120)
    plt.ylim(-1, 1)
    plt.xticks([0, 24, 48, 72, 96, 120], ['0', '24', '48', '72', '96', '120'])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    
    plt.xlabel('Time (hour)', fontsize=20, color='black')
    plt.ylabel('FR signal', fontsize=20, color='black')
    plt.title('FR signal vs Time', 
              fontsize=16, fontweight='bold', color='#374151', pad=20)
    
    plt.figtext(0.5, 0.02, 
                f'Continuous frequency regulation data over 5 days (120 hours total) • '
                f'Sample rate: 2 seconds • Total data points: {len(hours_list):,}',
                ha='center', fontsize=10, color='#6b7280')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")