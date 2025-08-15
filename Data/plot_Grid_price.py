import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def plot_dam_price_seasonal_240(csv_file_path):
    """
    Plot DAM price data for seasonal periods (10 days = 240 hourly data points each)
    using step lines. Produces a square-shaped figure (independent x/y scales).
    """
    print("Loading DAM price data...")
    df = pd.read_csv(csv_file_path, header=None)
    print(f"Data shape: {df.shape} (expected 365 rows × 24 columns)")

    # Scale values
    df = df * 1000

    # Define seasonal periods (10 days each)
    seasons = {
        'Winter': {'start': 0,   'end': 9,   'color': '#1f77b4'},  # Days 1–10
        'Summer': {'start': 180, 'end': 189, 'color': '#ff7f0e'}   # Days 181–190
    }

    # Precompute global y-limits across all plotted seasons (so scales match)
    y_min, y_max = np.inf, -np.inf
    for s in seasons.values():
        block = df.iloc[s['start']:s['end'] + 1, :].to_numpy().reshape(-1)[:240]
        y_min = min(y_min, float(np.min(block)))
        y_max = max(y_max, float(np.max(block)))

    # Square figure (not equal aspect)
    fig, ax = plt.subplots(figsize=(8, 8))

    for season_name, season_info in seasons.items():
        # Extract 10 days -> 240 hours
        seasonal_data = df.iloc[season_info['start']:season_info['end'] + 1, :]
        hours = list(range(240))
        hourly_prices = seasonal_data.to_numpy().reshape(-1)[:240]

        # Step plot (keep step style)
        ax.step(hours, hourly_prices,
                color=season_info['color'],
                linewidth=1.5,
                alpha=0.8,
                where='post',
                label=season_name)

    # Axes limits and ticks
    ax.set_xlim(0, 240)
    ax.set_ylim(y_min, y_max)

    xticks = [0, 40, 80, 120, 160, 200, 240]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontfamily='Times New Roman', fontsize=18)

    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Grid + labels
    ax.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    ax.set_xlabel('Time (hour)', fontsize=24, color='black', fontfamily='Times New Roman')
    ax.set_ylabel('DAM Price ($/MWh)', fontsize=24, color='black', fontfamily='Times New Roman')

    # Legend
    font_prop = font_manager.FontProperties(family='Times New Roman', size=14)
    ax.legend(loc='upper right', prop=font_prop)

    # Info text
    plt.figtext(0.5, 0.02,
                'Hourly DAM prices for Winter (Days 1–10) and Summer (Days 181–190) • 240 hours each season',
                ha='center', fontsize=10, color='#6b7280', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.12)
    plt.show()

    return seasons


def analyze_dam_data_structure(csv_file_path):
    """
    Analyze the structure of the DAM price data.
    """
    df = pd.read_csv(csv_file_path, header=None)
    print("=== DAM Price Data Structure Analysis ===")
    print(f"Total rows (days): {df.shape[0]}")
    print(f"Total columns (hours): {df.shape[1]}")
    print("Expected: 365 rows × 24 columns")

    print("\nFirst 5 rows, first 12 columns:")
    print(df.iloc[:5, :12])

    print("\nOriginal data statistics (before multiplying by 1000):")
    print(df.describe())

    df_scaled = df * 1000
    print("\nScaled data statistics (after multiplying by 1000):")
    print(df_scaled.describe())

    return df


def get_seasonal_statistics_240(csv_file_path):
    """
    Get statistics for each seasonal period (10 days each = 240 hours).
    """
    df = pd.read_csv(csv_file_path, header=None) * 1000

    seasons = {
        'Winter': {'start': 0,   'end': 9},
        'Summer': {'start': 180, 'end': 189}
    }

    print("=== Seasonal DAM Price Statistics (10 days / 240 hours each) ===")
    for season_name, season_info in seasons.items():
        seasonal_data = df.iloc[season_info['start']:season_info['end'] + 1, :]
        hourly_prices = seasonal_data.to_numpy().reshape(-1)[:240]

        print(f"\n{season_name} (Days {season_info['start'] + 1}-{season_info['end'] + 1}):")
        print(f"  Total hourly data points: {len(hourly_prices)}")
        print(f"  Mean hourly price: ${np.mean(hourly_prices):.2f}/MWh")
        print(f"  Std hourly price:  ${np.std(hourly_prices):.2f}/MWh")
        print(f"  Min hourly price:  ${np.min(hourly_prices):.2f}/MWh")
        print(f"  Max hourly price:  ${np.max(hourly_prices):.2f}/MWh")
        print(f"  Price range:       ${np.max(hourly_prices) - np.min(hourly_prices):.2f}/MWh")


def save_dam_price_plot_240(csv_file_path, output_path="dam_price_seasonal_240.png", dpi=300):
    """
    Save the 240-hour seasonal DAM price plot to a file (square figure, step lines).
    """
    df = pd.read_csv(csv_file_path, header=None) * 1000

    seasons = {
        'Winter': {'start': 0,   'end': 9,   'color': '#1f77b4'},
        'Summer': {'start': 180, 'end': 189, 'color': '#ff7f0e'}
    }

    # Compute global y-lims
    y_min, y_max = np.inf, -np.inf
    for s in seasons.values():
        block = df.iloc[s['start']:s['end'] + 1, :].to_numpy().reshape(-1)[:240]
        y_min = min(y_min, float(np.min(block)))
        y_max = max(y_max, float(np.max(block)))

    fig, ax = plt.subplots(figsize=(8, 8))  # square figure

    for season_name, season_info in seasons.items():
        seasonal_data = df.iloc[season_info['start']:season_info['end'] + 1, :]
        hours = list(range(240))
        hourly_prices = seasonal_data.to_numpy().reshape(-1)[:240]

        ax.step(hours, hourly_prices,
                color=season_info['color'],
                linewidth=1.5,
                alpha=0.8,
                where='post',
                label=season_name)

    ax.set_xlim(0, 240)
    ax.set_ylim(y_min, y_max)

    xticks = [0, 40, 80, 120, 160, 200, 240]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontfamily='Times New Roman', fontsize=18)

    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    ax.set_xlabel('Time (hour)', fontsize=24, color='black', fontfamily='Times New Roman')
    ax.set_ylabel('DAM Price ($/MWh)', fontsize=24, color='black', fontfamily='Times New Roman')

    font_prop = font_manager.FontProperties(family='Times New Roman', size=14)
    ax.legend(loc='upper right', prop=font_prop)

    plt.figtext(0.5, 0.02,
                'Hourly DAM prices for Winter (Days 1–10) and Summer (Days 181–190) • 240 hours each season',
                ha='center', fontsize=10, color='#6b7280', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.12)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {output_path}")


# Example usage:
if __name__ == "__main__":
    # Update to your path
    csv_file = r"C:\Users\ASUS\Desktop\My Hell\FR\slow_Price.csv"

    try:
        analyze_dam_data_structure(csv_file)
        get_seasonal_statistics_240(csv_file)
        plot_dam_price_seasonal_240(csv_file)
        # save_dam_price_plot_240(csv_file)  # uncomment to save
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        print("Please update the file path in the script.")
    except Exception as e:
        print(f"Error: {e}")
