import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_fr_price_first_240_hours(csv_file_path):
    """
    Plot FR incentive price data for the first 240 hours
    with a square figure shape (independent x/y scales).
    """
    print("Loading FR incentive price data...")
    df = pd.read_csv(csv_file_path, header=None)
    print(f"Data shape: {df.shape} (expected 365 rows × 24 columns)")

    # Scale values
    df = df * 1000

    # First 10 days -> 240 hours
    first_10_days = df.iloc[0:10, :]
    hours = list(range(240))
    hourly_prices = first_10_days.to_numpy().reshape(-1)[:240]

    # Build plot with square window
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.step(hours, hourly_prices, where='post',
            linewidth=1.5, alpha=0.8, color='#1f77b4')
    #ax.plot(hours, hourly_prices, 
    #    marker='o', markersize=0, 
    #    linestyle='-', linewidth=1.0, 
    #    color='#1f77b4', alpha=0.8)



    # Limits
    ax.set_xlim(0, 240)
    ax.set_ylim(min(hourly_prices), max(hourly_prices))

    # X ticks: 0, 40, 80, 120, 160, 200, 240
    xticks = [0, 40, 80, 120, 160, 200, 240]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontfamily='Times New Roman', fontsize=18)

    # Y ticks style
    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Grid + labels
    ax.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    ax.set_xlabel('Time (hour)', fontsize=24, color='black', fontfamily='Times New Roman')
    ax.set_ylabel('FR Price ($/MW)', fontsize=24, color='black', fontfamily='Times New Roman')

    # Info text
    plt.figtext(0.5, 0.02,
                'Hourly FR incentive prices for first 240 hours (Days 1–10)',
                ha='center', fontsize=10, color='#6b7280', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.12)
    plt.show()

    return hourly_prices


def save_fr_price_plot(csv_file_path, output_path="fr_price_first_240_hours.png", dpi=300):
    """
    Save the FR incentive price plot for the first 240 hours with a square figure shape.
    """
    df = pd.read_csv(csv_file_path, header=None) * 1000
    first_10_days = df.iloc[0:10, :]
    hours = list(range(240))
    hourly_prices = first_10_days.to_numpy().reshape(-1)[:240]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.step(hours, hourly_prices, where='post',
            linewidth=1.5, alpha=0.8, color='#1f77b4')

    ax.set_xlim(0, 240)
    ax.set_ylim(min(hourly_prices), max(hourly_prices))

    xticks = [0, 40, 80, 120, 160, 200, 240]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontfamily='Times New Roman', fontsize=18)

    ax.tick_params(axis='y', labelsize=18)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.grid(True, linestyle='--', alpha=0.7, color='#e0e0e0')
    ax.set_xlabel('Time (hour)', fontsize=24, color='black', fontfamily='Times New Roman')
    ax.set_ylabel('FR Price ($/MW)', fontsize=24, color='black', fontfamily='Times New Roman')

    plt.figtext(0.5, 0.02,
                'Hourly FR incentive prices for first 240 hours (Days 1–10)',
                ha='center', fontsize=10, color='#6b7280', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.12)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    csv_file = r"C:\Users\ASUS\Desktop\My Hell\FR\FR_incentive.csv"
    try:
        plot_fr_price_first_240_hours(csv_file)
        # save_fr_price_plot(csv_file)  # uncomment to save
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")
