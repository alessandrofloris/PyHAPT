import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def analyze_distributions(mode="train", data_dir="../data/output/"):
    print(f"--- Loading Data for {mode} set ---")
    
    # Path configuration
    crowd_path = os.path.join(data_dir, f"{mode}_crowd_features.npy")
    label_path = os.path.join(data_dir, f"{mode}_label.pkl")
    
    # 1. Data Loading
    try:
        crowd_feat = np.load(crowd_path) # (N, 300, 3)
        with open(label_path, 'rb') as f:
            temp_labels = pickle.load(f)
            
        # Tuple Management & Label Flattening
        if isinstance(temp_labels, tuple) and len(temp_labels) == 4:
            raw_labels = temp_labels[2] # Index 2 is id_action/label
        elif isinstance(temp_labels, dict):
            raw_labels = temp_labels.get('id_action', temp_labels.get('label'))
        else:
            print("Label format not recognized.")
            return
            
        # Clean labels: handles cases where labels might be wrapped in lists [1] -> 1
        # This prevents the "unhashable type: list" error in Seaborn
        clean_labels = [int(l[0] if isinstance(l, (list, np.ndarray)) else l) for l in raw_labels]

    except FileNotFoundError:
        print(f"Files not found at {data_dir}. Please check your paths.")
        return
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return

    # Prepare DataFrame for easy plotting
    # Calculate the temporal mean for each sample to analyze per-action biases
    # Shape crowd_feat: (N, 300, 3) -> Mean over axis 1 -> (N, 3)
    means_per_sample = crowd_feat.mean(axis=1)
    
    df = pd.DataFrame({
        'Action_ID': clean_labels,
        'Avg_Area': means_per_sample[:, 0],
        'Avg_Visibility': means_per_sample[:, 1],
        'Avg_Motion': means_per_sample[:, 2]
    })

    # --- PLOT 1: Global Histograms (Feature Quality) ---
    print("Generating Global Histograms...")
    plt.figure(figsize=(18, 5))
    
    # Area
    plt.subplot(1, 3, 1)
    sns.histplot(df['Avg_Area'], bins=30, kde=True, color='skyblue')
    plt.title('BBox Area Distribution (Mean per Sample)')
    plt.xlabel('Normalized Area')
    
    # Visibility
    plt.subplot(1, 3, 2)
    sns.histplot(df['Avg_Visibility'], bins=30, kde=True, color='orange')
    plt.axvline(0.2, color='red', linestyle='--', label='Noise Threshold (0.2)')
    plt.title('Visibility Distribution (Mean per Sample)')
    plt.legend()
    
    # Motion
    plt.subplot(1, 3, 3)
    sns.histplot(df['Avg_Motion'], bins=30, kde=True, color='green')
    plt.title('Motion Proxy Distribution (Mean per Sample)')
    plt.xlabel('Normalized Motion')
    
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Class Balancing ---
    print("Generating Class Balance Plot...")
    plt.figure(figsize=(10, 6))
    class_counts = df['Action_ID'].value_counts().sort_index()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title(f'Number of Samples per Class ({mode})')
    plt.xlabel('Action ID')
    plt.ylabel('Count')
    plt.show()

    # --- PLOT 3: Bias Check (Area vs Action) ---
    print("Generating Bias Analysis (Area vs Action)...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Action_ID', y='Avg_Area', data=df, palette="coolwarm")
    plt.title('Bias Check: Subject Size per Class')
    plt.ylabel('Normalized BBox Area')
    plt.xlabel('Action ID')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Textual Report ---
    print("\n--- QUICK REPORT ---")
    low_vis_count = (df['Avg_Visibility'] < 0.2).sum()
    print(f"Samples with mean visibility < 0.2: {low_vis_count} out of {len(df)} ({low_vis_count/len(df)*100:.1f}%)")
    
    # Bias Check
    mean_areas = df.groupby('Action_ID')['Avg_Area'].mean()
    print("\nMean Area per Class:")
    print(mean_areas)
    
    if (mean_areas.max() - mean_areas.min()) > 0.1: # Threshold example
        print("⚠️ WARNING: Significant scale difference detected between classes!")
        print(f"Smallest class: {mean_areas.idxmin()} ({mean_areas.min():.4f})")
        print(f"Largest class:  {mean_areas.idxmax()} ({mean_areas.max():.4f})")
    else:
        print("✅ Subject sizes appear balanced across classes.")

if __name__ == "__main__":
    analyze_distributions(mode="train", data_dir="../data/output/")