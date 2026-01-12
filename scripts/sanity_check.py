import numpy as np
import pickle
import os

def run_sanity_check(mode="train"):
    print(f"--- Starting Sanity Check for {mode} set ---")
    
    # Paths
    joints_path = f"../data/output/{mode}_data_joint.npy"
    bbox_path = f"../data/output/{mode}_bbox.npy"
    crowd_path = f"../data/output/{mode}_crowd_features.npy"
    label_path = f"../data/output/{mode}_label.pkl"

    # 1. Data loading
    try:
        joints = np.load(joints_path)
        bboxes = np.load(bbox_path)
        crowd = np.load(crowd_path)
        with open(label_path, 'rb') as f:
            temp_labels = pickle.load(f)
        # Dictionary reconstruction
        if isinstance(temp_labels, tuple) and len(temp_labels) == 4:
            labels = {
                'sample_name': temp_labels[0],
                'label': temp_labels[1],
                'frame': temp_labels[2],
                'video_path': temp_labels[3]
            }
        else:
            labels = temp_labels
            
    except Exception as e:
        print(f"Error during loading: {e}")
        return

    # 2. Dimensionality check (N, C, T, V, M)
    print(f"\n[1] Shape Check:")
    print(f" - Joints: {joints.shape} (N, C, T, V, M)")
    print(f" - BBoxes: {bboxes.shape} (N, T, 4)")
    print(f" - Crowd:  {crowd.shape} (N, T, 3)")
    
    N = joints.shape[0]
    if not (N == bboxes.shape[0] == crowd.shape[0] == len(labels["label"])):
        print("WARNING: Misalignment in the number of samples (N)!")
    else:
        print(f"Consistent number of samples: N = {N}")

    # 3. Joint analysis (x, y, score)
    print(f"\n[2] Joint Analysis (C=3):")
    x_coords = joints[:, 0, :, :, :]
    y_coords = joints[:, 1, :, :, :]
    scores = joints[:, 2, :, :, :]
    
    print(f" - X range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    print(f" - Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
    print(f" - Score mean: {scores.mean():.3f}")
    print(f" - Interpolated joints (score=0.15): {np.sum(np.isclose(scores, 0.15)) / scores.size * 100:.1f}%")

    # 4. Crowd features analysis [Area, Visibility, Motion]
    print(f"\n[3] Crowd Features Analysis (Vector o):")
    area = crowd[:, :, 0]
    vis = crowd[:, :, 1]
    motion = crowd[:, :, 2]
    
    print(f" - Area BBox (norm):  min={area.min():.4f}, max={area.max():.4f}, mean={area.mean():.4f}")
    print(f" - Visibility (score): min={vis.min():.4f}, max={vis.max():.4f}, mean={vis.mean():.4f}")
    print(f" - Motion Proxy:      min={motion.min():.4f}, max={motion.max():.4f}, mean={motion.mean():.4f}")
    
    if motion.max() > 1.0:
        print("WARNING: Motion Proxy appears to be out of range (>1.0). Check global normalization.")
    if np.any(np.isnan(crowd)):
        print("WARNING: Found NaN values in crowd features!")

    # 5. Check Label Metadata
    print(f"\n[4] Metadata Check:")
    keys = labels.keys()
    print(f" - Keys in label.pkl: {list(keys)}")

if __name__ == "__main__":
    run_sanity_check("train")