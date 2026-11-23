# utils_splits.py — labeled_ratio interpreted as fraction of TOTAL cells
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def create_stratified_splits(
    n_samples: int,
    labels: np.ndarray,
    labeled_ratio: float = 0.05,   # fraction of TOTAL cells (e.g. 0.05 = 5%)
    val_ratio: float = 0.1,        # fraction of TOTAL cells
    test_ratio: float = 0.2,       # fraction of TOTAL cells
    random_state: int = 42,
    min_samples_per_class: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified splits where labeled_ratio/val_ratio/test_ratio are all
    fractions of the FULL dataset (matches scREFINE result tables).
    """

    if labeled_ratio + val_ratio + test_ratio >= 1.0:
        raise ValueError("labeled_ratio + val_ratio + test_ratio must be < 1.0")

    # Filter out rare classes (< min_samples_per_class samples)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    rare_classes = unique_classes[class_counts < min_samples_per_class]

    if len(rare_classes) > 0:
        keep_mask = ~np.isin(labels, rare_classes)
        valid_indices = np.where(keep_mask)[0]
        labels_filtered = labels[keep_mask]
        n_samples_filtered = len(valid_indices)

        def map_back(filtered_indices):
            return valid_indices[filtered_indices]
    else:
        valid_indices = np.arange(n_samples)
        labels_filtered = labels
        n_samples_filtered = n_samples

        def map_back(filtered_indices):
            return filtered_indices

    # Step 1: test set = test_ratio of TOTAL
    idx_temp, idx_test_f = train_test_split(
        np.arange(n_samples_filtered),
        test_size=test_ratio,
        stratify=labels_filtered,
        random_state=random_state,
    )

    # Step 2: validation = val_ratio of TOTAL (adjust within remaining pool)
    val_size_adjusted = val_ratio / (1.0 - test_ratio)
    idx_train_full_f, idx_val_f = train_test_split(
        idx_temp,
        test_size=val_size_adjusted,
        stratify=labels_filtered[idx_temp],
        random_state=random_state + 1,
    )

    # Step 3: labeled = labeled_ratio of TOTAL (adjust within train pool)
    train_pool_frac = 1.0 - test_ratio - val_ratio
    labeled_size_adjusted = labeled_ratio / train_pool_frac
    idx_labeled_f, idx_unlabeled_f = train_test_split(
        idx_train_full_f,
        test_size=1.0 - labeled_size_adjusted,
        stratify=labels_filtered[idx_train_full_f],
        random_state=random_state + 2,
    )

    idx_train_labeled = map_back(idx_labeled_f)
    idx_train_unlabeled = map_back(idx_unlabeled_f)
    idx_val = map_back(idx_val_f)
    idx_test = map_back(idx_test_f)

    n_used = n_samples_filtered
    print(f"\u1F4CA Data splits (ratios are % of TOTAL cells):")
    print(f"   Labeled train:   {len(idx_train_labeled)} ({len(idx_train_labeled)/n_used:.1%})")
    print(f"   Unlabeled train: {len(idx_train_unlabeled)} ({len(idx_train_unlabeled)/n_used:.1%})")
    print(f"   Validation:      {len(idx_val)} ({len(idx_val)/n_used:.1%})")
    print(f"   Test:            {len(idx_test)} ({len(idx_test)/n_used:.1%})")
    if len(rare_classes) > 0:
        print(f"   Removed {len(rare_classes)} rare class(es) with < {min_samples_per_class} samples")

    return idx_train_labeled, idx_train_unlabeled, idx_val, idx_test


def create_partial_labels(y_full, idx_labeled, idx_unlabeled, unknown_label=-1):
    y_partial = np.full(len(y_full), unknown_label, dtype=int)
    y_partial[idx_labeled] = y_full[idx_labeled]
    return y_partial


def verify_no_leakage(idx_train_labeled, idx_train_unlabeled, idx_val, idx_test) -> bool:
    all_indices = np.concatenate([idx_train_labeled, idx_train_unlabeled, idx_val, idx_test])
    unique_indices = np.unique(all_indices)
    if len(all_indices) != len(unique_indices):
        raise ValueError("Data leakage detected: overlapping indices in splits!")
    for name, idx in [("labeled", idx_train_labeled), ("unlabeled", idx_train_unlabeled),
                      ("val", idx_val), ("test", idx_test)]:
        if len(idx) == 0:
            raise ValueError(f"Empty split: {name}")
    print("No data leakage detected between splits")
    return True