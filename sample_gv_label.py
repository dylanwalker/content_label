import pandas as pd
import numpy as np


df = pd.read_feather(r'C:\Users\dwalker\Downloads\gun_tv_labeled_localnews_all.feather')

# Define the columns to stratify on
columns_to_stratify = ["mentions_guns_firearms", "gun_violence_incident", "gun_policy_discussion"]

# Target sample size
total_samples = 500
target_true_ratio = 2/3  # 2/3 True
target_false_ratio = 1/3  # 1/3 False

# Calculate target counts for each combination
target_true_count = int(total_samples * target_true_ratio)
target_false_count = int(total_samples * target_false_ratio)

def sample_with_minimums(df, columns, min_true_counts, total_samples):
    """
    Simple sampling that guarantees minimum True cases for each column in order
    
    Args:
        df: DataFrame to sample from
        columns: List of column names in priority order
        min_true_counts: List of minimum True counts for each column (same order as columns)
        total_samples: Total number of samples to return
    """
    # Clean data - remove rows with missing values in target columns
    clean_df = df.copy()
    for col in columns:
        clean_df = clean_df[clean_df[col].notna()]
    
    # Reset index to avoid duplicate index issues
    clean_df = clean_df.reset_index(drop=True)
    
    print(f"Clean data shape: {clean_df.shape}")
    print(f"Target total samples: {total_samples}")
    
    # Validate inputs
    if len(columns) != len(min_true_counts):
        raise ValueError("columns and min_true_counts must have same length")
    
    # Check data availability
    print("\nChecking data availability:")
    for col, min_count in zip(columns, min_true_counts):
        true_available = (clean_df[col] == True).sum()
        print(f"{col}: {true_available} True available (need minimum: {min_count})")
        if true_available < min_count:
            print(f"  WARNING: Not enough True cases available!")
    
    # Start sampling
    selected_indices = set()
    
    print("\nSampling in order:")
    
    # Process each column in order
    for col, min_count in zip(columns, min_true_counts):
        print(f"\nProcessing {col} (need minimum {min_count} True cases):")
        
        # Check how many True cases we already have for this column
        if selected_indices:
            current_sample = clean_df.loc[list(selected_indices)]
            current_true_count = (current_sample[col] == True).sum()
        else:
            current_true_count = 0
        
        print(f"  Already have {current_true_count} True cases for {col}")
        
        # Calculate how many more True cases we need
        need_more = max(0, min_count - current_true_count)
        
        if need_more > 0:
            print(f"  Need {need_more} more True cases for {col}")
            
            # Get available True cases for this column (not already selected)
            available_true = clean_df[(clean_df[col] == True) & (~clean_df.index.isin(selected_indices))]
            
            if len(available_true) >= need_more:
                # Sample the needed True cases
                new_true_sample = available_true.sample(n=need_more, replace=False, random_state=67)
                selected_indices.update(new_true_sample.index)
                print(f"  Added {len(new_true_sample)} True cases for {col}")
            else:
                # Take all available True cases
                selected_indices.update(available_true.index)
                print(f"  Added {len(available_true)} True cases for {col} (all available)")
        else:
            print(f"  Already have enough True cases for {col}")
    
    print(f"\nAfter processing all columns: {len(selected_indices)} samples selected")
    
    # Fill remaining slots with random samples
    remaining_needed = total_samples - len(selected_indices)
    if remaining_needed > 0:
        print(f"Filling remaining {remaining_needed} slots randomly...")
        
        available_indices = clean_df.index[~clean_df.index.isin(selected_indices)]
        
        if len(available_indices) >= remaining_needed:
            random_sample = clean_df.loc[available_indices].sample(n=remaining_needed, replace=False, random_state=67)
            selected_indices.update(random_sample.index)
        else:
            # Take all remaining available (but this might be less than needed)
            selected_indices.update(available_indices)
            print(f"  Warning: Only added {len(available_indices)} samples, still short of target")
    
    # Create final sample (limit to total_samples if we have too many)
    print(f"Selected indices count: {len(selected_indices)}")
    final_indices = list(selected_indices)[:total_samples]
    print(f"Final indices count after slicing: {len(final_indices)}")
    
    # Check for and remove duplicates
    unique_final_indices = list(set(final_indices))
    print(f"Unique final indices count: {len(unique_final_indices)}")
    
    # If we lost indices due to duplicates, we need to handle this
    if len(unique_final_indices) < total_samples:
        print(f"Warning: Lost {total_samples - len(unique_final_indices)} indices due to duplicates")
    
    final_sample = clean_df.loc[unique_final_indices].copy()
    
    print(f"Final sample size: {len(final_sample)}")
    
    return final_sample

# Set minimum True counts for each column (approximately 500/3 = 167 per column)
min_true_counts = [167, 167, 167]  # Minimum True cases for each of the 3 columns

# Create the sample with guaranteed minimums
sampled_df = sample_with_minimums(df, columns_to_stratify, min_true_counts, total_samples)

# Verify the distribution
print(f"Total samples: {len(sampled_df)}")
print("\nDistribution for each column:")
for col in columns_to_stratify:
    true_count = sampled_df[col].sum()
    false_count = len(sampled_df) - true_count
    true_pct = (true_count / len(sampled_df)) * 100
    false_pct = (false_count / len(sampled_df)) * 100
    print(f"{col}:")
    print(f"  True: {true_count} ({true_pct:.1f}%)")
    print(f"  False: {false_count} ({false_pct:.1f}%)")

# Save the sampled dataframe
sampled_df.to_feather(r'data.feather')
print(f"\nSampled dataframe saved to: data.feather")