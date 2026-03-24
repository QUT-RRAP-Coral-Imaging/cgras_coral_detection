#! /usr/bin/env python3

# script to compare output of manual counts to CCVS counts (from export .xlsx file)
# establish file paths and settings
# read manual counts into a matrix
# read CCVS counts into matrix
# find corresponding elements in both matrices
# compare the tab values
# average the results for 



# first, do for a single file
# then figure out how to do for all files in a folder (manual counts), and all sheets in a .xlsx file

###################################


# single file comparison:

import pandas as pd
import numpy as np  
import os
import matplotlib.pyplot as plt
import seaborn as sns

# file with manual counts from Mikaela
manual_counts_whole_tiles_file = '202505_CGRAS_ManualValidationCounts_tile-layout-data_wholetiles.xlsx'
# manual_counts_sheet_name = 'T05_CG1-202411122300'
manual_counts_sheet_name = 'T05_CG1-202411262300'
# manual_counts_sheet_name = 'T05_CG1-202412112300'

# file with CCVS counts from export (corresponding tile number)
ccvs_counts_file = 'T05_data_v2.xlsx'
# ccvs_sheet_name = 'CM-POLYP_MULTI-2024-11-12'
ccvs_sheet_name = 'CM-POLYP_MULTI-2024-11-26'
# ccvs_sheet_name = 'CM-POLYP_MULTI-2024-12-11'

def compare_counts(manual_file, manual_sheet, ccvs_file, ccvs_sheet):
    """
    Compare 20x20 matrices from manual and CCVS count Excel files
    
    Args:
        manual_file: Path to manual count Excel file
        manual_sheet: Sheet name for manual counts
        ccvs_file: Path to CCVS count Excel file
        ccvs_sheet: Sheet name for CCVS counts
        
    Returns:
        Dictionary with comparison results
    """
    print(f"Comparing manual counts from {manual_file}:{manual_sheet}")
    print(f"with CCVS counts from {ccvs_file}:{ccvs_sheet}")
    
    # Read Excel files
    manual_df = pd.read_excel(manual_file, sheet_name=manual_sheet)
    ccvs_df = pd.read_excel(ccvs_file, sheet_name=ccvs_sheet)
    
    # Extract 20x20 matrices (rows 2-21, columns A-T)
    # In pandas, indices are 0-based, so rows 2-21 are indices 1-20
    # Columns A-T are the first 20 columns (0-19)
    manual_matrix = manual_df.iloc[0:20, 1:21].values
    ccvs_matrix = ccvs_df.iloc[0:20, 0:20].values
    
    # convert objects to arrays
    manual_matrix = np.array(manual_matrix, dtype=int)
    ccvs_matrix = np.array(ccvs_matrix, dtype=int)
    
    
    # Compare differences
    difference_matrix = manual_matrix - ccvs_matrix
    absolute_difference = np.abs(difference_matrix)
    
    # import code
    # code.interact(local=dict(globals(), **locals()))  # Debugging line to inspect matrices
    
    # Calculate statistics
    mean_difference = np.mean(difference_matrix)
    mean_absolute_difference = np.mean(absolute_difference)
    max_difference = np.max(absolute_difference)
    total_manual_count = np.sum(manual_matrix)
    total_ccvs_count = np.sum(ccvs_matrix)
    percentage_difference = 100 * (total_manual_count - total_ccvs_count) / total_manual_count
    
    # Create visualization of the difference
    plt.figure(figsize=(12, 10))
    
    # Create a mask for zero values and custom annotations
    # Replace zeros with empty strings in annotations
    annot_matrix = difference_matrix.copy().astype(str)
    annot_matrix[difference_matrix == 0] = ''
    
    # Create a heatmap of the difference matrix
    sns.heatmap(difference_matrix, cmap='coolwarm', center=0, annot=annot_matrix, fmt="",
                cbar_kws={'label': 'Difference (Manual - CCVS)'})
    plt.title(f'Difference Matrix: {os.path.basename(ccvs_file)}, sheet: {ccvs_sheet_name} vs \n {os.path.basename(manual_file)}, sheet: {manual_counts_sheet_name}')
    plt.tight_layout()
    
    # Save heatmap figure
    output_file = f"comparison_{os.path.splitext(os.path.basename(ccvs_file))[0]}_{manual_counts_sheet_name}.png"
    plt.savefig(output_file)
    
    # Create histogram of differences
    plt.figure(figsize=(10, 6))
    
    # Flatten the difference matrix to get all difference values
    differences_flat = difference_matrix.flatten()
    
    # Create bins based on the actual range of differences with bin size of 1
    min_diff = int(np.min(differences_flat))
    max_diff = int(np.max(differences_flat))
    
    # Ensure we have at least 10 bins, but use actual data range
    bin_range = max(10, max_diff - min_diff + 1)
    bins = np.arange(min_diff, max_diff + 2, 1)  # +2 to include the max value
    
    # Create histogram
    plt.hist(differences_flat, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.title(f'Histogram of Differences (Manual - CCVS)\n{os.path.basename(manual_file)}: {manual_sheet}')
    plt.xlabel('Difference (Manual - CCVS)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show all integer values in the range
    plt.xticks(range(min_diff, max_diff + 1))
    
    # Add statistics text to the plot
    stats_text = f'Mean: {mean_difference:.2f}\nStd: {np.std(differences_flat):.2f}\nTotal pairs: {len(differences_flat)}\nRange: [{min_diff}, {max_diff}]'
    plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save histogram figure
    histogram_output_file = f"histogram_{os.path.splitext(os.path.basename(ccvs_file))[0]}_{manual_counts_sheet_name}.png"
    plt.savefig(histogram_output_file)
    
    print(f"Heatmap saved to: {output_file}")
    print(f"Histogram saved to: {histogram_output_file}")
    
    results = {
        "manual_counts": manual_matrix,
        "ccvs_counts": ccvs_matrix,
        "difference_matrix": difference_matrix,
        "mean_difference": mean_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "max_difference": max_difference,
        "total_manual_count": total_manual_count,
        "total_ccvs_count": total_ccvs_count,
        "percentage_difference": percentage_difference
    }
    
    # Print summary statistics
    print(f"Total manual count: {total_manual_count}")
    print(f"Total CCVS count: {total_ccvs_count}")
    print(f"Mean difference: {mean_difference:.2f}")
    print(f"Mean absolute difference: {mean_absolute_difference:.2f}")
    print(f"Maximum absolute difference: {max_difference}")
    print(f"Percentage difference: {percentage_difference:.2f}%")
    
    return results

def main():
    # Compare single file
    results = compare_counts(
        manual_counts_whole_tiles_file,
        manual_counts_sheet_name,
        ccvs_counts_file,
        ccvs_sheet_name
    )
    
    plt.show()  # Display the visualizations

if __name__ == "__main__":
    main()