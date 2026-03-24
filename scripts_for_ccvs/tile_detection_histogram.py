# python code
# read in the target spreadsheet file
# read in the specified sheet that matches the particular string pattern "Detect-YYYY-MM-DD", there may be multiple sheets with different dates
# import the data from these sheets, the rows of the data represent individual detections
# plot a histogram of the frequency of detections versus the size_x and the size_y
# we want to calculate the area of each detection, area = size_x * size_y
# we want to plot the histogram of area for detection counts

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os

# Specify the folder containing Excel files
excel_folder = '/home/dtsai/Data/cgras_datasets/cgras_amag_2024_highdensityexperiment/ccvs_output_20250723_140amagmodel'
excel_files = glob.glob(os.path.join(excel_folder, '*.xlsx'))

# Collect all detections from all Excel files
all_detections = []
for excel_path in excel_files:
    try:
        # Read the Excel file and get all sheet names
        xls = pd.ExcelFile(excel_path)
        sheet_pattern = re.compile(r'Detect-\d{4}-\d{2}-\d{2}')
        sheet_names = [name for name in xls.sheet_names if sheet_pattern.match(name)]
        
        # Process each matching sheet
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            # Filter for only "mask_live" yolo_class detections
            if 'yolo_class' in df.columns:
                df = df[df['yolo_class'] == 'alive_coral']
            # Expect columns: size_x, size_y
            if 'size_x' in df.columns and 'size_y' in df.columns and not df.empty:
                all_detections.append(df[['size_x', 'size_y']])
        
        print(f"Processed: {os.path.basename(excel_path)}")
    except Exception as e:
        print(f"Error processing {excel_path}: {e}")

print(f"Total Excel files processed: {len(excel_files)}")
print(f"Total detection datasets found: {len(all_detections)}")

detections_df = pd.concat(all_detections, ignore_index=True)

# Calculate area and aspect ratio
if not detections_df.empty:
    detections_df['area'] = detections_df['size_x'] * detections_df['size_y'] * 1e6
    detections_df['aspect_ratio'] = detections_df[['size_x', 'size_y']].max(axis=1) / detections_df[['size_x', 'size_y']].min(axis=1)
    
    # Define aspect ratio threshold
    aspect_ratio_threshold = 3.5
    num_bins = 40
    
    # Split data based on aspect ratio threshold
    normal_detections = detections_df[detections_df['aspect_ratio'] <= aspect_ratio_threshold]
    high_aspect_detections = detections_df[detections_df['aspect_ratio'] > aspect_ratio_threshold]

    # Plot histograms
    plt.figure(figsize=(15, 10))
    
    # Calculate common bin edges for consistent binning
    size_x_bins = np.linspace(detections_df['size_x'].min(), detections_df['size_x'].max(), 51)
    size_y_bins = np.linspace(detections_df['size_y'].min(), detections_df['size_y'].max(), 51)
    area_bins = np.linspace(detections_df['area'].min(), detections_df['area'].max(), 51)
    aspect_ratio_bins = np.linspace(detections_df['aspect_ratio'].min(), detections_df['aspect_ratio'].max(), 51)
    
    plt.subplot(2, 2, 1)
    plt.hist(normal_detections['size_x'], bins=size_x_bins, color='skyblue', edgecolor='black', alpha=0.7, label=f'Aspect ratio ≤ {aspect_ratio_threshold}')
    if not high_aspect_detections.empty:
        plt.hist(high_aspect_detections['size_x'], bins=size_x_bins, color='red', edgecolor='black', alpha=0.7, label=f'Aspect ratio > {aspect_ratio_threshold}')
    plt.title('Frequency vs. size_x')
    plt.xlabel('size_x')
    plt.ylabel('Frequency')
    plt.locator_params(axis='x', nbins=num_bins)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist(normal_detections['size_y'], bins=size_y_bins, color='salmon', edgecolor='black', alpha=0.7, label=f'Aspect ratio ≤ {aspect_ratio_threshold}')
    if not high_aspect_detections.empty:
        plt.hist(high_aspect_detections['size_y'], bins=size_y_bins, color='red', edgecolor='black', alpha=0.7, label=f'Aspect ratio > {aspect_ratio_threshold}')
    plt.title('Frequency vs. size_y')
    plt.xlabel('size_y')
    plt.ylabel('Frequency')
    plt.locator_params(axis='x', nbins=num_bins)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(normal_detections['area'], bins=area_bins, color='lightgreen', edgecolor='black', alpha=0.7, label=f'Aspect ratio ≤ {aspect_ratio_threshold}')
    if not high_aspect_detections.empty:
        plt.hist(high_aspect_detections['area'], bins=area_bins, color='red', edgecolor='black', alpha=0.7, label=f'Aspect ratio > {aspect_ratio_threshold}')
    plt.title('Frequency vs. Area')
    plt.xlabel('Area (size_x * size_y) * 10^6')
    plt.ylabel('Frequency')
    plt.locator_params(axis='x', nbins=num_bins)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.hist(normal_detections['aspect_ratio'], bins=aspect_ratio_bins, color='orange', edgecolor='black', alpha=0.7, label=f'Aspect ratio ≤ {aspect_ratio_threshold}')
    if not high_aspect_detections.empty:
        plt.hist(high_aspect_detections['aspect_ratio'], bins=aspect_ratio_bins, color='red', edgecolor='black', alpha=0.7, label=f'Aspect ratio > {aspect_ratio_threshold}')
    plt.title('Frequency vs. Aspect Ratio')
    plt.xlabel('Aspect Ratio (max/min)')    
    plt.ylabel('Frequency')
    # plt.ylim(0, 40000)  # Set maximum y-value to 1000 (adjust as needed)
    plt.locator_params(axis='x', nbins=num_bins)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.legend()

    plt.tight_layout()
    
    # Save as high-resolution JPEG
    
    plt.savefig('ccvs_histogram_analysis.jpg', dpi=300, bbox_inches='tight', format='jpg')
    
    plt.show()
else:
    print('No detections found with size_x and size_y columns.')