#!/usr/bin/env python3

import os
import json
import pandas as pd
import glob
from typing import Dict, List

def extract_info_from_filename(filename: str) -> Dict:
    """Extract relevant information from the filename."""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    
    try:
        # Extract model name (gemma_2_9b_it)
        model_idx = parts.index('gemma')
        model_name = '_'.join(parts[model_idx:model_idx+4])
        
        # Extract dimension (care, liberty, etc.)
        dimension = parts[model_idx+4]
        
        # Extract cluster (cl1, cl2, etc.)
        cluster = next(part for part in parts if part.startswith('cl'))
        
        # Extract comparison type (moral_vs_immoral or moral_vs_neutral)
        comp_start = parts.index('moral')
        comparison = '_'.join(parts[comp_start:comp_start+3])
        
        # Extract ablation value
        value_idx = parts.index('value') + 1
        ablation_value = float(parts[value_idx])
        
        return {
            'model': model_name,
            'dimension': dimension,
            'cluster': cluster,
            'comparison': comparison,
            'ablation_value': round(ablation_value, 3)
        }
    except Exception as e:
        print(f"Error parsing filename {basename}: {str(e)}")
        print(f"Parts: {parts}")
        raise

def process_results_file(filename: str) -> Dict:
    """Process a single results file and extract relevant information."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Extract metadata from filename
        metadata = extract_info_from_filename(filename)
        
        # Extract summary statistics
        summary = data['probe_analysis']['summary']
        metadata.update({
            'avg_moral_pred_change': round(summary['avg_moral_pred_change'], 3),
            'std_moral_pred_change': round(summary['std_moral_pred_change'], 3),
            'avg_immoral_pred_change': round(summary['avg_immoral_pred_change'], 3),
            'std_immoral_pred_change': round(summary['std_immoral_pred_change'], 3),
            'moral_effect_size': round(summary['moral_effect_size'], 3),
            'immoral_effect_size': round(summary['immoral_effect_size'], 3)
        })
        
        return metadata
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        raise

def main():
    # Define the base directory for ablation results
    base_dir = "./results/ablation/gemma_2_9b_it"
    
    # Find all results.json files
    pattern = os.path.join(base_dir, "**/*results.json")
    result_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(result_files)} result files")
    
    # Process all files
    results = []
    for file in result_files:
        try:
            print(f"Processing {file}")
            result = process_results_file(file)
            results.append(result)
        except Exception as e:
            print(f"Skipping {file} due to error: {str(e)}")
            continue
    
    if not results:
        print("No results were processed successfully!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Columns:")
    print(df.columns.tolist())
    
    # Reorder columns for better readability
    columns = [
        'model', 'dimension', 'cluster', 'comparison', 'ablation_value',
        'avg_moral_pred_change', 'std_moral_pred_change',
        'avg_immoral_pred_change', 'std_immoral_pred_change',
        'moral_effect_size', 'immoral_effect_size'
    ]
    
    # Check if all required columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: Missing columns: {missing_cols}")
        # Only use columns that exist
        columns = [col for col in columns if col in df.columns]
    
    df = df[columns]
    
    # Round all float columns to 3 decimal places
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].round(3)
    
    # Sort the DataFrame
    df = df.sort_values(by=['dimension', 'cluster', 'ablation_value', 'comparison'])
    
    # Save to CSV
    output_file = "./results/ablation/ablation_summary.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"\nSummary table saved to: {output_file}")
    print(f"Number of rows in summary: {len(df)}")

if __name__ == "__main__":
    main() 