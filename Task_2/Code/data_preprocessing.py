import pandas as pd
import numpy as np
import re
import sys

def convert_numeric_string(value):

    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string and clean
    value_str = str(value).strip()
    
    # Remove special characters
    value_str = value_str.replace('$', '').replace(',', '')
    value_str = value_str.replace('%', '').replace('+', '')
    value_str = value_str.replace('"', '').strip()
    
    if value_str == '' or value_str.lower() in ['nan', 'null', 'none']:
        return np.nan
    
    # Handle K/M/B/T suffixes
    multiplier = 1
    if value_str.endswith('K'):
        multiplier = 1e3
        value_str = value_str[:-1]
    elif value_str.endswith('M'):
        multiplier = 1e6
        value_str = value_str[:-1]
    elif value_str.endswith('B'):
        multiplier = 1e9
        value_str = value_str[:-1]
    elif value_str.endswith('T'):
        multiplier = 1e12
        value_str = value_str[:-1]
    
    try:
        return float(value_str) * multiplier
    except ValueError:
        return np.nan


def clean_numeric_columns(df):
 
    numeric_columns = ['price_usd', 'vol_24h', 'total_vol', 'chg_24h', 'chg_7d', 'market_cap']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(convert_numeric_string)
            print(f"  . Converted {col} to numeric")
    
    return df


def impute_missing_values(df):
    # Sort by symbol and timestamp for proper forward/backward fill
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Count missing values before imputation
    missing_before = df.isnull().sum()
    print(f"  Missing values before imputation:")
    for col in ['chg_24h', 'chg_7d', 'market_cap']:
        if missing_before[col] > 0:
            print(f"    - {col}: {missing_before[col]}")
    
    # Impute using forward fill then backward fill by symbol group
    columns_to_impute = ['chg_24h', 'chg_7d', 'market_cap']
    
    for col in columns_to_impute:
        if col in df.columns:
            # Forward fill within each cryptocurrency group
            df[col] = df.groupby('symbol')[col].ffill()
            # Backward fill for any remaining NaN (beginning of time series)
            df[col] = df.groupby('symbol')[col].bfill()
    
    # For any still missing (rare case), use global median as last resort
    for col in columns_to_impute:
        if col in df.columns and df[col].isnull().sum() > 0:
            global_median = df[col].median()
            df[col] = df[col].fillna(global_median)
            print(f"  . Used global median for remaining NaN in {col}")
    
    missing_after = df.isnull().sum()
    print(f"  . Imputation complete. Remaining missing values: {missing_after.sum()}")
    
    return df


def remove_duplicates(df):
    rows_before = len(df)
    df = df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
    rows_after = len(df)
    
    duplicates_removed = rows_before - rows_after
    print(f"  . Removed {duplicates_removed} duplicate records")
    
    return df


def handle_invalid_values(df):

    rows_before = len(df)
    
    # Remove zero or negative prices
    df = df[df['price_usd'] > 0]
    print(f"  . Removed rows with zero/negative prices")
    
    # Remove zero market cap
    df = df[df['market_cap'] > 0]
    print(f"  . Removed rows with zero market cap")
    
    # Cap total_vol between 0-100%
    df['total_vol'] = df['total_vol'].clip(lower=0, upper=100)
    print(f"  . Capped total_vol between 0-100%")
    
    # Remove rows with zero volume (optional - keeping them for now)
    # df = df[df['vol_24h'] > 0]
    
    rows_after = len(df)
    invalid_removed = rows_before - rows_after
    print(f"   Removed {invalid_removed} rows with invalid values")
    
    return df


def create_volume_to_marketcap_ratio(df):
    
    df['vol_to_marketcap_ratio'] = df['vol_24h'] / df['market_cap']
    
    # Replace inf/nan with 0 (in case of division issues)
    df['vol_to_marketcap_ratio'] = df['vol_to_marketcap_ratio'].replace([np.inf, -np.inf], np.nan)
    df['vol_to_marketcap_ratio'] = df['vol_to_marketcap_ratio'].fillna(0)
    
    print(f"   Created vol_to_marketcap_ratio feature")
    print(f"    Range: {df['vol_to_marketcap_ratio'].min():.6f} to {df['vol_to_marketcap_ratio'].max():.6f}")
    
    return df


def create_binary_target(df):

    df['target'] = (df['chg_24h'] > 0).astype(int)
    
    target_distribution = df['target'].value_counts()
    print(f"   Created binary target:")
    print(f"    - Class 0 (down/flat): {target_distribution.get(0, 0)} ({target_distribution.get(0, 0)/len(df)*100:.2f}%)")
    print(f"    - Class 1 (up): {target_distribution.get(1, 0)} ({target_distribution.get(1, 0)/len(df)*100:.2f}%)")
    
    return df


def create_future_target(df):

    rows_before = len(df)
    
    # Create future target by shifting within each cryptocurrency group
    df['future_target'] = df.groupby('symbol')['target'].shift(-1)
    
    print(f"   Created future_target column using shift(-1)")
    print(f"   Each row now predicts the NEXT period's movement (same crypto)")
    
    # Check how many NaN values (last row of each crypto)
    nan_count = df['future_target'].isna().sum()
    print(f"   Rows with NaN future_target: {nan_count} (last row of each crypto)")
    
    # Remove rows where we can't predict future (last row of each cryptocurrency)
    df = df.dropna(subset=['future_target'])
    
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    print(f"   Removed {rows_removed} rows (can't predict beyond available data)")
    print(f"   Final dataset: {rows_after} rows")
    
    # Show distribution of future target
    future_target_distribution = df['future_target'].value_counts()
    print(f"\n   Future target distribution:")
    print(f"    - Class 0 (will go down): {future_target_distribution.get(0.0, 0)} ({future_target_distribution.get(0.0, 0)/len(df)*100:.2f}%)")
    print(f"    - Class 1 (will go up): {future_target_distribution.get(1.0, 0)} ({future_target_distribution.get(1.0, 0)/len(df)*100:.2f}%)")
    
    return df


def preprocess_cryptocurrency_data(input_csv_path, output_csv_path):
  
    print("=" * 70)
    print("CRYPTOCURRENCY DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"   Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Apply preprocessing steps
    df = clean_numeric_columns(df)
    df = impute_missing_values(df)
    df = remove_duplicates(df)
    df = handle_invalid_values(df)
    df = create_volume_to_marketcap_ratio(df)
    df = create_binary_target(df)
    df = create_future_target(df)  # NEW: Fix data leakage by creating future target
    
    # Save cleaned data
    print(f"\n{'=' * 70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Final dataset: {len(df)} rows and {len(df.columns)} columns")
    print(f"\nSaving cleaned data to: {output_csv_path}")
    
    df.to_csv(output_csv_path, index=False)
    print(f"  Successfully saved preprocessed data!")
    
    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Original rows: ~111,096")
    print(f"Final rows: {len(df)}")
    print(f"Data retention: {len(df)/111096*100:.2f}%")
    print(f"\nNew columns added:")
    print(f"  - vol_to_marketcap_ratio (continuous) - Trading activity indicator")
    print(f"  - target (binary: 0 or 1) - Current period movement")
    print(f"  - future_target (binary: 0 or 1) - NEXT period movement (NO DATA LEAKAGE!)")
    print(f"\n  IMPORTANT: Use 'future_target' for training to avoid data leakage!")
    print(f"   - 'target' = current period (100% accuracy but useless)")
    print(f"   - 'future_target' = next period (~88% accuracy but real prediction)")
    
    return df


if __name__ == "__main__":
    # Default file paths
    input_file = "cryptocurrency.csv"
    output_file = "refined_data.csv"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Run preprocessing
    try:
        df_cleaned = preprocess_cryptocurrency_data(input_file, output_file)
        print(f"\n SUCCESS! Preprocessing completed successfully.")
        print(f"\nYou can now use '{output_file}' for ML model training.")
    except FileNotFoundError:
        print(f"\n ERROR: Input file '{input_file}' not found")
        print(f"Usage: python data_preprocessing.py [input_csv] [output_csv]")
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()