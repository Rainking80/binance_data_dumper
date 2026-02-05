
import os
import pandas as pd
import glob

def print_beautiful_df(df, title):
    print("=" * 100)
    print(f"FILE: {title}")
    print("-" * 100)
    
    # Configure pandas to display all columns and ensure width fits
    with pd.option_context('display.max_columns', None, 
                           'display.width', 1000, 
                           'display.max_colwidth', 50):
        
        print("\nTOP 5 ROWS:")
        print(df.head(5))
        
        print("\n" + "." * 100 + "\n")
        
        print("BOTTOM 5 ROWS:")
        print(df.tail(5))
        
    print("=" * 100)
    print("\n")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'condensed_data')
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    feather_files = glob.glob(os.path.join(data_dir, "*.feather"))
    
    if not feather_files:
        print(f"No feather files found in '{data_dir}'")
        return

    print(f"Found {len(feather_files)} feather files.\n")
    
    for file_path in feather_files:
        try:
            df = pd.read_feather(file_path)
            file_name = os.path.basename(file_path)
            
            target_columns = ['oi_zscore', 'volume_zscore', 'funding_zscore', 'date', 'datetime', 'open_time']
            # Only keep columns that actually exist in the dataframe
            existing_columns = [col for col in target_columns if col in df.columns]
            
            if existing_columns:
                print_beautiful_df(df[existing_columns], file_name)
                for col in ['oi_zscore', 'funding_zscore']:
                    if col in df.columns:
                        non_zero = (df[col] != 0).sum()
                        print(f"{col} non-zero count: {non_zero}/{len(df)}")
            else:
                print(f"FILE: {file_name} - None of the requested columns found.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    main()
