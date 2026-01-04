import pandas as pd

def create_zwicky_names_with_decimals(input_file, output_file):
    # 1. Read the CSV
    df = pd.read_csv(input_file)
    
    def format_name(row):
        # We need both RA and Dec strings to create the name
        ra_str = str(row['RA1950']).strip()
        dec_str = str(row['DE1950']).strip()
        
        # Clean RA: "00 00.1" -> "0000.1" 
        # Just removes the space, keeps the decimal
        ra_clean = ra_str.replace(" ", "")
        
        # Clean Dec: "+08 06" -> "+0806"
        # Just removes the space
        dec_clean = dec_str.replace(" ", "")
        
        return f"ZwCl {ra_clean}{dec_clean}"

    # 2. Create the new column
    print("Generating zwicky_name column (keeping decimals)...")
    df['zwicky_name'] = df.apply(format_name, axis=1)
    
    # 3. Save the result
    df.to_csv(output_file, index=False)
    print(f"Done! Saved to {output_file}")

# Usage
create_zwicky_names_with_decimals('zwicky.csv', 'zwicky_with_names.csv')