#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from astroquery.ned import Ned
from astropy.coordinates import SkyCoord
import astropy.units as u
import time
import argparse
import logging

# Set up logging to a file
logging.basicConfig(
    filename='ned_search.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("master_clusters.parquet"))
    p.add_argument("--output", type=Path, default=Path("master_clusters_ned.parquet"))
    p.add_argument("--checkpoint", type=Path, default=Path("ned_checkpoint.csv"))
    p.add_argument("--radius-arcmin", type=float, default=0.5)
    p.add_argument("--delay", type=float, default=0.2)
    return p.parse_args()

def query_ned_for_cluster(cluster_id, ra, dec, radius):
    """Queries NED and logs the specific findings for each cluster."""
    try:
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        results = Ned.query_region(coord, radius=radius*u.arcmin)
        
        if results is None or len(results) == 0:
            msg = f"ID {cluster_id}: No NED objects found within {radius}'"
            print(f"  [-] {msg}")
            logging.info(msg)
            return None

        results.sort('Separation')
        best = results[0]
        obj_name = str(best['Object Name'])
        obj_type = str(best['Type'])
        # Separation is usually in arcmin in query_region results
        sep_arcsec = float(best['Separation']) * 60.0 
        z = best['Redshift'] if 'Redshift' in best.colnames else np.nan

        # Detailed summary string
        z_str = f"{z:.4f}" if pd.notna(z) else "N/A"
        summary = f"Found {obj_name} ({obj_type}) | z={z_str} | Sep={sep_arcsec:.2f}\""
        
        print(f"  [+] ID {cluster_id}: {summary}")
        logging.info(f"ID {cluster_id}: {summary}")

        # Get Cross-IDs
        alt_names = obj_name
        try:
            ids = Ned.list_identifications(obj_name)
            if ids is not None:
                alt_names = " | ".join(ids['Object Name'].astype(str))
        except:
            pass

        return {
            "master_id": cluster_id,
            "ned_name": obj_name,
            "ned_z": z,
            "ned_alt_names": alt_names,
            "ned_sep_arcsec": sep_arcsec,
            "ned_type": obj_type
        }
    except Exception as e:
        err_msg = f"ID {cluster_id}: Error querying NED - {str(e)}"
        print(f"  [!] {err_msg}")
        logging.error(err_msg)
        return None

def main():
    args = parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        return

    df = pd.read_parquet(args.input)
    
    # Checkpoint logic
    if args.checkpoint.exists():
        processed_df = pd.read_csv(args.checkpoint)
        processed_ids = set(processed_df['master_id'])
        results = processed_df.to_dict('records')
        print(f"Resuming: {len(processed_ids)} already processed.")
    else:
        results = []
        processed_ids = set()

    to_process = df[~df['master_id'].isin(processed_ids)]
    print(f"Querying {len(to_process):,} clusters via NED...")

    try:
        for i, (idx, row) in enumerate(to_process.iterrows()):
            res = query_ned_for_cluster(row['master_id'], row['ra_deg'], row['dec_deg'], args.radius_arcmin)
            
            if res:
                results.append(res)
            
            # Checkpoint every 50
            if (i + 1) % 50 == 0:
                pd.DataFrame(results).to_csv(args.checkpoint, index=False)
                print(f"--- Checkpoint saved at {i+1} rows ---")
            
            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
        pd.DataFrame(results).to_csv(args.checkpoint, index=False)
        return

    # Merge and Save
    print("\nFinalizing catalog...")
    ned_df = pd.DataFrame(results)
    final_df = df.merge(ned_df, on="master_id", how="left")

    # Update z_best if missing
    z_mask = final_df['z_best'].isna() & final_df['ned_z'].notna()
    final_df.loc[z_mask, 'z_best'] = final_df.loc[z_mask, 'ned_z']

    final_df.to_parquet(args.output, index=False)
    if args.checkpoint.exists(): args.checkpoint.unlink()
    
    print(f"Done! Enriched catalog saved to {args.output}")

if __name__ == "__main__":
    main()