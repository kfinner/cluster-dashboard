#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

# -----------------------
# 1. Column Mapping Dictionaries
# -----------------------
act_hilton_dict = {"name": "ACT-CL", "mass": "M500cUPP", "z": "z", "SNR": "SNR"}
act_dr5_dict = {"name": "Name", "mass": "M500-1CCAL", "z": "zsp1", "SNR": "SNR"}
spt_dict = {"name": "SPT-CL", "mass": "M500c", "z": "z", "SNR": "xi"}
spt_ecs_dict = {"name": "SPT-CL", "mass": "M500c", "z": "z", "SNR": "Xi"}
psz2_dict = {"name": "Name", "mass": "MSZ", "z": "z", "SNR": "SNR"}
abell_dict = {"name": "ACO", "rich": "Rich", "z": "z", "Dclass": "Dclass"}
camira_sdss_dict = {"name": "recno", "z": "z", "Nmem": "Ncor"}
HSC_camira_dict = {"name": "Name", "zph": "zph", "zsp": "zsp", "Nmem": "Nmem"}
madcows1_dict = {"name": "Name", "zph": "Photz", "rich": "Rich"}
madcows2_dict = {"name": "MOO2", "zph": "zphot", "zsp": "zspec", "SNR": "S_N"}
mcxc_dict = {"name": "MCXC", "original_name": "OName", "mass": "M500", "z": "z"}
mcxc2_dict = {"name": "MCXC", "original_name": "OName", "mass": "M500", "z": "z"}
noras_dict = {"name": "RXC", "original_name": "Name", "z": "z"}
redmapper_dict = {"name": "Name", "Nmem": "Ng", "zsp": "zspec", "zph": "zlambda"}
reflex_dict = {"name": "RXC", "original_name": "Name", "z": "z", "Nmem": "Ngal"}
whl_dict = {"name": "WHL", "zph": "zph", "zsp": "zsp", "Nmem": "N200"}
xcs_dict = {"name": "Name", "z": "z"}
efeds_dict = {"name": "ID", "z": "z", "SNR": "SNRMax"}
erass_dict = {"name": "Name", "z": "zBest"}
macs_dist = {"name": "MACS", "z": "z"}
rcs2_dict = {"name": "RCS", "z": "z" }
xxl_dict = {"name": "XLSSC", "z": "z", "Nmem": "Ngal"}
zwicky_dict = {"name": "zwicky_name", "Nmem": "GalCnt"}
ciza1_dict = {"name": "CIZA", "z": "z", "original_name": "Name", }
ciza2_dict = {"name": "CIZA", "z": "z", "original_name": "Name", }

MASS_SCALE_TO_MSUN = {"psz2": 1e14, "mcxc": 1e14, "spt": 1e14, "spt_ecs": 1e14, "act_hilton": 1e14, "mcxc2": 1e14}
MASS_MIN_MSUN, MASS_MAX_MSUN = 1e12, 1e16

# -----------------------
# 2. Helpers
# -----------------------

def format_sdss_name(ra, dec):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    ra_hms = c.ra.hms
    dec_dms = c.dec.dms
    ra_str = f"{int(ra_hms.h):02d}{int(ra_hms.m):02d}{ra_hms.s:05.2f}"
    sign = "+" if dec_dms.d >= 0 else "-"
    dec_str = f"{sign}{int(abs(dec_dms.d)):02d}{int(abs(dec_dms.m)):02d}{abs(dec_dms.s):04.1f}"
    return f"SDSS J{ra_str}{dec_str}"

def apply_catalog_prefix(val, prefix_from_dict, ra=None, dec=None):
    if pd.isna(val): return val
    if prefix_from_dict == "recno" and ra is not None and dec is not None:
        return format_sdss_name(ra, dec)
    if isinstance(val, (float, np.float64, np.float32)):
        if np.isnan(val): return val
        s = str(int(val))
    else:
        s = str(val).strip()
    if prefix_from_dict == "RCS":
        if not s.upper().startswith("RCS"): return f"RCS {s}"
        return s
    if prefix_from_dict == "XLSSC":
        if not s.upper().startswith("XLSSC"): return f"XLSSC {s}"
        return s
    if prefix_from_dict == "ACO":
        if s.lower().startswith("abell"): return s
        return f"Abell {s}"
    if re.match(r'^J\d', s):
        if prefix_from_dict == "Name": return s
        return f"{prefix_from_dict} {s}"
    return s

def sanitize_redshift(z, zmax=20.0):
    if not isinstance(z, (pd.Series, np.ndarray)):
        val = pd.to_numeric(z, errors="coerce")
        return val if (pd.notna(val) and 0 <= val <= zmax) else np.nan
    z = pd.to_numeric(z, errors="coerce")
    z[(z < 0) | (z > zmax)] = np.nan
    return z

def sanitize_mass_for_cluster_use(m):
    if not isinstance(m, (pd.Series, np.ndarray)):
        val = pd.to_numeric(m, errors="coerce")
        if pd.isna(val) or val < MASS_MIN_MSUN or val > MASS_MAX_MSUN: return np.nan
        return val
    m = pd.to_numeric(m, errors="coerce")
    m[(m < MASS_MIN_MSUN) | (m > MASS_MAX_MSUN)] = np.nan
    return m

def find_radec_columns(df):
    for ra_c, dec_c in [("RA_deg", "Dec_deg"), ("_RA.icrs", "_DE.icrs"), ("RA", "Dec"), ("RAJ2000", "DEJ2000")]:
        if ra_c in df.columns and dec_c in df.columns: return ra_c, dec_c
    raise ValueError("RA/Dec columns not found.")

@dataclass
class CatalogSpec:
    name: str
    csv_file: str
    colmap: Dict[str, str]

CATALOGS = [
    CatalogSpec("abell", "abell_catalog.csv", abell_dict),
    CatalogSpec("psz2", "psz2_catalog.csv", psz2_dict),
    CatalogSpec("spt", "spt_catalog.csv", spt_dict),
    CatalogSpec("spt_ecs", "spt_ecs.csv", spt_ecs_dict),
    CatalogSpec("act_hilton", "act_catalog.csv", act_hilton_dict),
    CatalogSpec("act_dr5", "act_dr5_mcmf.csv", act_dr5_dict),
    CatalogSpec("mcxc", "mcxc.csv", mcxc_dict),
    CatalogSpec("noras", "noras.csv", noras_dict),
    CatalogSpec("reflex70", "reflex70.csv", reflex_dict),
    CatalogSpec("redmapper", "redmapper_dr8.csv", redmapper_dict),
    CatalogSpec("whl2012", "whl2012.csv", whl_dict),
    CatalogSpec("hsc_camira", "hsc_camira_s16a.csv", HSC_camira_dict),
    CatalogSpec("madcows1", "madcows.csv", madcows1_dict),
    CatalogSpec("madcows2", "madcows2.csv", madcows2_dict),
    CatalogSpec("xcs", "xcs_dr1.csv", xcs_dict),
    CatalogSpec("camira_sdss", "camira_sdss_dr8.csv", camira_sdss_dict),
    CatalogSpec("efeds", "efeds.csv", efeds_dict),
    CatalogSpec("erass", "erass.csv", erass_dict),
    CatalogSpec("macs", "macs.csv", macs_dist),
    CatalogSpec("rcs2", "rcs2.csv", rcs2_dict),
    CatalogSpec("xxl", "xxl.csv", xxl_dict),
    CatalogSpec("zwicky", "zwicky_with_names.csv", zwicky_dict),
    CatalogSpec("ciza1", "ciza.csv", ciza1_dict),
    CatalogSpec("ciza2", "ciza2.csv", ciza2_dict),
]

# -----------------------
# 3. Processing Logic
# -----------------------

def downselect_large_catalog(df, catalog, max_rows):
    if len(df) <= max_rows: return df
    key = next((k for k in ["SNR", "Nmem", "rich"] if k in df.columns), None)
    if key:
        df = df.sort_values(by=key, ascending=False).head(max_rows).reset_index(drop=True)
        print(f"[info] {catalog:12s}: kept top {max_rows:6,} rows (sorted by {key})")
    else:
        df = df.head(max_rows).reset_index(drop=True)
    return df

def normalize_catalog(df_raw, catalog, colmap, zmax):
    df = pd.DataFrame()
    df["row_id"] = np.arange(len(df_raw))
    df["catalog"] = catalog
    ra_c, dec_c = find_radec_columns(df_raw)
    df["ra_deg"] = pd.to_numeric(df_raw[ra_c], errors="coerce")
    df["dec_deg"] = pd.to_numeric(df_raw[dec_c], errors="coerce")
    
    for std_key, raw_col in colmap.items():
        if raw_col in df_raw.columns:
            if std_key == "name":
                df[std_key] = [apply_catalog_prefix(v, colmap["name"], ra=r, dec=d) 
                               for v, r, d in zip(df_raw[raw_col], df["ra_deg"], df["dec_deg"])]
            else:
                df[std_key] = df_raw[raw_col]
            
    if "mass" in df.columns and catalog in MASS_SCALE_TO_MSUN:
        df["mass"] = pd.to_numeric(df["mass"], errors="coerce") * MASS_SCALE_TO_MSUN[catalog]
    
    z_val = df.get("z", df.get("zsp", df.get("zph", df.get("zbest", np.nan))))
    df["z_best"] = sanitize_redshift(z_val, zmax=zmax)
    return df.dropna(subset=["ra_deg", "dec_deg"])

def build_unique_clusters(cat_dfs, radius_arcmin):
    radius = radius_arcmin * u.arcmin
    m_rows, memberships = [], []
    m_coords, next_id = None, 0

    print("\n[stage] cross-matching catalogs...")
    for df in cat_dfs:
        cat_name = df["catalog"].iloc[0]
        src_coords = SkyCoord(ra=df["ra_deg"].values*u.deg, dec=df["dec_deg"].values*u.deg)
        if m_coords is None:
            print(f" -> initializing master with {cat_name} ({len(df)} clusters)")
            for idx in range(len(df)):
                m_rows.append({"master_id": next_id, "ra_deg": df.iloc[idx]["ra_deg"], "dec_deg": df.iloc[idx]["dec_deg"]})
                memberships.append({"master_id": next_id, "catalog": cat_name, "row_id": df.iloc[idx]["row_id"], "sep_arcsec": 0.0})
                next_id += 1
            m_coords = src_coords
            continue
            
        idx, sep2d, _ = src_coords.match_to_catalog_sky(m_coords)
        sep_ok = sep2d <= radius
        new_count, match_count = 0, 0
        for k in range(len(df)):
            if sep_ok[k]:
                mid = m_rows[idx[k]]["master_id"]
                memberships.append({"master_id": mid, "catalog": cat_name, "row_id": df.iloc[k]["row_id"], "sep_arcsec": sep2d[k].to(u.arcsec).value})
                match_count += 1
            else:
                m_rows.append({"master_id": next_id, "ra_deg": df.iloc[k]["ra_deg"], "dec_deg": df.iloc[k]["dec_deg"]})
                memberships.append({"master_id": next_id, "catalog": cat_name, "row_id": df.iloc[k]["row_id"], "sep_arcsec": 0.0})
                next_id += 1
                new_count += 1
        print(f" -> {cat_name:12s} matches: {match_count:6,}, unique: {new_count:6,}")
        m_coords = SkyCoord(ra=[r["ra_deg"] for r in m_rows]*u.deg, dec=[r["dec_deg"] for r in m_rows]*u.deg)
    return pd.DataFrame(m_rows), pd.DataFrame(memberships)

def apply_priority_best(master_df, membership_df, per_cat_norm, priority, field):
    cat_idx = {cat: df.set_index("row_id") for cat, df in per_cat_norm.items()}
    best_vals = []
    for mid in master_df["master_id"]:
        mems = membership_df[membership_df["master_id"] == mid].sort_values("sep_arcsec")
        val = np.nan
        for p_cat in priority:
            match = mems[mems["catalog"] == p_cat]
            if not match.empty:
                rid = match.iloc[0]["row_id"]
                v = cat_idx[p_cat].at[rid, field] if field in cat_idx[p_cat].columns else np.nan
                if pd.notna(v) and str(v).strip() != "":
                    val = v
                    break
        if pd.isna(val) or str(val).strip() == "":
            for _, m_row in mems.iterrows():
                c, rid = m_row["catalog"], m_row["row_id"]
                if c in cat_idx and field in cat_idx[c].columns:
                    v = cat_idx[c].at[rid, field]
                    if pd.notna(v) and str(v).strip() != "":
                        val = v
                        break
        best_vals.append(val)
    return pd.Series(best_vals)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=Path, default=Path("data"))
    p.add_argument("--outdir", type=Path, default=Path("output"))
    p.add_argument("--radius-arcmin", type=float, default=1.0)
    p.add_argument("--zmax", type=float, default=5.0)
    p.add_argument("--max-large", type=int, default=10000)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    per_cat_norm = {}
    print("\n[stage] loading and normalizing catalogs...")
    for spec in CATALOGS:
        path = args.indir / spec.csv_file
        if path.exists():
            per_cat_norm[spec.name] = normalize_catalog(pd.read_csv(path), spec.name, spec.colmap, args.zmax)
            print(f" -> {spec.name:12s} loaded: {len(per_cat_norm[spec.name]):7,}")

    sizes = sorted(((k, len(v)) for k, v in per_cat_norm.items()), key=lambda x: x[1], reverse=True)
    top4 = [k for k, _ in sizes[:4]]
    for k in top4: per_cat_norm[k] = downselect_large_catalog(per_cat_norm[k], k, args.max_large)

    master_df, membership_df = build_unique_clusters(list(per_cat_norm.values()), args.radius_arcmin)
    print(f"\n[info] final master catalog size: {len(master_df):,}")

    print("\n[stage] applying column selection (priority + fallback)...")
    prio_names = ["abell", "psz2", "spt", "act_hilton", "mcxc"]
    master_df["name_best"] = apply_priority_best(master_df, membership_df, per_cat_norm, prio_names, "name").astype(str)
    master_df["z_best"] = sanitize_redshift(apply_priority_best(master_df, membership_df, per_cat_norm, prio_names, "z_best"), args.zmax)
    master_df["mass_best"] = sanitize_mass_for_cluster_use(apply_priority_best(master_df, membership_df, per_cat_norm, prio_names, "mass"))

    print("\n[stage] aggregating mass statistics...")
    cat_idx = {cat: df.set_index("row_id") for cat, df in per_cat_norm.items()}
    stats = []
    for mid in master_df["master_id"]:
        mems = membership_df[membership_df["master_id"] == mid]
        m_vals = [sanitize_mass_for_cluster_use(cat_idx[r["catalog"]].at[r["row_id"], "mass"]) 
                  for _, r in mems.iterrows() if "mass" in cat_idx[r["catalog"]].columns]
        m_vals = [v for v in m_vals if pd.notna(v)]
        stats.append({
            "master_id": mid, "n_catalogs": len(mems),
            "mass_min": np.min(m_vals) if m_vals else np.nan,
            "mass_max": np.max(m_vals) if m_vals else np.nan,
            "mass_std": np.std(m_vals) if len(m_vals) > 1 else np.nan
        })
    master_df = master_df.merge(pd.DataFrame(stats), on="master_id")

    print("\n[stage] merging suffixed catalog-specific columns...")
    for spec in CATALOGS:
        if spec.name in per_cat_norm:
            # FIX: Prevent row count ballooning by deduplicating memberships per master_id
            # Sort by separation and keep only the closest match per catalog
            sub = membership_df[membership_df["catalog"] == spec.name].copy()
            sub = sub.sort_values("sep_arcsec").drop_duplicates(subset=["master_id"], keep="first")
            sub = sub[["master_id", "row_id"]]
            
            cat_subset = per_cat_norm[spec.name][["row_id"] + list(spec.colmap.keys())]
            merged = sub.merge(cat_subset, on="row_id").drop(columns="row_id")
            merged.columns = [f"{c}__{spec.name}" if c != "master_id" else c for c in merged.columns]
            if f"name__{spec.name}" in merged.columns:
                merged[f"name__{spec.name}"] = merged[f"name__{spec.name}"].astype(str)
            
            # Use left join to ensure we only attach columns to the master list
            master_df = master_df.merge(merged, on="master_id", how="left")

    master_df["ra_wrapped"] = np.where(master_df["ra_deg"] > 180, master_df["ra_deg"] - 360, master_df["ra_deg"])

    # Final guardrail to ensure master_id remains unique
    master_df = master_df.drop_duplicates(subset=["master_id"])

    master_df.to_parquet(args.outdir / "master_clusters.parquet", index=False)
    print(f"\n[done] Saved {len(master_df):,} clusters.")

if __name__ == "__main__":
    main()