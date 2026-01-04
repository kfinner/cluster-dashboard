#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from astropy.coordinates import SkyCoord
import astropy.units as u
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode

# --- Constants ---
DEFAULT_FOV_ARCMIN = 12.0
MASS_UNIT = 1e14 

SURVEYS: Dict[str, Optional[str]] = {
    "Pan-STARRS DR1 color": "P/PanSTARRS/DR1/color-i-r-g",
    "DSS2 color": "P/DSS2/color",
    "DESI Legacy Survey (DR9) color": "P/DESI-Legacy-Surveys/DR9/color",
    "2MASS color": "P/2MASS/color",
    "WISE color": "P/WISE/color",
    "Aladin default": None,
}

# --- Helpers ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, default=Path("master_clusters.parquet"))
    p.add_argument("--fov-arcmin", type=float, default=DEFAULT_FOV_ARCMIN)
    return p.parse_args()

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)
        if "ra_wrapped" not in df.columns and "ra_deg" in df.columns:
            df["ra_wrapped"] = np.where(df["ra_deg"] > 180, df["ra_deg"] - 360, df["ra_deg"])
        
        for col in ["z_best", "mass_best", "n_catalogs", "ra_deg", "dec_deg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return None

@st.cache_data(show_spinner=False)
def get_static_bg(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=min(len(df), 15000))

def fmt_float(x, fmt="{:.4g}"):
    if x is None or pd.isna(x): return "—"
    try:
        xf = float(x)
        return fmt.format(xf) if np.isfinite(xf) else "—"
    except: return str(x)

# --- State & Filter Logic ---
def reset_redshift(z_min, z_max):
    st.session_state["f_zmin"], st.session_state["f_zmax"] = z_min, z_max
def reset_mass(m_min, m_max):
    st.session_state["f_mmin"], st.session_state["f_mmax"] = m_min, m_max
def reset_spatial():
    st.session_state["f_ra_search"], st.session_state["f_dec_search"] = None, None
    st.session_state["f_rad_search"] = 10.0

def normalize_query(q: str):
    return re.sub(r"\s+", " ", q.strip()).lower()

@st.cache_data(show_spinner=False)
def build_search_blobs(master: pd.DataFrame):
    name_cols = [c for c in master.columns if c == "name_best" or c.startswith("name__")]
    blob = master[name_cols[0]].fillna("").astype(str)
    for c in name_cols[1:]:
        blob = blob + " | " + master[c].fillna("").astype(str)
    compact = blob.str.lower().str.replace(r"[\s\-\_\(\)\[\]\{\}\,\.\;\:]", "", regex=True)
    return blob.str.lower(), compact

def apply_search(master, blob, blob_compact, query):
    q = normalize_query(query)
    if not q: return master.copy()
    m = re.fullmatch(r"(abell|a|aco)(\d{1,5})", re.sub(r"\s+", "", q))
    if m:
        return master.loc[blob_compact.str.contains(re.sub(r"\s+", "", q), na=False)].copy()
    if q.isdigit():
        return master.loc[blob.str.contains(rf"\b{re.escape(q)}\b", na=False, regex=True)].copy()
    return master.loc[blob.str.contains(q, na=False, regex=False)].copy()

def apply_spatial_filter(df, ra_target, dec_target, radius_arcmin):
    if ra_target is None or dec_target is None: return df
    cat_coords = SkyCoord(ra=df["ra_deg"].values * u.deg, dec=df["dec_deg"].values * u.deg)
    target_coord = SkyCoord(ra=ra_target * u.deg, dec=dec_target * u.deg)
    sep = target_coord.separation(cat_coords)
    mask = sep <= (radius_arcmin * u.arcmin)
    out = df[mask].copy()
    out.loc[:, "sep_arcmin"] = sep[mask].to(u.arcmin).value
    return out

def apply_filters(df, z_range, mass_range, ncat_range, required_cats, keep_nans, must_z, must_m):
    out = df.copy()
    for col, rng, must in [("z_best", z_range, must_z), ("mass_best", mass_range, must_m)]:
        v = pd.to_numeric(out[col], errors="coerce")
        if must:
            out = out[v.notna()]
            v = pd.to_numeric(out[col], errors="coerce")
        if keep_nans:
            mask = v.isna() | ((v >= rng[0]) & (v <= rng[1]))
        else:
            mask = v.notna() & (v >= rng[0]) & (v <= rng[1])
        out = out[mask]

    if "n_catalogs" in out.columns:
        n = pd.to_numeric(out["n_catalogs"], errors="coerce").fillna(0)
        out = out[(n >= ncat_range[0]) & (n <= ncat_range[1])]
    for cat in required_cats:
        if f"name__{cat}" in out.columns: out = out[out[f"name__{cat}"].notna()]
    return out

# --- Visualizations ---
@st.fragment
def render_plots(bg_sample, filtered_df, selected_id):
    p1, p2 = st.columns([1, 1])
    with p1:
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(lat=bg_sample["dec_deg"], lon=bg_sample["ra_wrapped"], mode='markers', marker=dict(color="#E5ECF6", size=2), name="All"))
        disp_df = filtered_df if len(filtered_df) < 5000 else filtered_df.sample(5000)
        fig.add_trace(go.Scattergeo(lat=disp_df["dec_deg"], lon=disp_df["ra_wrapped"], mode='markers', marker=dict(color="black", size=4, opacity=0.6), name="Filtered"))
        sel_df = filtered_df[filtered_df["master_id"] == selected_id]
        if not sel_df.empty:
            fig.add_trace(go.Scattergeo(lat=sel_df["dec_deg"], lon=sel_df["ra_wrapped"], mode='markers', marker=dict(color="red", size=12, symbol="circle", line=dict(color='white', width=2)), name="Selected"))
        fig.update_geos(projection_type="mollweide", showland=False, showcoastlines=False, showframe=True, bgcolor="white")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), uirevision='constant', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        z_fg = pd.to_numeric(filtered_df["z_best"], errors="coerce").dropna()
        fig_z = go.Figure()
        fig_z.add_trace(go.Histogram(x=z_fg, nbinsx=50, name="Filtered", marker_color="black"))
        fig_z.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), uirevision='constant', xaxis_title="Redshift (z)", showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)

def aladin_html(ra, dec, fov, survey_id, label):
    return f"""<!doctype html><html><head><script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script></head><body style="margin:0"><div id="al" style="width:100%;height:500px"></div><script>A.init.then(()=>{{var a=A.aladin('#al',{{target:'{ra} {dec}',fov:{fov/60},survey:'{survey_id if survey_id else ""}'}});a.addCatalog(A.catalog().addSources([A.source({ra},{dec},{{name:'{label}'}})]));}});</script></body></html>"""

def main():
    args = parse_args()
    st.set_page_config(page_title="Cluster Dashboard", layout="wide")
    master = load_data(args.master)
    if master is None: st.stop()

    blob, blob_compact = build_search_blobs(master)
    bg_sample = get_static_bg(master)

    z_vals = pd.to_numeric(master["z_best"], errors="coerce")
    Z_MIN, Z_MAX = float(z_vals.min() or 0.0), float(z_vals.max() or 3.0)
    m_vals = pd.to_numeric(master["mass_best"], errors="coerce")
    M_MIN, M_MAX = float(m_vals.min() or 0.0)/MASS_UNIT, float(m_vals.max() or 15.0)/MASS_UNIT
    
    if "selected_master_id" not in st.session_state:
        st.session_state["selected_master_id"] = int(master.iloc[0]["master_id"])

    # --- TOP SEARCH BAR ---
    name_query = st.text_input("Search Name:", value="", key="q_name")

    with st.expander("Scientific Filters", expanded=True):
        r1_c1, r1_c2, r1_c3 = st.columns([1, 1, 1.8])
        with r1_c1:
            zc1, zc2 = st.columns(2)
            z_min = zc1.number_input("Min z", value=st.session_state.get("f_zmin", Z_MIN), format="%.3f", key="f_zmin")
            z_max = zc2.number_input("Max z", value=st.session_state.get("f_zmax", Z_MAX), format="%.3f", key="f_zmax")
            st.button("Reset z", on_click=reset_redshift, args=(Z_MIN, Z_MAX), width="stretch")
        with r1_c2:
            mc1, mc2 = st.columns(2)
            m_min = mc1.number_input("Min M", value=st.session_state.get("f_mmin", M_MIN), format="%.2f", key="f_mmin")
            m_max = mc2.number_input("Max M", value=st.session_state.get("f_mmax", M_MAX), format="%.2f", key="f_mmax")
            st.button("Reset M", on_click=reset_mass, args=(M_MIN, M_MAX), width="stretch")
        with r1_c3:
            rc1, rc2, rc3, rc4 = st.columns([1, 1, 0.7, 1])
            ra_v = rc1.number_input("RA", value=st.session_state.get("f_ra_search"), key="f_ra_search", placeholder="RA")
            dec_v = rc2.number_input("Dec", value=st.session_state.get("f_dec_search"), key="f_dec_search", placeholder="Dec")
            rad_v = rc3.number_input("Rad", value=st.session_state.get("f_rad_search", 10.0), key="f_rad_search")
            rc4.button("Clear Spatial", on_click=reset_spatial, width="stretch")
        
        st.divider()
        r2_c1, r2_c2, r2_c3 = st.columns([2, 1, 1.5])
        with r2_c1: required_cats = st.multiselect("Require Catalogs", sorted({c.split("__")[1] for c in master.columns if "__" in c}), key="f_cats")
        with r2_c2:
            nc1, nc2 = st.columns(2)
            n_min = nc1.number_input("Nmin", value=0, key="f_nmin")
            n_max = nc2.number_input("Nmax", value=50, key="f_nmax")
        with r2_c3:
            qc = st.columns(3)
            qc[0].checkbox("Keep NaNs", value=True, key="f_nan")
            qc[1].checkbox("Must z", value=False, key="f_mustz")
            qc[2].checkbox("Must M", value=False, key="f_mustm")

    # Filter Pipeline
    view = apply_search(master, blob, blob_compact, name_query)
    view = apply_spatial_filter(view, ra_v, dec_v, rad_v)
    view = apply_filters(view, (z_min, z_max), (m_min*MASS_UNIT, m_max*MASS_UNIT), (n_min, n_max), required_cats, st.session_state.f_nan, st.session_state.f_mustz, st.session_state.f_mustm)

    render_plots(bg_sample, view, st.session_state["selected_master_id"])

    # --- THE CLUSTERS SECTION WITH INTEGRATED SORTING ---
    st.divider()
    st.header(f"Clusters ({len(view):,})")
    
    # Sorting UI placed right under the title
    sc1, sc2, sc3 = st.columns([1.5, 1, 1])
    with sc1:
        sort_by = st.selectbox("Sort Whole Catalog By:", 
                              ["name_best", "z_best", "mass_best", "n_catalogs", "ra_deg"], index=0)
    with sc2:
        sort_order = st.radio("Order:", ["Ascending", "Descending"], index=0 if sort_by == "name_best" else 1, horizontal=True)
    with sc3:
        max_rows = st.number_input("Display Limit", 100, 20000, 2000)

    # Apply Global Pandas Sort
    if sort_by in view.columns:
        view = view.sort_values(by=sort_by, ascending=(sort_order == "Ascending"), na_position='last')

    left, right = st.columns([1.5, 1.5], gap="large") 
    with left:
        display_df = view.head(int(max_rows)).copy()

        gb = GridOptionsBuilder.from_dataframe(display_df)
        gb.configure_default_column(resizable=True, filter=True, minWidth=100, flex=1, sortable=False)
        gb.configure_column("name_best", pinned='left', width=180, flex=0)
        gb.configure_selection(selection_mode="single")
        
        grid_key = f"grid_{sort_by}_{sort_order}_{len(view)}_{name_query[:3]}"
        
        grid_resp = AgGrid(
            display_df, 
            gridOptions=gb.build(), 
            height=500, 
            theme="streamlit", 
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
            key=grid_key,
            update_mode="MODEL_CHANGED"
        )
        
        sel = grid_resp.get("selected_rows", [])
        if isinstance(sel, pd.DataFrame) and not sel.empty: 
            st.session_state["selected_master_id"] = int(sel.iloc[0]["master_id"])
        elif isinstance(sel, list) and len(sel) > 0: 
            st.session_state["selected_master_id"] = int(sel[0].get("master_id"))

    # Details Section
    if not master[master["master_id"] == st.session_state["selected_master_id"]].empty:
        row = master[master["master_id"] == st.session_state["selected_master_id"]].iloc[0]
        with right:
            dc1, dc2 = st.columns([1, 1.4])
            with dc1:
                st.subheader("Summary")
                st.markdown(f"**Best Name:** {row['name_best']}")
                st.markdown(f"**Redshift (z):** {fmt_float(row['z_best'])}")
                st.markdown(f"**Mass:** {fmt_float(row['mass_best']/MASS_UNIT)} $10^{{14}} M_{{\odot}}$")
                with st.expander("Alternate Names"):
                    st.write(", ".join([str(row[c]) for c in master.columns if c.startswith("name__") and pd.notna(row[c])]))
                st.caption(f"RA: {row['ra_deg']:.4f}, Dec: {row['dec_deg']:.4f}")
            with dc2:
                survey = st.selectbox("Aladin Survey", list(SURVEYS.keys()), key="al_srv")
                st.components.v1.html(aladin_html(row['ra_deg'], row['dec_deg'], args.fov_arcmin, SURVEYS[survey], row['name_best']), height=500)

if __name__ == "__main__":
    main()