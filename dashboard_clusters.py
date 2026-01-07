#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Optional

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

# Styling constants
DARK_GOLD = "#B8860B"  # dark goldenrod
HILITE_RED = "red"
MZ_GRAY = "#7f7f7f"

# Make the top row plots smaller vertically
PLOT_HEIGHT = 200
PLOT_MARGIN = dict(l=0, r=0, t=0, b=0)

SURVEYS = {
    "Pan-STARRS DR1 color": "P/PanSTARRS/DR1/color-i-r-g",
    "DSS2 color": "P/DSS2/color",
    "DESI Legacy Survey (DR10) color": "P/DESI-Legacy-Surveys/DR10/color",
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
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)

        # Ensure master_id is numeric and consistent
        if "master_id" in df.columns:
            df["master_id"] = pd.to_numeric(df["master_id"], errors="coerce").astype("Int64")

        if "ra_wrapped" not in df.columns and "ra_deg" in df.columns:
            df["ra_wrapped"] = np.where(df["ra_deg"] > 180, df["ra_deg"] - 360, df["ra_deg"])

        for col in ["z_best", "mass_best", "n_catalogs", "ra_deg", "dec_deg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    return None


@st.cache_data(show_spinner=False)
def get_static_bg(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=min(len(df), 15000), random_state=0)


def fmt_float(x, fmt="{:.4g}"):
    if x is None or pd.isna(x):
        return "—"
    try:
        xf = float(x)
        return fmt.format(xf) if np.isfinite(xf) else "—"
    except Exception:
        return str(x)


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
    if not name_cols:
        blob = pd.Series([""] * len(master), index=master.index)
        compact = blob.copy()
        return blob, compact

    blob = master[name_cols[0]].fillna("").astype(str)
    for c in name_cols[1:]:
        blob = blob + " | " + master[c].fillna("").astype(str)
    compact = blob.str.lower().str.replace(r"[\s\-\_\(\)\[\]\{\}\,\.\;\:]", "", regex=True)
    return blob.str.lower(), compact


def apply_search(master, blob, blob_compact, query):
    q = normalize_query(query)
    if not q:
        return master.copy()
    m = re.fullmatch(r"(abell|a|aco)(\d{1,5})", re.sub(r"\s+", "", q))
    if m:
        return master.loc[blob_compact.str.contains(re.sub(r"\s+", "", q), na=False)].copy()
    if q.isdigit():
        return master.loc[blob.str.contains(rf"\b{re.escape(q)}\b", na=False, regex=True)].copy()
    return master.loc[blob.str.contains(q, na=False, regex=False)].copy()


def apply_spatial_filter(df, ra_target, dec_target, radius_arcmin):
    if ra_target is None or dec_target is None:
        return df
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
        if col not in out.columns:
            continue
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
        if f"name__{cat}" in out.columns:
            out = out[out[f"name__{cat}"].notna()]
    return out


# --- Default-only sorting helpers (Abell/ACO/A<number> first, numeric order) ---
_ABELL_RE = re.compile(r"^\s*(abell|aco|a)\s*[-_]?\s*(\d{1,5})\b", re.IGNORECASE)


def abell_group_and_number(name: str) -> tuple[int, int]:
    """
    Returns (group, number)
      group = 0 for Abell/ACO/A<number> names
      group = 1 for everything else
      number = extracted numeric id (int) else large sentinel
    """
    if not isinstance(name, str):
        return 1, 10**9
    m = _ABELL_RE.match(name.strip())
    if not m:
        return 1, 10**9
    try:
        return 0, int(m.group(2))
    except Exception:
        return 0, 10**9


# --- Visualizations ---
def render_plots(bg_sample, filtered_df, selected_id):
    p1, p2, p3 = st.columns([1, 1, 1])

    # Sky distribution
    with p1:
        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=bg_sample["dec_deg"],
            lon=bg_sample["ra_wrapped"],
            mode="markers",
            marker=dict(color="#E5ECF6", size=2),
            name="All",
        ))
        disp_df = filtered_df if len(filtered_df) < 5000 else filtered_df.sample(5000, random_state=0)
        fig.add_trace(go.Scattergeo(
            lat=disp_df["dec_deg"],
            lon=disp_df["ra_wrapped"],
            mode="markers",
            marker=dict(color="black", size=4, opacity=0.6),
            name="Filtered",
        ))

        sel_df = filtered_df[filtered_df["master_id"] == selected_id]
        if not sel_df.empty:
            fig.add_trace(go.Scattergeo(
                lat=sel_df["dec_deg"],
                lon=sel_df["ra_wrapped"],
                mode="markers",
                marker=dict(color=HILITE_RED, size=12, symbol="circle", line=dict(color="white", width=2)),
                name="Selected",
            ))

        fig.update_geos(
            projection_type="mollweide",
            showland=False,
            showcoastlines=False,
            showframe=True,
            bgcolor="white",
        )
        fig.update_layout(height=PLOT_HEIGHT, margin=PLOT_MARGIN, uirevision="constant", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Redshift histogram
    with p2:
        z_fg = pd.to_numeric(filtered_df.get("z_best", pd.Series([], dtype=float)), errors="coerce").dropna()
        fig_z = go.Figure()
        fig_z.add_trace(go.Histogram(
            x=z_fg,
            nbinsx=50,
            name="Filtered",
            marker=dict(
                color="dodgerblue",
                line=dict(color=DARK_GOLD, width=1.0),  # edgecolor dark gold
            ),
        ))

        sel_df = filtered_df[filtered_df["master_id"] == selected_id]
        if not sel_df.empty:
            z_sel = pd.to_numeric(sel_df.iloc[0].get("z_best"), errors="coerce")
            if pd.notna(z_sel) and np.isfinite(z_sel):
                fig_z.add_vline(x=float(z_sel), line_width=3, line_color=HILITE_RED)

        fig_z.update_layout(
            height=PLOT_HEIGHT,
            margin=PLOT_MARGIN,
            uirevision="constant",
            xaxis_title="Redshift (z)",
            showlegend=False,
        )
        st.plotly_chart(fig_z, use_container_width=True)

    # Mass vs redshift
    with p3:
        z = pd.to_numeric(filtered_df.get("z_best", pd.Series([], dtype=float)), errors="coerce")
        m14 = pd.to_numeric(filtered_df.get("mass_best", pd.Series([], dtype=float)), errors="coerce") / MASS_UNIT
        ok = z.notna() & m14.notna() & np.isfinite(z) & np.isfinite(m14)

        if ok.sum() == 0:
            st.info("No clusters with both z and mass in current filter.")
            return

        df_sc = filtered_df.loc[ok, ["master_id"]].copy()
        df_sc["z"] = z[ok].values
        df_sc["m14"] = m14[ok].values

        # keep selected row even if we downsample
        sel_row = df_sc[df_sc["master_id"] == selected_id]
        if len(df_sc) > 7000:
            df_sc = df_sc.sample(7000, random_state=0)
            if (not sel_row.empty) and (df_sc["master_id"] == selected_id).sum() == 0:
                df_sc = pd.concat([df_sc, sel_row], ignore_index=True)

        fig_mz = go.Figure()

        # Base markers: gray
        fig_mz.add_trace(go.Scatter(
            x=df_sc["z"],
            y=df_sc["m14"],
            mode="markers",
            name="Clusters",
            marker=dict(size=7, color=MZ_GRAY, opacity=0.75),
            customdata=df_sc["master_id"],
            hovertemplate="z=%{x:.4f}<br>M=%{y:.3g}e14 Msun<br>master_id=%{customdata}<extra></extra>",
            showlegend=False,
        ))

        # highlight selected: red
        sel = df_sc[df_sc["master_id"] == selected_id]
        if not sel.empty:
            fig_mz.add_trace(go.Scatter(
                x=sel["z"],
                y=sel["m14"],
                mode="markers",
                name="Selected",
                marker=dict(size=14, color=HILITE_RED, symbol="circle", line=dict(width=2, color="white")),
                hovertemplate="Selected<br>z=%{x:.4f}<br>M=%{y:.3g}e14 Msun<extra></extra>",
                showlegend=False,
            ))

        fig_mz.update_layout(
            height=PLOT_HEIGHT,
            margin=PLOT_MARGIN,
            uirevision="constant",
            xaxis_title="Redshift (z)",
            yaxis_title="Mass (1e14 Msun)",
        )
        st.plotly_chart(fig_mz, use_container_width=True)


def aladin_html(ra, dec, fov, survey_id, label, expand_sidebar=False):
    survey_js = f"survey: '{survey_id}'," if survey_id else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body {{ margin:0; padding:0; }}
    #al {{ width:100%; height:500px; }}
  </style>
</head>
<body>
  <div id="al"></div>
  <script>
    A.init.then(() => {{
      const a = A.aladin('#al', {{
        target: '{ra} {dec}',
        fov: {fov/60.0},
        {survey_js}
        showLayersControl: true,
        expandLayersControl: {str(bool(expand_sidebar)).lower()},
        showGotoControl: true,
        showZoomControl: true,
        showFullscreenControl: true,
        showContextMenu: true,
        showShareControl: true
      }});

      const cat = A.catalog({{name: 'Target', sourceSize: 18}});
      a.addCatalog(cat);
      cat.addSources([A.source({ra}, {dec}, {{name: '{label}'}})]);
    }});
  </script>
</body>
</html>"""


def main():
    args = parse_args()
    st.set_page_config(page_title="Cluster Dashboard", layout="wide")
    st.markdown(
    """
    <style>
      /* reduce extra whitespace above/between blocks */
      .block-container { padding-top: 2.0rem; padding-bottom: 1rem; }

      /* tighten vertical spacing between elements */
      div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
    master = load_data(args.master)
    if master is None or master.empty:
        st.stop()

    blob, blob_compact = build_search_blobs(master)
    bg_sample = get_static_bg(master)

    # default ranges
    Z_MIN, Z_MAX = 0.0, 2.0

    m_vals = pd.to_numeric(master.get("mass_best", pd.Series([], dtype=float)), errors="coerce")
    m_min_raw = m_vals.min(skipna=True)
    m_max_raw = m_vals.max(skipna=True)
    M_MIN = (float(m_min_raw) if pd.notna(m_min_raw) else 0.0) / MASS_UNIT
    M_MAX = (float(m_max_raw) if pd.notna(m_max_raw) else 15.0) / MASS_UNIT
    if not np.isfinite(M_MIN):
        M_MIN = 0.0
    if not np.isfinite(M_MAX) or M_MAX <= 0:
        M_MAX = 15.0

    if "selected_master_id" not in st.session_state or st.session_state["selected_master_id"] is None:
        st.session_state["selected_master_id"] = int(master["master_id"].dropna().iloc[0])

    # -------------------------
    # SIDEBAR
    # -------------------------
    with st.sidebar:
        st.title("Search & Filters")

        name_query = st.text_input(
            "Search Name",
            value="",
            key="q_name",
            placeholder="e.g., ACO 2744, MACS1149, PSZ2...",
        )

        st.subheader("Spatial Search")
        rc1, rc2 = st.columns(2)
        ra_v = rc1.number_input(
            "RA (deg)",
            value=st.session_state.get("f_ra_search"),
            key="f_ra_search",
            placeholder="RA",
        )
        dec_v = rc2.number_input(
            "Dec (deg)",
            value=st.session_state.get("f_dec_search"),
            key="f_dec_search",
            placeholder="Dec",
        )
        rad_v = st.number_input("Radius (arcmin)", value=st.session_state.get("f_rad_search", 10.0), key="f_rad_search")
        st.button("Clear Spatial", on_click=reset_spatial, use_container_width=True)

        st.divider()
        st.subheader("Scientific Filters")

        zc1, zc2 = st.columns(2)
        z_min = zc1.number_input("Min z", value=st.session_state.get("f_zmin", Z_MIN), format="%.3f", key="f_zmin")
        z_max = zc2.number_input("Max z", value=st.session_state.get("f_zmax", Z_MAX), format="%.3f", key="f_zmax")
        st.button("Reset z", on_click=reset_redshift, args=(Z_MIN, Z_MAX), use_container_width=True)

        st.divider()

        mc1, mc2 = st.columns(2)
        m_min = mc1.number_input("Min M (1e14 Msun)", value=st.session_state.get("f_mmin", M_MIN), format="%.2f", key="f_mmin")
        m_max = mc2.number_input("Max M (1e14 Msun)", value=st.session_state.get("f_mmax", M_MAX), format="%.2f", key="f_mmax")
        st.button("Reset M", on_click=reset_mass, args=(M_MIN, M_MAX), use_container_width=True)

        st.divider()

        cats = sorted({c.split("__")[1] for c in master.columns if "__" in c})
        required_cats = st.multiselect("Require Catalogs", cats, key="f_cats")

        nc1, nc2 = st.columns(2)
        n_min = nc1.number_input("N catalogs min", value=0, key="f_nmin")
        n_max = nc2.number_input("N catalogs max", value=50, key="f_nmax")

        qc1, qc2, qc3 = st.columns(3)
        keep_nans = qc1.checkbox("Keep NaNs", value=True, key="f_nan")
        must_z = qc2.checkbox("Must z", value=False, key="f_mustz")
        must_m = qc3.checkbox("Must M", value=False, key="f_mustm")

    # -------------------------
    # Filtering
    # -------------------------
    view = apply_search(master, blob, blob_compact, name_query)
    view = apply_spatial_filter(view, ra_v, dec_v, rad_v)
    view = apply_filters(
        view,
        (z_min, z_max),
        (m_min * MASS_UNIT, m_max * MASS_UNIT),
        (n_min, n_max),
        required_cats,
        keep_nans,
        must_z,
        must_m,
    )

    # -------------------------
    # TITLE MOVED TO TOP (above plots)
    # -------------------------
    # make the title small so it doesn't move when the table updates
    st.markdown(f"### Clusters ({len(view):,})")

    # plot slot: render AFTER selection updates
    plots_slot = st.container()

    # -------------------------
    # Table controls
    # -------------------------
    st.divider()

    sc1, sc2, sc3 = st.columns([1.5, 1, 1])
    with sc1:
        sort_by = st.selectbox("Sort Whole Catalog By:", ["name_best", "z_best", "mass_best", "n_catalogs", "ra_deg"], index=0)
    with sc2:
        sort_order = st.radio("Order:", ["Ascending", "Descending"], index=0 if sort_by == "name_best" else 1, horizontal=True)
    with sc3:
        max_rows = st.number_input("Display Limit", 100, 20000, 2000)

    # default-only sorting: Abell/ACO/A<number> first, numerically ordered
    if sort_by in view.columns:
        if sort_by == "name_best" and sort_order == "Ascending":
            names = view.get("name_best", pd.Series([""] * len(view), index=view.index)).astype(str)
            grp_num = names.apply(abell_group_and_number)
            view["_abell_group"] = [g for g, n in grp_num]
            view["_abell_num"] = [n for g, n in grp_num]
            view = view.sort_values(
                by=["_abell_group", "_abell_num", "name_best"],
                ascending=[True, True, True],
                na_position="last",
            )
            view = view.drop(columns=["_abell_group", "_abell_num"], errors="ignore")
        else:
            view = view.sort_values(by=sort_by, ascending=(sort_order == "Ascending"), na_position="last")

    # -------------------------
    # Export full filtered catalog
    # -------------------------
    csv_export = view.copy()

    # safety: remove any temporary helper columns
    csv_export = csv_export.drop(columns=["_abell_group", "_abell_num"], errors="ignore")

    # nice-to-have: master_id first
    if "master_id" in csv_export.columns:
        cols = ["master_id"] + [c for c in csv_export.columns if c != "master_id"]
        csv_export = csv_export[cols]

    csv_bytes = csv_export.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download all {len(csv_export):,} filtered clusters (CSV)",
        data=csv_bytes,
        file_name="filtered_clusters.csv",
        mime="text/csv",
    )
    # -------------------------
    # Layout: table + details
    # -------------------------
    left, right = st.columns([1.2, 1.8], gap="large")

    with left:
        display_df = view.head(int(max_rows)).copy()

        gb = GridOptionsBuilder.from_dataframe(display_df)

        # hide master_id from display but keep it for selection
        if "master_id" in display_df.columns:
            gb.configure_column("master_id", hide=True)

        gb.configure_default_column(resizable=True, filter=True, minWidth=100, flex=1, sortable=False)
        if "name_best" in display_df.columns:
            gb.configure_column("name_best", pinned="left", width=180, flex=0)
        gb.configure_selection(selection_mode="single")

        grid_key = f"grid_{sort_by}_{sort_order}_{len(view)}_{name_query[:3]}"
        grid_resp = AgGrid(
            display_df,
            gridOptions=gb.build(),
            height=500,
            theme="streamlit",
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
            key=grid_key,
            update_mode="MODEL_CHANGED",
        )

        sel = grid_resp.get("selected_rows", [])
        new_selected_id = None

        # st_aggrid can return list[dict] or DataFrame depending on version/config
        if isinstance(sel, pd.DataFrame) and not sel.empty:
            new_selected_id = sel.iloc[0].get("master_id")
        elif isinstance(sel, list) and len(sel) > 0:
            new_selected_id = sel[0].get("master_id")

        if new_selected_id is not None and pd.notna(new_selected_id):
            st.session_state["selected_master_id"] = int(new_selected_id)

    # NOW render plots using the freshly-updated selected id
    with plots_slot:
        render_plots(bg_sample, view, st.session_state["selected_master_id"])

    # -------------------------
    # Details panel
    # -------------------------
    sel_id = st.session_state["selected_master_id"]
    match = master[master["master_id"] == sel_id]
    if not match.empty:
        row = match.iloc[0]
        with right:
            dc1, dc2 = st.columns([1, 1.4])
            with dc1:
                st.subheader("Summary")
                st.markdown(f"**Best Name:** {row.get('name_best', '—')}")
                st.markdown(f"**Redshift (z):** {fmt_float(row.get('z_best'))}")
                mb = row.get("mass_best")
                st.markdown(
                    f"**Mass:** {fmt_float((mb / MASS_UNIT) if pd.notna(mb) else np.nan)} "
                    r"$10^{14}\,M_{\odot}$"
                )
                with st.expander("Alternate Names"):
                    alts = []
                    for c in master.columns:
                        if c.startswith("name__") and pd.notna(row.get(c)):
                            alts.append(str(row.get(c)))
                    st.write(", ".join(alts) if alts else "—")

                ra0 = row.get("ra_deg")
                dec0 = row.get("dec_deg")
                if pd.notna(ra0) and pd.notna(dec0):
                    st.caption(f"RA: {float(ra0):.4f}, Dec: {float(dec0):.4f}")
                else:
                    st.caption("RA/Dec: —")

            with dc2:
                survey = st.selectbox("Aladin Survey", list(SURVEYS.keys()), key="al_srv")
                if pd.notna(row.get("ra_deg")) and pd.notna(row.get("dec_deg")):
                    st.components.v1.html(
                        aladin_html(
                            float(row["ra_deg"]),
                            float(row["dec_deg"]),
                            args.fov_arcmin,
                            SURVEYS[survey],
                            str(row.get("name_best", "selected")),
                            expand_sidebar=False,  # layers menu starts closed
                        ),
                        height=500,
                    )
                else:
                    st.info("Selected row is missing RA/Dec; cannot render Aladin view.")


if __name__ == "__main__":
    main()
