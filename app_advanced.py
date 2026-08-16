"""Project Kessler -- orbital traffic control console.

Three modes:
  Seeded threat  -- deterministic demo encounter, guaranteed to fire
  Live scan      -- screen one asset against the real catalog, no seeding
  Fleet monitor  -- continuous sweep of many assets, auto-resolving breaches
"""
from __future__ import annotations

import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from kessler.agents import run_resolution
from kessler.assurance import Authorization, EngagementLog, Mode, cross_check
from kessler.bus import Bus, Event
from kessler.catalog import classify, load_demo_catalog
from kessler.conjunction import screen
from kessler.mission import (Constraints, MissionState, altitude_shortlist, find_tca,
                             requires_action, synthesize_threat)
from kessler.monitor import select_fleet, sweep_fleet
from kessler.physics import (R_EARTH, apply_burn, elements, propagate,
                             teme_positions_many, teme_state, timescale)

VOID, PANEL, LINE = "#060911", "#111A2E", "#22304C"
INK, MUTED = "#E6EBF4", "#8494AD"
AMBER, ICE, ALERT, NOMINAL = "#F2A03D", "#6FB6E8", "#FF4757", "#4FD1A5"
LEO_VIEW_KM = 8200.0

# LeoLabs' public visualizations. Both are served without authentication and
# carry no X-Frame-Options or CSP frame-ancestors directive, so they embed
# cleanly. Use is covered by LeoLabs' terms for sharing, which permit
# non-commercial educational use provided credit and a link are given --
# render_leolabs() below carries that attribution and must stay with the frame.
LEOLABS = {
    "LEO catalog (all tracked objects)": "https://platform.leolabs.space/visualizations/leo",
    "Today's conjunctions": "https://platform.leolabs.space/visualizations/conjunctions/today",
}
LEOLABS_TERMS = "https://platform.leolabs.space/visualizations_terms_for_sharing"

st.set_page_config(page_title="Project Kessler", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"""<style>
.stApp {{ background:{VOID}; }}
h1,h2,h3 {{ font-family:'Arial Narrow',Arial,sans-serif !important; letter-spacing:-.01em; }}
.kessler-log {{ background:{PANEL}; border:1px solid {LINE}; padding:.7rem .9rem;
  height:270px; overflow-y:auto; font-family:'SF Mono',Menlo,monospace; font-size:.74rem;
  line-height:1.65; white-space:pre-wrap; }}
.kessler-hd {{ font-family:'SF Mono',Menlo,monospace; font-size:.62rem; letter-spacing:.18em;
  text-transform:uppercase; color:{MUTED}; padding-bottom:.35rem; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- plumbing
@st.cache_resource(show_spinner="Pulling the CelesTrak catalog…")
def get_catalog(limit):
    return load_demo_catalog(limit=limit)


EARTH_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "earth_720x360.npy")

# Object classes, coloured the way trackers label them.
CLASS_STYLE = {
    "payload":     ("#5BE8A8", 2.2, 0.95),
    "rocket_body": ("#F2C14E", 1.9, 0.80),
    "debris":      ("#FF6B7A", 1.5, 0.55),
}

# Earth surface: ocean -> land -> ice, driven by the Blue Marble luminance map.
EARTH_SCALE = [
    [0.00, "#050B1A"], [0.18, "#0A1E3C"], [0.34, "#0E3355"],
    [0.46, "#14503F"], [0.58, "#2E6B3A"], [0.70, "#6E7A3C"],
    [0.82, "#A08B54"], [0.92, "#CFC2A0"], [1.00, "#FFFFFF"],
]


@st.cache_data(show_spinner=False)
def _earth_tex():
    return np.load(EARTH_TEX).astype(np.float32) if os.path.exists(EARTH_TEX) else None


def earth_surface(gast_hours: float, n_lon: int = 200, n_lat: int = 100):
    """Sphere mesh plus a Blue Marble luminance field for surfacecolor.

    Everything else in this app lives in TEME, which is inertial -- the Earth
    turns underneath it. Drawing the texture at fixed longitudes would park the
    continents in the wrong place relative to every satellite on screen, so the
    map is rotated by Greenwich apparent sidereal time.
    """
    tex = _earth_tex()
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    x = R_EARTH * np.cos(LAT) * np.cos(LON)
    y = R_EARTH * np.cos(LAT) * np.sin(LON)
    z = R_EARTH * np.sin(LAT)
    if tex is None:
        return x, y, z, 0.35 + 0.3 * np.cos(LAT)
    geo_lon = LON - np.radians(gast_hours * 15.0)          # TEME -> geographic
    geo_lon = (geo_lon + np.pi) % (2 * np.pi) - np.pi
    ti = ((geo_lon + np.pi) / (2 * np.pi) * (tex.shape[1] - 1)).astype(int)
    tj = ((np.pi / 2 - LAT) / np.pi * (tex.shape[0] - 1)).astype(int)
    return x, y, z, tex[tj, ti]


def sphere(radius, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    return (radius * np.outer(np.cos(u), np.sin(v)),
            radius * np.outer(np.sin(u), np.sin(v)),
            radius * np.outer(np.ones_like(u), np.cos(v)))


@st.cache_data(show_spinner=False)
def starfield(n=520, radius=7800.0, seed=4):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def leo_cloud(objects, t_grid, max_points=5200):
    """Catalog snapshot split by object class, trimmed to the LEO view."""
    pts = teme_positions_many(objects, t_grid)[:, 0, :]
    ok = ~np.isnan(pts[:, 0])
    pts, objs = pts[ok], [o for o, k in zip(objects, ok) if k]
    r = np.linalg.norm(pts, axis=1)
    keep = r < LEO_VIEW_KM
    pts, objs = pts[keep], [o for o, k in zip(objs, keep) if k]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, objs = pts[idx], [objs[i] for i in idx]
    groups = {k: [] for k in CLASS_STYLE}
    for p, o in zip(pts, objs):
        groups[classify(o.name)].append(p)
    return {k: np.array(v) for k, v in groups.items() if len(v)}


def globe_figure(cloud=None, tracks=(), markers=(), height=700, show_stars=True,
                 gast_hours: float = 0.0):
    fig = go.Figure()

    # --- atmosphere: two translucent shells for a limb glow ---
    for scale, op, col in ((1.055, 0.055, "#7FD4FF"), (1.020, 0.085, "#4FA8E8")):
        sx, sy, sz = sphere(R_EARTH * scale, 34)
        fig.add_surface(x=sx, y=sy, z=sz, showscale=False, opacity=op, hoverinfo="skip",
                        colorscale=[[0, col], [1, col]], lighting=dict(ambient=1.0, diffuse=0))

    # --- Earth ---
    ex, ey, ez, shade = earth_surface(gast_hours)
    fig.add_surface(x=ex, y=ey, z=ez, surfacecolor=shade, colorscale=EARTH_SCALE,
                    showscale=False, hoverinfo="skip", cmin=0.0, cmax=1.0,
                    lighting=dict(ambient=0.62, diffuse=0.85, specular=0.12, roughness=0.92),
                    lightposition=dict(x=1.6e4, y=1.1e4, z=8e3))

    if show_stars:
        st_pts = starfield()
        fig.add_scatter3d(x=st_pts[:, 0], y=st_pts[:, 1], z=st_pts[:, 2], mode="markers",
                          marker=dict(size=1.1, color="#8FA5C8", opacity=.55),
                          hoverinfo="skip", showlegend=False)

    if cloud:
        labels = {"payload": "Payload", "rocket_body": "Rocket body", "debris": "Debris"}
        for cls, pts in cloud.items():
            col, size, op = CLASS_STYLE[cls]
            fig.add_scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker=dict(size=size, color=col, opacity=op,
                            line=dict(width=0)),
                name=f"{labels[cls]} ({len(pts):,})", hoverinfo="skip")

    for pts, color, name, width in tracks:
        fig.add_scatter3d(x=pts[0], y=pts[1], z=pts[2], mode="lines",
                          line=dict(color=color, width=width), name=name, hoverinfo="skip")
    for p, color, name in markers:
        fig.add_scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                          marker=dict(size=7, color=color, symbol="diamond",
                                      line=dict(color="#FFFFFF", width=1)),
                          name=name, hoverinfo="skip")

    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=VOID, showlegend=True,
        legend=dict(bgcolor="rgba(6,9,17,.55)", bordercolor=LINE, borderwidth=1,
                    font=dict(color=INK, size=11, family="monospace"),
                    orientation="h", y=1.04, x=0, itemsizing="constant"),
        scene=dict(bgcolor=VOID, aspectmode="cube",
                   xaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                   yaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                   zaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                   camera=dict(eye=dict(x=1.05, y=1.05, z=.62),
                               up=dict(x=0, y=0, z=1))))
    return fig


def encounter_figure(r0, v0, r_burn, v_post, tr, tv, tca_offset_s, burn_offset_s,
                     miss_before, miss_after, rel_speed_kms, height=460):
    """The encounter in the satellite's own frame -- the standard conjunction view.

    Plotted is the debris' position relative to the satellite, so the origin IS
    the satellite and the closest approach is where each curve comes nearest the
    origin. Two curves: the original trajectory and the post-burn one. At
    11 km/s of closing speed the whole encounter is over in about a second, so
    the window is sized from the relative velocity rather than fixed.
    """
    rel_speed = max(float(rel_speed_kms), 0.1)
    window_s = float(np.clip(8.0 * max(miss_after, 0.5) / rel_speed, 0.6, 30.0))
    half = window_s / 2.0
    dt = window_s / 400.0

    def arc(rr, vv, lead):
        _, a, b = propagate(rr, vv, max(lead, 0.0), dt_s=max(lead / 200.0, 0.05)) \
            if lead > 0 else (None, rr[:, None], vv[:, None])
        _, arr, _ = propagate(a[:, -1], b[:, -1], window_s, dt_s=dt)
        return arr

    hero_o = arc(r0, v0, tca_offset_s - half)
    threat = arc(tr, tv, tca_offset_s - half)
    hero_p = arc(r_burn, v_post, tca_offset_s - burn_offset_s - half)
    n = min(hero_o.shape[1], threat.shape[1], hero_p.shape[1])

    rel_o = threat[:, :n] - hero_o[:, :n]      # debris seen from the satellite
    rel_p = threat[:, :n] - hero_p[:, :n]

    fig = go.Figure()
    fig.add_scatter3d(x=rel_o[0], y=rel_o[1], z=rel_o[2], mode="lines",
                      line=dict(color=ALERT, width=6),
                      name=f"Debris — original path ({miss_before:.3f} km)", hoverinfo="skip")
    fig.add_scatter3d(x=rel_p[0], y=rel_p[1], z=rel_p[2], mode="lines",
                      line=dict(color=NOMINAL, width=6),
                      name=f"Debris — after the burn ({miss_after:.2f} km)", hoverinfo="skip")

    for arr, col in ((rel_o, ALERT), (rel_p, NOMINAL)):
        k = int(np.argmin(np.linalg.norm(arr, axis=0)))
        fig.add_scatter3d(x=[0, arr[0, k]], y=[0, arr[1, k]], z=[0, arr[2, k]],
                          mode="lines", line=dict(color=col, width=3, dash="dash"),
                          showlegend=False, hoverinfo="skip")
        fig.add_scatter3d(x=[arr[0, k]], y=[arr[1, k]], z=[arr[2, k]], mode="markers",
                          marker=dict(size=6, color=col), showlegend=False, hoverinfo="skip")

    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers",
                      marker=dict(size=11, color=ICE, symbol="diamond",
                                  line=dict(color="#FFFFFF", width=2)),
                      name="Satellite", hoverinfo="skip")

    span = max(miss_after * 1.9, miss_before * 3.0, 1.0)
    ax = dict(showbackground=False, gridcolor="#1B2740", zerolinecolor="#2C3E5C",
              color=MUTED, ticks="", range=[-span, span], nticks=5)
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=VOID,
        legend=dict(bgcolor="rgba(6,9,17,.6)", bordercolor=LINE, borderwidth=1,
                    font=dict(color=INK, size=11, family="monospace"),
                    orientation="h", y=1.07, x=0),
        scene=dict(bgcolor=VOID, aspectmode="cube",
                   xaxis=dict(**ax, title=dict(text="km", font=dict(size=10, color=MUTED))),
                   yaxis=dict(**ax, title=dict(text="km", font=dict(size=10, color=MUTED))),
                   zaxis=dict(**ax, title=dict(text="km", font=dict(size=10, color=MUTED))),
                   camera=dict(eye=dict(x=1.35, y=1.35, z=.95))))
    return fig


COLORS = {"status": MUTED, "alert": ALERT, "agent1": ICE, "agent2": AMBER,
          "tool": MUTED, "verdict": NOMINAL, "error": ALERT, "clear": NOMINAL}
TAGS = {"status": "SYS ", "alert": "ALRT", "agent1": "FDO ", "agent2": "MAD ",
        "tool": "PHYS", "verdict": "EXEC", "error": "ERR ", "clear": "OK  "}


def render_log(events, kinds, placeholder, empty_text=""):
    rows = [e for e in events if e.kind in kinds]
    body = "".join(
        f"<span style='color:{MUTED}'>{time.strftime('%H:%M:%S', time.localtime(e.ts))}</span> "
        f"<b style='color:{COLORS.get(e.kind, INK)}'>{TAGS.get(e.kind, e.kind)}</b> "
        f"<span style='color:{COLORS.get(e.kind, INK)}'>{e.text}</span><br>"
        for e in rows) or f"<span style='color:{MUTED}'>{empty_text}</span>"
    placeholder.markdown(f"<div class='kessler-log'>{body}</div>", unsafe_allow_html=True)


def build_state(hero, r0, v0, tname, tr, tv, t0, el, constraints):
    return MissionState(hero_name=hero.name, hero_r0=r0, hero_v0=v0,
                        threat_name=tname, threat_r0=tr, threat_v0=tv, t0=t0,
                        nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2.0,
                        constraints=constraints)


def alert_dict(hero, tname, enc, tr=None, tv=None):
    return {"_tr": tr, "_tv": tv,
            "primary": hero.name, "secondary": tname,
            "tca_offset_s": enc["tca_offset_s"], "miss_km": round(enc["miss_km"], 4),
            "rel_speed_kms": round(enc["rel_speed_kms"], 3), "pc": enc["pc"],
            "radial_km": round(enc.get("radial_km", 0), 3),
            "in_track_km": round(enc.get("in_track_km", 0), 3),
            "cross_track_km": round(enc.get("cross_track_km", 0), 3)}


def show_result(out, alert, plot_slot, metric_slot, r0, v0, horizon, cloud,
                hero_pre, threat_tr, tca_point, hero_name, tname, elapsed, gast=0.0):
    if not out["approved"]:
        st.error("No safe maneuver inside the constraint set — escalated to a human operator.")
        return
    p, res = out["proposal"], out["result"]
    _, rb, vb = propagate(r0, v0, p["burn_offset_s"], dt_s=2.0)
    d = np.array(p["direction_ric"], float)
    v_post = apply_burn(rb[:, -1], vb[:, -1], d / np.linalg.norm(d) * p["delta_v_mps"])
    _, hero_post, _ = propagate(rb[:, -1], v_post, horizon - p["burn_offset_s"], dt_s=15.0)
    plot_slot.plotly_chart(globe_figure(
        cloud,
        [(hero_pre, LINE, "original path", 3), (threat_tr, ALERT, tname, 4),
         (hero_post, NOMINAL, f"{hero_name} (post-burn)", 7)],
        [(tca_point, ALERT, "TCA"), (rb[:, -1], AMBER, "burn")],
        gast_hours=float(gast)), use_container_width=True)
    with metric_slot:
        m = st.columns(5)
        m[0].metric("Miss distance", f"{res['new_miss_km']:.2f} km",
                    f"+{res['new_miss_km'] - alert['miss_km']:.2f}")
        m[1].metric("Probability of collision", f"{res['new_pc']:.1e}", "cleared", delta_color="off")
        m[2].metric("Delta-v spent", f"{p['delta_v_mps']:.3f} m/s")
        m[3].metric("Agent rounds", out["rounds"])
        m[4].metric("Loop time", f"{elapsed:.1f} s")
    st.success(f"BURN APPROVED AND ISSUED — {hero_name} clears {tname} by "
               f"{res['new_miss_km']:.2f} km.")
    st.markdown("<div class='kessler-hd'>Encounter close-up — the debris seen from "
                "the satellite. Origin is the spacecraft.</div>", unsafe_allow_html=True)
    st.plotly_chart(encounter_figure(
        r0, v0, rb[:, -1], v_post, alert["_tr"], alert["_tv"],
        alert["tca_offset_s"], p["burn_offset_s"],
        alert["miss_km"], res["new_miss_km"], alert["rel_speed_kms"]),
        use_container_width=True)


def render_leolabs(view: str, height: int = 560):
    """Embed LeoLabs' live radar-tracked view alongside our own screen.

    This is an independent source: LeoLabs tracks with a phased-array radar
    network, while everything else in this app is propagated from public TLEs.
    Showing both side by side is the point -- one is what we compute, the other
    is what a commercial tracking provider observes.
    """
    st.divider()
    st.markdown("<div class='kessler-hd'>LeoLabs — independent radar-tracked view</div>",
                unsafe_allow_html=True)
    st.link_button(f"Open LeoLabs live view in a new tab  ↗", LEOLABS[view],
                   use_container_width=False)
    components.iframe(LEOLABS[view], height=height, scrolling=False)
    st.caption(
        f"Visualization © LeoLabs, Inc. Embedded under LeoLabs' "
        f"[terms for sharing]({LEOLABS_TERMS}) for non-commercial educational use — "
        f"[leolabs.space](https://leolabs.space). LeoLabs is the production-grade "
        f"ingest path for this system; the screening engine here runs on the public "
        f"CelesTrak TLE catalog.")
    st.caption(
        ":grey[If the embedded 3D view shows a render error, use the link above — "
        "some browsers refuse WebGL inside a cross-origin frame. The link is the "
        "safer path mid-demo.]")


def render_engagement_log():
    if st.session_state.get("show_leo"):
        render_leolabs(st.session_state.get("leo_view", list(LEOLABS)[0]))
    elog = st.session_state.get("log")
    if not elog or not len(elog):
        return
    st.divider()
    st.markdown("<div class='kessler-hd'>Engagement log</div>", unsafe_allow_html=True)
    st.dataframe(elog.summary_rows(), use_container_width=True, hide_index=True)
    st.download_button("Download engagement log (JSON)", elog.to_json(),
                       file_name="kessler-engagements.json", mime="application/json")


# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### PROJECT KESSLER")
st.sidebar.caption("Autonomous orbital traffic control")
mode = st.sidebar.radio("Mode", ["Seeded threat", "Live scan", "Fleet monitor"],
                        help="Seeded guarantees an encounter; Live scan and Fleet monitor "
                             "screen the real catalog and may legitimately find nothing.")
limit = st.sidebar.select_slider("Catalog size", [500, 2000, 5000, 10000, None], value=None,
                                 format_func=lambda v: "all (16k)" if v is None else f"{v:,}")
cat = get_catalog(limit)
st.sidebar.caption(f"{len(cat):,} objects · {cat.source}")
op_mode = st.sidebar.radio(
    "Authority", [Mode.SUPERVISED.value, Mode.AUTONOMOUS.value], index=0,
    help="SUPERVISED holds the approved burn at a human authorization gate. "
         "AUTONOMOUS self-authorizes and records that it did. Neither uplinks "
         "anything — this build stops at a log.")
st.sidebar.divider()
show_leo = st.sidebar.checkbox("Show LeoLabs live view", value=False,
                               help="Embeds LeoLabs' public radar-tracked visualization "
                                    "beside our own screen. Adds a few seconds of load.")
leo_view = st.sidebar.selectbox("LeoLabs view", list(LEOLABS), index=0,
                                disabled=not show_leo)
st.sidebar.divider()

c = Constraints()
ts = timescale()
t0 = ts.now()
st.session_state["show_leo"] = show_leo
st.session_state["leo_view"] = leo_view
if "log" not in st.session_state:
    st.session_state["log"] = EngagementLog()
elog: EngagementLog = st.session_state["log"]

st.markdown("## Orbital traffic control")


# ================================================================ FLEET MONITOR
if mode == "Fleet monitor":
    pattern = st.sidebar.text_input("Asset name filter", "STARLINK")
    fleet_n = st.sidebar.slider("Assets monitored", 2, 40, 10, 1)
    horizon_h = st.sidebar.slider("Look-ahead (hours)", 1.0, 12.0, 6.0, 0.5)
    watch_km = st.sidebar.slider("Watch threshold (km)", 1.0, 100.0, 25.0, 1.0,
                                 help="Log anything closer than this")
    st.sidebar.divider()
    c.dv_budget_mps = st.sidebar.slider("Delta-v budget (m/s)", 0.05, 1.0, 0.35, 0.01)
    c.min_miss_km = st.sidebar.slider("Separation minimum (km)", 1.0, 20.0, 5.0, 0.5,
                                      help="Below this, the agents are dispatched")
    auto = st.sidebar.checkbox("Auto-rescan", value=False)
    every = st.sidebar.slider("Rescan interval (s)", 15, 300, 60, 5, disabled=not auto)
    inject = st.sidebar.checkbox("Inject a seeded threat", value=False,
                                 help="Adds one synthetic encounter so the monitor has "
                                      "something to catch")

    fleet = select_fleet(cat.objects, pattern, fleet_n)
    if not fleet:
        st.error(f"No objects matching {pattern!r}.")
        st.stop()

    st.session_state.setdefault("resolved", set())

    @st.fragment(run_every=f"{every}s" if auto else None)
    def fleet_panel():
        now = ts.now()
        res = sweep_fleet(fleet, cat.objects, now, horizon_s=horizon_h * 3600,
                          coarse_step_s=60.0, threshold_km=watch_km, constraints=c)

        injected = None
        if inject:
            tgt = fleet[0]
            r0, v0 = teme_state(tgt, now)
            tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
            enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
            injected = (tgt, tname, tr, tv, enc)
            for s in res.statuses:
                if s.name == tgt.name:
                    s.action_required, s.reason = requires_action(
                        {"miss_km": enc["miss_km"], "pc": enc["pc"]}, c)
                    s.reason = f"seeded {tname}: {s.reason}"

        k = st.columns(5)
        k[0].metric("Assets monitored", len(res.statuses))
        k[1].metric("Action required", len(res.actionable))
        k[2].metric("Watch list", len(res.watching))
        k[3].metric("States propagated", f"{res.states/1e6:.1f} M")
        k[4].metric("Sweep time", f"{res.elapsed_s:.2f} s")
        st.caption(f"One propagation of {res.catalog_size:,} objects × {res.epochs} epochs "
                   f"= {res.matrix_mb:.0f} MB resident, reused across every asset. "
                   f"Last scan {time.strftime('%H:%M:%S')}.")

        rows = []
        for s in res.statuses:
            rows.append({
                "Status": s.severity, "Asset": s.name, "Alt (km)": round(s.alt_km, 1),
                "Closest object": s.worst.secondary if s.worst else "—",
                "Miss (km)": round(s.worst.miss_km, 3) if s.worst else None,
                "Pc": f"{s.worst.pc:.1e}" if s.worst else "—",
                "TCA (min)": round(s.worst.tca_offset_s / 60, 1) if s.worst else None,
                "Assessment": s.reason,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        breaches = res.actionable
        if not breaches:
            st.info(f"No asset breaches the {c.min_miss_km:.1f} km separation minimum. "
                    f"{len(res.watching)} logged for watching — detection is not the same "
                    f"as needing to maneuver.")
            return

        for s in breaches:
            if s.name in st.session_state["resolved"]:
                continue
            st.markdown(f"#### Dispatching agents — {s.name}")
            log = st.empty()
            events: list[Event] = []
            bus = Bus(sink=lambda e: (events.append(e),
                                      render_log(events, set(TAGS), log))[0])
            tgt = next(a for a in fleet if a.name == s.name)
            r0, v0 = teme_state(tgt, now)
            el = elements(r0, v0)
            if injected and s.name == injected[0].name:
                _, tname, tr, tv, enc = injected
            else:
                obj = cat.objects[s.worst.secondary_index]
                tname = obj.name
                tr, tv = teme_state(obj, now)
                enc = find_tca(r0, v0, tr, tv, horizon_s=s.worst.tca_offset_s * 1.4)
            state = build_state(tgt, r0, v0, tname, tr, tv, now, el, c)
            out = run_resolution(state, altitude_shortlist(cat.objects, el, 50.0),
                                 alert_dict(tgt, tname, enc), bus)
            if out["approved"]:
                st.success(f"{s.name}: burn issued, {out['result']['new_miss_km']:.2f} km clearance")
            else:
                st.error(f"{s.name}: no safe maneuver — escalated")
            st.session_state["resolved"].add(s.name)

    fleet_panel()
    render_engagement_log()
    st.stop()


# ================================================================ SINGLE ASSET
names = [s.name for s in cat.objects]
default = next((i for i, n in enumerate(names) if "STARLINK-1008" in n), 0)
hero_name = st.sidebar.selectbox("Protected asset", names, index=default)

if mode == "Seeded threat":
    tca_min = st.sidebar.slider("Time to closest approach (min)", 20.0, 180.0, 92.0, 1.0)
    miss_km = st.sidebar.slider("Seeded miss distance (km)", 0.05, 10.0, 0.412, 0.001)
else:
    horizon_h = st.sidebar.slider("Look-ahead (hours)", 1.0, 24.0, 6.0, 0.5)
    watch_km = st.sidebar.slider("Screen threshold (km)", 1.0, 100.0, 25.0, 1.0)

st.sidebar.divider()
c.dv_budget_mps = st.sidebar.slider("Delta-v budget (m/s)", 0.05, 1.0, 0.35, 0.01)
c.min_miss_km = st.sidebar.slider("Separation minimum (km)", 1.0, 20.0, 2.0, 0.1)
run = st.sidebar.button("RUN AVOIDANCE", type="primary", use_container_width=True)

plot_slot = st.empty()
metric_slot = st.container()
left, right = st.columns(2)
left.markdown("<div class='kessler-hd'>Alert log</div>", unsafe_allow_html=True)
alert_slot = left.empty()
right.markdown("<div class='kessler-hd'>Agent console</div>", unsafe_allow_html=True)
agent_slot = right.empty()

ALERT_KINDS = ("status", "alert", "error", "clear")
AGENT_KINDS = ("agent1", "agent2", "tool", "verdict")

# A button click reruns the whole script. Recomputing the sweep and re-running
# the agents on every click would burn seconds and open a fresh engagement each
# time, so a completed run is cached and replayed until RUN is pressed again.
if run:
    st.session_state.pop("run_result", None)
rr = st.session_state.get("run_result")

hero = cat.by_name(hero_name)
r0, v0 = teme_state(hero, t0)
el = elements(r0, v0)

if not run and not rr:
    grid = ts.tt_jd(t0.tt + np.zeros(1))
    _, hp, _ = propagate(r0, v0, el["period_min"] * 60, dt_s=20.0)
    plot_slot.plotly_chart(globe_figure(leo_cloud(cat.objects, grid),
                                        [(hp, ICE, hero.name, 4)],
                                        gast_hours=float(t0.gast)),
                           use_container_width=True)
    render_log([], (), alert_slot, "Idle — press RUN AVOIDANCE.")
    render_log([], (), agent_slot, "Agents standing by.")
    render_engagement_log()
    st.stop()

if run:
    events: list[Event] = []

    def sink(ev: Event):
        events.append(ev)
        render_log(events, ALERT_KINDS, alert_slot)
        render_log(events, AGENT_KINDS, agent_slot)

    bus = Bus(sink=sink)
    t_run = time.time()
    bus.emit("status", f"{len(cat):,} objects loaded from {cat.source}")
    bus.emit("status", f"Protecting {hero.name} — {el['alt_km']:.1f} km, inc {el['inc_deg']:.2f}°")

    if mode == "Seeded threat":
        tca_s = tca_min * 60.0
        tname, tr, tv = synthesize_threat(r0, v0, tca_s, miss_km=miss_km)
        bus.emit("status", f"Seeded threat {tname} (deterministic demo encounter)")
        enc = find_tca(r0, v0, tr, tv, horizon_s=tca_s * 1.3)
    else:
        bus.emit("status", f"Screening {len(cat):,} objects over {horizon_h:.1f} h "
                           f"at {watch_km:.0f} km — no seeding")
        t_s = time.time()
        found = screen(hero, cat.objects, t0, horizon_s=horizon_h * 3600,
                       coarse_step_s=60.0, threshold_km=watch_km)
        bus.emit("status", f"Sweep complete in {time.time()-t_s:.2f}s — "
                           f"{len(found)} approach(es) under {watch_km:.0f} km")
        if not found:
            render_log(events, ALERT_KINDS, alert_slot)
            st.info(f"**{hero.name} is clear.** No object comes within {watch_km:.0f} km in "
                    f"the next {horizon_h:.1f} hours. That is the honest answer — genuine "
                    f"conjunctions for one satellite are rare in any given window. Widen the "
                    f"threshold, look further ahead, or use Seeded threat.")
            render_engagement_log()
            st.stop()
        worst = found[0]
        obj = cat.objects[worst.secondary_index]
        tname = obj.name
        tr, tv = teme_state(obj, t0)
        enc = find_tca(r0, v0, tr, tv, horizon_s=worst.tca_offset_s * 1.4)

    bus.emit("alert", f"CONJUNCTION — {hero.name} vs {tname}")
    bus.emit("alert", f"TCA T+{enc['tca_offset_s']/60:.1f} min · miss {enc['miss_km']:.3f} km · "
                      f"rel {enc['rel_speed_kms']:.2f} km/s · Pc {enc['pc']:.2e}")

    cc = cross_check(hero, t0, enc["tca_offset_s"], assumed_sigma_km=0.35)
    bus.emit("status", f"Consensus check — SGP4 vs RK4+J2 agree to {cc.residual_km*1000:.1f} m "
                       f"at TCA ({'consistent' if cc.consistent else 'DIVERGENT'})")
    if not cc.consistent:
        bus.emit("error", "Propagators disagree beyond tolerance — geometry not trustworthy.")

    geom = {k: v for k, v in alert_dict(hero, tname, enc).items() if not k.startswith("_")}
    eng = elog.open(primary=hero.name, secondary=tname, seeded=(mode == "Seeded threat"),
                    geometry=geom, consensus=cc.as_dict(), mode=op_mode)

    horizon = enc["tca_offset_s"] * 1.15
    _, hero_pre, _ = propagate(r0, v0, horizon, dt_s=15.0)
    _, threat_tr, _ = propagate(tr, tv, horizon, dt_s=15.0)
    cloud = leo_cloud(cat.objects, ts.tt_jd(t0.tt + np.zeros(1)))
    _, r_at, _ = propagate(r0, v0, enc["tca_offset_s"], dt_s=5.0)
    tca_point = r_at[:, -1]

    act, why = requires_action(enc, c)
    eng.action_required, eng.assessment = act, why
    out = None
    if act:
        bus.emit("alert", f"ACTION REQUIRED — {why}")
        state = build_state(hero, r0, v0, tname, tr, tv, t0, el, c)
        out = run_resolution(state, altitude_shortlist(cat.objects, el, 50.0),
                             {k: v for k, v in alert_dict(hero, tname, enc).items()
                              if not k.startswith("_")}, bus)
        eng.proposal = out.get("proposal")
        eng.engine_verdict = out.get("result")
        eng.agent_rounds = out.get("rounds", 0)
        if out["approved"] and op_mode == Mode.AUTONOMOUS.value:
            eng.authorize("agent (autonomous mode)")
            bus.emit("verdict", "Gate self-authorized — AUTONOMOUS mode, decision recorded.")
        elif not out["approved"]:
            eng.halt("engine", "no safe maneuver inside the constraint set")
    else:
        eng.authorization = Authorization.NOT_REQUIRED.value
        bus.emit("clear", f"NO ACTION REQUIRED — {why}")
    eng.record_reasoning(events)

    st.session_state["run_result"] = dict(
        eng_id=eng.engagement_id, events=events, enc=enc, out=out, cc=cc,
        r0=r0, v0=v0, horizon=horizon, cloud=cloud, hero_pre=hero_pre,
        threat_tr=threat_tr, tca_point=tca_point, hero_name=hero.name, tname=tname,
        alert=alert_dict(hero, tname, enc, tr=tr, tv=tv), elapsed=time.time() - t_run,
        action=act, why=why)
    rr = st.session_state["run_result"]

# ---- replay the cached run ----
events = rr["events"]
enc, out, cc = rr["enc"], rr["out"], rr["cc"]
render_log(events, ALERT_KINDS, alert_slot)
render_log(events, AGENT_KINDS, agent_slot,
           "Agents not dispatched — encounter is inside limits.")
eng = next(e for e in elog.entries if e.engagement_id == rr["eng_id"])

plot_slot.plotly_chart(globe_figure(
    rr["cloud"],
    [(rr["hero_pre"], ICE, f"{rr['hero_name']} (current)", 5),
     (rr["threat_tr"], ALERT, rr["tname"], 4)],
    [(rr["tca_point"], ALERT, "TCA")],
    gast_hours=float(t0.gast)), use_container_width=True)

# ---- the gate: detection is not the same as needing to maneuver ----
if not rr["action"]:
    with metric_slot:
        m = st.columns(4)
        m[0].metric("Miss distance", f"{enc['miss_km']:.2f} km")
        m[1].metric("Probability of collision", f"{enc['pc']:.1e}")
        m[2].metric("Separation minimum", f"{c.min_miss_km:.1f} km")
        m[3].metric("Delta-v spent", "0.000 m/s")
    st.info(f"**Conjunction logged, no maneuver.** {rr['why']}. Burning propellant here "
            f"would buy nothing — the encounter already clears both limits. Tighten the "
            f"separation minimum or reduce the seeded miss distance to force a response.")
    render_engagement_log()
    st.stop()

if out is None or not out["approved"]:
    st.error("No safe maneuver inside the constraint set — escalated to a human operator.")
    render_engagement_log()
    st.stop()

# ---- authorization gate ----
if eng.authorization == Authorization.PENDING.value:
    p, res = out["proposal"], out["result"]
    st.warning(f"**AUTHORIZATION REQUIRED — {eng.engagement_id}**  ·  engine cleared the "
               f"maneuver in {out['rounds']} round(s), holding at the gate.")
    g = st.columns([1, 1, 1, 1, 2])
    g[0].metric("Burn", f"{p['delta_v_mps']:.3f} m/s")
    g[1].metric("Miss after", f"{res['new_miss_km']:.2f} km")
    g[2].metric("Pc after", f"{res['new_pc']:.1e}")
    g[3].metric("Consensus", f"{cc.residual_km*1000:.0f} m")
    who = g[4].text_input("Authorizing operator", "flight director", key="who")
    b = st.columns(2)
    ok = b[0].button("AUTHORIZE BURN", type="primary", use_container_width=True)
    no = b[1].button("HALT", use_container_width=True)
    if ok:
        eng.authorize(who or "operator")
    elif no:
        eng.halt(who or "operator", "halted at operator review")
        st.error(f"**HALTED** by {eng.authorized_by} — no command issued. "
                 f"The satellite flies the original path.")
        render_engagement_log()
        st.stop()
    else:
        st.caption("Nothing is transmitted either way — the uplink is simulated and the "
                   "decision is written to the engagement log.")
        render_engagement_log()
        st.stop()

if eng.authorization == Authorization.HALTED.value:
    st.error(f"**HALTED** by {eng.authorized_by} — {eng.halt_reason}. No command issued.")
    render_engagement_log()
    st.stop()

show_result(out, rr["alert"], plot_slot, metric_slot, rr["r0"], rr["v0"], rr["horizon"],
            rr["cloud"], rr["hero_pre"], rr["threat_tr"], rr["tca_point"],
            rr["hero_name"], rr["tname"], rr["elapsed"], gast=float(t0.gast))
st.caption(f"Engagement {eng.engagement_id} · authorization {eng.authorization} "
           f"by {eng.authorized_by} · consensus {cc.residual_km*1000:.0f} m · "
           f"uplink {eng.uplink}")
render_engagement_log()
