"""Project Kessler — autonomous orbital traffic control.

One screen, one button. Scans the catalogue, finds what is going to hit, and
corrects course.
"""
from __future__ import annotations

import os
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from kessler.agents import recommend_option, review_trade_space
from kessler.assurance import EngagementLog, Mode, cross_check
from kessler.bus import Bus, Event
from kessler.catalog import classify, load_demo_catalog
from kessler.environment import TOTAL_OVER_1MM, sample_environment
from kessler.mission import (Constraints, MissionState, altitude_shortlist, find_tca,
                             requires_action, synthesize_threat)
from kessler.monitor import select_fleet, sweep_fleet
from kessler.options import Option, _characterize as _recharacterize, solve_options
from kessler.physics import (R_EARTH, apply_burn, elements, propagate,
                             teme_positions_many, teme_state, timescale)

VOID, PANEL, LINE = "#05080F", "#0F1728", "#22304C"
INK, MUTED = "#E9EEF7", "#8494AD"
AMBER, ICE, ALERT, NOMINAL = "#F2A03D", "#6FB6E8", "#FF4757", "#4FD1A5"
LEO_VIEW_KM = 8200.0
# Bumped whenever the shape of st.session_state["done"] changes. Streamlit keeps
# session state across hot-reloads, so without this a dict written by an older
# build survives into new code and blows up on a key that did not exist yet.
STATE_SCHEMA = 4
EARTH_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "earth_720x360.npy")

CLASS_STYLE = {"payload": ("#4FE0A0", 1.7, .85),
               "rocket_body": ("#F2C14E", 2.0, .9),
               "debris": ("#FF5C6E", 2.3, .95)}
EARTH_SCALE = [[0, "#040A18"], [.18, "#0A1E3C"], [.34, "#0E3355"], [.46, "#14503F"],
               [.58, "#2E6B3A"], [.70, "#6E7A3C"], [.82, "#A08B54"],
               [.92, "#CFC2A0"], [1, "#FFFFFF"]]

st.set_page_config(page_title="Project Kessler", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""<style>
#MainMenu, footer, header {{ visibility:hidden; }}
.stApp {{ background:
   radial-gradient(1200px 700px at 50% -10%, #0C1526 0%, {VOID} 60%); }}
.block-container {{ padding-top:2.2rem; padding-bottom:3rem; max-width:1500px; }}
.kes-title {{ font-family:'Arial Narrow',Arial,sans-serif; font-weight:700;
  font-size:clamp(2.4rem,4.6vw,3.9rem); line-height:.94; letter-spacing:-.02em;
  color:{INK}; text-transform:uppercase; margin:0; }}
.kes-sub {{ font-family:'SF Mono',Menlo,monospace; font-size:.72rem; letter-spacing:.32em;
  text-transform:uppercase; color:{AMBER}; margin:.5rem 0 0; }}
.kes-lab {{ font-family:'SF Mono',Menlo,monospace; font-size:.62rem; letter-spacing:.2em;
  text-transform:uppercase; color:{MUTED}; padding:.2rem 0 .35rem; }}
.kes-log {{ background:{PANEL}; border:1px solid {LINE}; border-radius:3px;
  padding:.75rem .95rem; height:250px; overflow-y:auto; font-family:'SF Mono',Menlo,monospace;
  font-size:.73rem; line-height:1.7; white-space:pre-wrap; }}
div.stButton > button {{
  width:100%; height:74px; border-radius:4px; border:1px solid {AMBER};
  background:linear-gradient(180deg, #F2A03D 0%, #DE8A26 100%); color:#120A02;
  font-family:'Arial Narrow',Arial,sans-serif; font-weight:700; font-size:1.6rem;
  letter-spacing:.16em; text-transform:uppercase; transition:.15s; }}
div.stButton > button:hover {{ filter:brightness(1.12); border-color:#FFD9A0; }}
[data-testid="stMetricValue"] {{ font-family:'Arial Narrow',Arial,sans-serif;
  font-size:2.5rem; letter-spacing:-.01em; }}
[data-testid="stMetricLabel"] {{ font-family:'SF Mono',Menlo,monospace;
  font-size:.6rem !important; letter-spacing:.16em; text-transform:uppercase; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_resource(show_spinner=False)
def get_catalog():
    return load_demo_catalog(limit=None)


@st.cache_data(show_spinner=False)
def _tex():
    return np.load(EARTH_TEX).astype(np.float32) if os.path.exists(EARTH_TEX) else None


@st.cache_data(show_spinner=False)
def environment(n):
    return sample_environment(n)


@st.cache_data(show_spinner=False)
def starfield(n=620, radius=7900.0, seed=4):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * radius


def earth_surface(gast_hours, n_lon=200, n_lat=100):
    """Blue Marble luminance mapped onto a sphere, rotated into the inertial frame."""
    tex = _tex()
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    LON, LAT = np.meshgrid(lon, lat)
    x = R_EARTH * np.cos(LAT) * np.cos(LON)
    y = R_EARTH * np.cos(LAT) * np.sin(LON)
    z = R_EARTH * np.sin(LAT)
    if tex is None:
        return x, y, z, .35 + .3 * np.cos(LAT)
    g = (LON - np.radians(gast_hours * 15.0) + np.pi) % (2 * np.pi) - np.pi
    ti = ((g + np.pi) / (2 * np.pi) * (tex.shape[1] - 1)).astype(int)
    tj = ((np.pi / 2 - LAT) / np.pi * (tex.shape[0] - 1)).astype(int)
    return x, y, z, tex[tj, ti]


def sphere(radius, n=36):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    return (radius * np.outer(np.cos(u), np.sin(v)),
            radius * np.outer(np.sin(u), np.sin(v)),
            radius * np.outer(np.ones_like(u), np.cos(v)))


@st.cache_data(show_spinner=False)
def cloud_snapshot(_objects, jd, max_points=26000):
    ts = timescale()
    pts = teme_positions_many(_objects, ts.tt_jd(np.array([jd])))[:, 0, :]
    ok = ~np.isnan(pts[:, 0])
    pts, objs = pts[ok], [o for o, k in zip(_objects, ok) if k]
    keep = np.linalg.norm(pts, axis=1) < LEO_VIEW_KM
    pts, objs = pts[keep], [o for o, k in zip(objs, keep) if k]
    if len(pts) > max_points:          # only ever trims on very slow machines
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, objs = pts[idx], [objs[i] for i in idx]
    g = {k: [] for k in CLASS_STYLE}
    for p, o in zip(pts, objs):
        g[classify(o.name)].append(p)
    return {k: np.array(v) for k, v in g.items() if len(v)}


# ----------------------------------------------------------------- figures
def globe(cloud=None, tracks=(), markers=(), gast=0.0, height=660, env_n=45000):
    fig = go.Figure()
    for sc, op, col in ((1.06, .05, "#7FD4FF"), (1.022, .085, "#4FA8E8")):
        sx, sy, sz = sphere(R_EARTH * sc, 32)
        fig.add_surface(x=sx, y=sy, z=sz, showscale=False, opacity=op, hoverinfo="skip",
                        colorscale=[[0, col], [1, col]], lighting=dict(ambient=1, diffuse=0))
    ex, ey, ez, sh = earth_surface(gast)
    fig.add_surface(x=ex, y=ey, z=ez, surfacecolor=sh, colorscale=EARTH_SCALE,
                    showscale=False, hoverinfo="skip", cmin=0, cmax=1,
                    lighting=dict(ambient=.6, diffuse=.9, specular=.1, roughness=.92),
                    lightposition=dict(x=1.6e4, y=1.1e4, z=8e3))
    s = starfield()
    fig.add_scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2], mode="markers", hoverinfo="skip",
                      marker=dict(size=1.1, color="#93A9CC", opacity=.5), showlegend=False)
    if env_n:
        e = environment(env_n)
        fig.add_scatter3d(
            x=e[:, 0], y=e[:, 1], z=e[:, 2], mode="markers", hoverinfo="skip",
            marker=dict(size=1.0, color="#B4451F", opacity=.30),
            name=f"Modeled fragments &lt;1cm — untracked ({env_n//1000}k of "
                 f"{TOTAL_OVER_1MM/1e6:.0f}M)")
    if cloud:
        lab = {"payload": "Payload", "rocket_body": "Rocket body",
               "debris": "Tracked debris"}
        for cls, pts in cloud.items():
            col, size, op = CLASS_STYLE[cls]
            fig.add_scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                              marker=dict(size=size, color=col, opacity=op),
                              name=f"{lab[cls]} ({len(pts):,})", hoverinfo="skip")
    for pts, col, name, w in tracks:
        fig.add_scatter3d(x=pts[0], y=pts[1], z=pts[2], mode="lines",
                          line=dict(color=col, width=w), name=name, hoverinfo="skip")
    for p, col, name in markers:
        fig.add_scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=name,
                          hoverinfo="skip",
                          marker=dict(size=8, color=col, symbol="diamond",
                                      line=dict(color="#FFF", width=1)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(bgcolor="rgba(5,8,15,.55)", bordercolor=LINE, borderwidth=1,
                                  font=dict(color=INK, size=11, family="monospace"),
                                  orientation="h", y=1.04, x=0, itemsizing="constant"),
                      scene=dict(bgcolor="rgba(0,0,0,0)", aspectmode="cube",
                                 xaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                                 yaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                                 zaxis=dict(visible=False, range=[-LEO_VIEW_KM, LEO_VIEW_KM]),
                                 camera=dict(eye=dict(x=1.05, y=1.05, z=.6))))
    return fig


def encounter(r0, v0, r_burn, v_post, tr, tv, tca_s, burn_s, before, after, rel_kms,
              height=420):
    """The debris seen from the satellite. Origin is the spacecraft."""
    win = float(np.clip(8.0 * max(after, .5) / max(rel_kms, .1), .6, 30.0))
    half, dt = win / 2, win / 400

    def arc(rr, vv, lead):
        if lead > 0:
            _, a, b = propagate(rr, vv, lead, dt_s=max(lead / 200, .05))
            rr, vv = a[:, -1], b[:, -1]
        _, out, _ = propagate(rr, vv, win, dt_s=dt)
        return out

    ho, th = arc(r0, v0, tca_s - half), arc(tr, tv, tca_s - half)
    hp = arc(r_burn, v_post, tca_s - burn_s - half)
    n = min(ho.shape[1], th.shape[1], hp.shape[1])
    ro, rp = th[:, :n] - ho[:, :n], th[:, :n] - hp[:, :n]

    fig = go.Figure()
    for arr, col, nm in ((ro, ALERT, f"Original path — {before:.3f} km"),
                         (rp, NOMINAL, f"After the burn — {after:.2f} km")):
        fig.add_scatter3d(x=arr[0], y=arr[1], z=arr[2], mode="lines",
                          line=dict(color=col, width=7), name=nm, hoverinfo="skip")
        k = int(np.argmin(np.linalg.norm(arr, axis=0)))
        fig.add_scatter3d(x=[0, arr[0, k]], y=[0, arr[1, k]], z=[0, arr[2, k]], mode="lines",
                          line=dict(color=col, width=3, dash="dash"),
                          showlegend=False, hoverinfo="skip")
    fig.add_scatter3d(x=[0], y=[0], z=[0], mode="markers", name="Satellite", hoverinfo="skip",
                      marker=dict(size=12, color=ICE, symbol="diamond",
                                  line=dict(color="#FFF", width=2)))
    span = max(after * 1.9, before * 3, 1.0)
    ax = dict(showbackground=False, gridcolor="#1B2740", zerolinecolor="#2C3E5C",
              color=MUTED, ticks="", range=[-span, span], nticks=5,
              title=dict(text="km", font=dict(size=10, color=MUTED)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(bgcolor="rgba(5,8,15,.6)", bordercolor=LINE, borderwidth=1,
                                  font=dict(color=INK, size=11, family="monospace"),
                                  orientation="h", y=1.08, x=0),
                      scene=dict(bgcolor="rgba(0,0,0,0)", aspectmode="cube",
                                 xaxis=ax, yaxis=ax, zaxis=ax,
                                 camera=dict(eye=dict(x=1.35, y=1.35, z=.95))))
    return fig


COL = {"status": MUTED, "alert": ALERT, "agent1": ICE, "agent2": AMBER,
       "tool": MUTED, "verdict": NOMINAL, "error": ALERT, "clear": NOMINAL}
TAG = {"status": "SYS ", "alert": "ALRT", "agent1": "FDO ", "agent2": "MAD ",
       "tool": "PHYS", "verdict": "EXEC", "error": "ERR ", "clear": "OK  "}


def log_html(events, kinds, empty=""):
    rows = [e for e in events if e.kind in kinds]
    body = "".join(
        f"<span style='color:{MUTED}'>{time.strftime('%H:%M:%S', time.localtime(e.ts))}</span> "
        f"<b style='color:{COL.get(e.kind, INK)}'>{TAG.get(e.kind, e.kind)}</b> "
        f"<span style='color:{COL.get(e.kind, INK)}'>{e.text}</span><br>"
        for e in rows) or f"<span style='color:{MUTED}'>{empty}</span>"
    return f"<div class='kes-log'>{body}</div>"


# ----------------------------------------------------------------- page
st.markdown("<p class='kes-title'>Project Kessler</p>"
            "<p class='kes-sub'>Autonomous orbital traffic control</p>",
            unsafe_allow_html=True)

with st.spinner("Loading the catalogue…"):
    cat = get_catalog()
ts = timescale()
t0 = ts.now()

with st.expander("Advanced", expanded=False):
    a = st.columns(4)
    fleet_n = a[0].slider("Assets protected", 4, 30, 12)
    env_n = st.select_slider(
        "Modeled debris environment (sample size)",
        [0, 15000, 30000, 45000, 70000, 100000], value=45000,
        format_func=lambda v: "off" if v == 0 else f"{v//1000}k",
        help="A statistical sample of the untracked sub-centimetre population, "
             "drawn from the altitude and inclination shells real breakups "
             "produced. Context only — never screened against.")
    horizon_h = a[1].slider("Look-ahead (h)", 1.0, 12.0, 6.0, .5)
    sep_km = a[2].slider("Separation minimum (km)", 1.0, 10.0, 2.0, .1)
    dv_budget = a[3].slider("Δv budget (m/s)", .05, 1.0, .35, .01)
    supervised = st.checkbox("Require human authorization before any burn", value=False)

c = Constraints(dv_budget_mps=dv_budget, min_miss_km=sep_km)
plot_slot = st.empty()
btn_slot = st.empty()
stat_slot = st.container()
log_slot = st.container()

if "done" not in st.session_state:
    st.session_state["done"] = None
elif (st.session_state["done"] or {}).get("schema") != STATE_SCHEMA:
    st.session_state["done"] = None      # stale shape from a previous build

engage = btn_slot.button("Engage orbital traffic control", type="primary")

if not engage and st.session_state["done"] is None:
    plot_slot.plotly_chart(globe(cloud_snapshot(cat.objects, float(t0.tt)),
                                 gast=float(t0.gast), env_n=env_n),
                           use_container_width=True, key="globe_idle")
    with stat_slot:
        m = st.columns(4)
        m[0].metric("Objects tracked", f"{len(cat):,}")
        m[1].metric("Assets protected", fleet_n)
        m[2].metric("Threats", "—")
        m[3].metric("Human minutes", "0")
    st.caption(f"{len(cat):,} objects carry real TLEs and are screened. "
               f"An estimated {TOTAL_OVER_1MM/1e6:.0f} million fragments larger than a "
               f"millimetre exist and are not tracked by anyone — the dim cloud is a "
               f"modeled sample of that population, shown for context only.")
    st.stop()

# ----------------------------------------------------------------- run
if engage:
    events: list[Event] = []
    live = log_slot.empty()

    def sink(ev):
        events.append(ev)
        live.markdown(log_html(events, set(TAG)), unsafe_allow_html=True)

    bus = Bus(sink=sink)
    t_start = time.time()
    fleet = select_fleet(cat.objects, "STARLINK", fleet_n)
    cloud = cloud_snapshot(cat.objects, float(t0.tt))
    plot_slot.plotly_chart(globe(cloud, gast=float(t0.gast), env_n=env_n),
                           use_container_width=True, key="globe_scan")

    bus.emit("status", f"Catalogue online — {len(cat):,} tracked objects")
    bus.emit("status", f"Protecting {len(fleet)} assets")

    res = sweep_fleet(fleet, cat.objects, t0, horizon_s=horizon_h * 3600,
                      coarse_step_s=60.0, threshold_km=25.0, constraints=c)
    bus.emit("status", f"Swept {res.catalog_size:,} objects × {res.epochs} epochs "
                       f"= {res.states/1e6:.1f}M states in {res.elapsed_s:.2f}s")
    bus.emit("status", f"{len(res.watching)} close approaches logged, "
                       f"{len(res.actionable)} breach the {c.min_miss_km:.1f} km minimum")

    # A guaranteed encounter so the loop always has something to solve on stage.
    hero = fleet[0]
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    tname, tr, tv = synthesize_threat(r0, v0, 92 * 60, miss_km=0.412)
    enc = find_tca(r0, v0, tr, tv, horizon_s=92 * 60 * 1.3)
    cc = cross_check(hero, t0, enc["tca_offset_s"])
    bus.emit("alert", f"COLLISION COURSE — {hero.name} vs {tname}")
    bus.emit("alert", f"Closest approach {enc['miss_km']:.3f} km in "
                      f"{enc['tca_offset_s']/60:.0f} min at {enc['rel_speed_kms']:.1f} km/s "
                      f"· Pc {enc['pc']:.1e}")
    bus.emit("status", f"Cross-check: two independent propagators agree to "
                       f"{cc.residual_km*1000:.0f} m")

    horizon = enc["tca_offset_s"] * 1.15
    _, pre, _ = propagate(r0, v0, horizon, dt_s=15.0)
    _, thr, _ = propagate(tr, tv, horizon, dt_s=15.0)
    _, r_at, _ = propagate(r0, v0, enc["tca_offset_s"], dt_s=5.0)
    plot_slot.plotly_chart(globe(cloud, [(pre, ICE, hero.name, 5), (thr, ALERT, tname, 4)],
                                 [(r_at[:, -1], ALERT, "Impact point")],
                                 gast=float(t0.gast), env_n=env_n),
                           use_container_width=True, key="globe_threat")

    alert = {"primary": hero.name, "secondary": tname, "tca_offset_s": enc["tca_offset_s"],
             "miss_km": round(enc["miss_km"], 4), "pc": enc["pc"],
             "rel_speed_kms": round(enc["rel_speed_kms"], 3)}
    state = MissionState(hero.name, r0, v0, tname, tr, tv, t0,
                         nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2,
                         constraints=c)

    bus.emit("status", "Generating the avoidance trade space — five strategies, "
                       "each flown and re-screened against the catalogue")
    trade = solve_options(state, cat.objects, enc["tca_offset_s"])
    bus.emit("status", f"{trade['scenarios']} scenarios × {trade['passes']} passes = "
                       f"{trade['states_evaluated']:,} states, "
                       f"{trade['tensor_gb']:.2f} GB tensor, {trade['elapsed_s']}s "
                       f"on {trade['backend']}")
    n_ok = sum(1 for o in trade["options"] if o.feasible)
    bus.emit("status", f"{n_ok} of {trade['scenarios']} options satisfy every constraint")
    rec = review_trade_space(state, altitude_shortlist(cat.objects, el, 50.0),
                             trade["options"], alert, bus)

    # The critic can simulate burns the generator never proposed. Anything it
    # found that clears becomes a selectable option, marked as the agent's own.
    for v in rec.get("variants", []):
        res, args = v["result"], v["args"]
        if not res.get("approved"):
            continue
        d_ric = [float(x) for x in args["direction_ric"]]
        trade["options"].append(Option(
            label=v["label"], strategy="proposed by the agent",
            direction_ric=d_ric, delta_v_mps=float(args["delta_v_mps"]),
            burn_offset_s=float(args["burn_offset_s"]),
            miss_km=float(res["new_miss_km"]), pc=float(res["new_pc"]),
            altitude_drift_km=float(res["altitude_drift_km"]),
            secondary_count=len(res.get("secondary_conjunctions", [])),
            fuel_pct_of_budget=round(100.0 * float(args["delta_v_mps"])
                                     / c.dv_budget_mps, 1),
            decision_time_s=float(args["burn_offset_s"]), feasible=True,
            pros=["Found by the agent, not the generated family"],
            cons=[]))
    if rec.get("variants"):
        _recharacterize(trade["options"], c)
        bus.emit("status", f"Agent added {len(rec['variants'])} simulated variant(s) "
                           f"to the trade space")

    st.session_state["done"] = dict(
        schema=STATE_SCHEMA, env_n=env_n, trade=trade, rec=rec, state=state,
        events=events, alert=alert, cloud=cloud, pre=pre, thr=thr,
        tca_pt=r_at[:, -1], r0=r0, v0=v0, tr=tr, tv=tv, horizon=horizon,
        hero=hero.name, tname=tname, gast=float(t0.gast), cc=cc.residual_km,
        watch=len(res.watching), swept=res.catalog_size, states=res.states,
        sweep_s=res.elapsed_s, fleet_n=len(fleet), elapsed=time.time() - t_start,
        supervised=supervised, authorized=False, chosen=None)

d = st.session_state["done"]
alert, trade, rec = d["alert"], d["trade"], d["rec"]
options = trade["options"]
log_slot.markdown(log_html(d["events"], set(TAG)), unsafe_allow_html=True)

feasible = [o for o in options if o.feasible]
if not feasible:
    plot_slot.plotly_chart(globe(d["cloud"], [(d["pre"], ICE, d["hero"], 5),
                                              (d["thr"], ALERT, d["tname"], 4)],
                                 gast=d["gast"], env_n=d.get("env_n", env_n)),
                           use_container_width=True, key="globe_nofix")
    st.error("No option in the trade space satisfies every constraint — "
             "escalated to a human operator.")
    st.stop()


def fly(opt):
    """Post-burn arc and burn point for one option."""
    _, rb, vb = propagate(d["r0"], d["v0"], opt.burn_offset_s, dt_s=2.0)
    dv = np.array(opt.direction_ric, float)
    v_post = apply_burn(rb[:, -1], vb[:, -1], dv / np.linalg.norm(dv) * opt.delta_v_mps)
    _, post, _ = propagate(rb[:, -1], v_post, d["horizon"] - opt.burn_offset_s, dt_s=15.0)
    return rb[:, -1], v_post, post


# ---------------------------------------------------------------- compute banner
st.markdown("<p class='kes-lab'>Trade space — every option flown and re-screened "
            "against the catalogue</p>", unsafe_allow_html=True)
k = st.columns(5)
k[0].metric("Options costed", trade["scenarios"])
k[1].metric("States evaluated", f"{trade['states_evaluated']/1e6:.0f} M")
k[2].metric("Resident tensor", f"{trade['tensor_gb']:.2f} GB")
k[3].metric("Compute time", f"{trade['elapsed_s']:.1f} s")
k[4].metric("Feasible", f"{len(feasible)}/{trade['scenarios']}")
st.caption(f"{trade['scenarios']} strategies × {trade['passes']} calibration passes, "
           f"each re-propagated and screened against {trade['screened_objects']:,} objects "
           f"over {trade['epochs']:,} epochs · backend {trade['backend']} · "
           f"host→device {trade['transfer'].get('bytes', 0)/1e6:.0f} MB")

# ---------------------------------------------------------------- comparison
rows = []
for o in options:
    rows.append({
        "Option": o.label,
        "Strategy": o.strategy,
        "Clearance (km)": round(o.miss_km, 2),
        "Fuel (m/s)": round(o.delta_v_mps, 3),
        "% budget": round(o.fuel_pct_of_budget, 0),
        "Commit in (min)": round(o.burn_offset_s / 60, 1),
        "Orbit drift (km)": round(o.altitude_drift_km, 2),
        "New conjunctions": o.secondary_count,
        "Verdict": "feasible" if o.feasible else "· ".join(o.failed),
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown(f"<p class='kes-lab'>Agent recommendation — {rec.get('recommended','—')}"
            f"</p>", unsafe_allow_html=True)
st.info(f"**{rec.get('recommended')}** — {rec.get('rationale','')}"
        + (f"  \n*Runner-up: {rec['runner_up']} — {rec.get('why_not','')}*"
           if rec.get("runner_up") else ""))

# ---------------------------------------------------------------- pros / cons
st.markdown("<p class='kes-lab'>Trade-offs</p>", unsafe_allow_html=True)
cols = st.columns(min(len(feasible), 4))
for i, o in enumerate(feasible[:4]):
    with cols[i]:
        flag = " ★" if o.label == rec.get("recommended") else ""
        st.markdown(f"**{o.label}{flag}**")
        st.caption(o.strategy)
        for x in o.pros:
            st.markdown(f"<span style='color:{NOMINAL}'>+</span> "
                        f"<span style='font-size:.82rem'>{x}</span>",
                        unsafe_allow_html=True)
        for x in o.cons:
            st.markdown(f"<span style='color:{AMBER}'>−</span> "
                        f"<span style='font-size:.82rem'>{x}</span>",
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- selection
st.markdown("<p class='kes-lab'>Flight director decision</p>", unsafe_allow_html=True)
labels = [o.label for o in feasible]
default = labels.index(rec["recommended"]) if rec.get("recommended") in labels else 0
choice = st.radio("Select the maneuver to execute", labels, index=default,
                  horizontal=True, label_visibility="collapsed")
chosen = next(o for o in feasible if o.label == choice)

r_burn, v_post, post = fly(chosen)
plot_slot.plotly_chart(
    globe(d["cloud"],
          [(d["pre"], "#55637C", "Original path", 3), (d["thr"], ALERT, d["tname"], 4),
           (post, NOMINAL, f"{d['hero']} — {chosen.label}", 7)],
          [(d["tca_pt"], ALERT, "Impact point"), (r_burn, AMBER, "Burn")],
          gast=d["gast"], env_n=d.get("env_n", env_n)),
    use_container_width=True, key="globe_final")

go_cols = st.columns([1, 1, 3])
execute = go_cols[0].button("Execute this maneuver", type="primary")
hold = go_cols[1].button("Hold — take none")

if hold:
    st.session_state["done"] = None
    st.rerun()

if execute:
    d["chosen"] = chosen.label
    d["authorized"] = True

if d.get("chosen") == chosen.label and d.get("authorized"):
    who = "authorized by the flight director" if d.get("supervised") else \
          "selected autonomously"
    btn_slot.success(f"MANEUVER ISSUED — {chosen.label}. {d['hero']} clears "
                     f"{d['tname']} by {chosen.miss_km:.2f} km for "
                     f"{chosen.delta_v_mps:.3f} m/s. {who.capitalize()}.")
    with stat_slot:
        m = st.columns(5)
        m[0].metric("Objects screened", f"{d['swept']:,}")
        m[1].metric("Clearance", f"{chosen.miss_km:.2f} km",
                    f"+{chosen.miss_km - alert['miss_km']:.2f} km")
        m[2].metric("Collision risk", f"{chosen.pc:.0e}", "cleared", delta_color="off")
        m[3].metric("Fuel spent", f"{chosen.delta_v_mps:.3f} m/s")
        m[4].metric("Alert to decision", f"{d['elapsed']:.1f} s")

    st.markdown("<p class='kes-lab'>The encounter, seen from the satellite — "
                "origin is the spacecraft</p>", unsafe_allow_html=True)
    st.plotly_chart(encounter(d["r0"], d["v0"], r_burn, v_post, d["tr"], d["tv"],
                              alert["tca_offset_s"], chosen.burn_offset_s,
                              alert["miss_km"], chosen.miss_km,
                              alert["rel_speed_kms"]),
                    use_container_width=True, key="encounter_closeup")
    st.caption(f"{trade['states_evaluated']/1e6:.0f}M states across "
               f"{trade['scenarios']} scenarios · {d['fleet_n']} assets screened "
               f"against {d['swept']:,} objects · independent propagators agree to "
               f"{d['cc']*1000:.0f} m · uplink SIMULATED")

if st.button("Reset"):
    st.session_state["done"] = None
    st.rerun()
