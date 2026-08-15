#!/usr/bin/env python
"""Project Kessler -- end-to-end console demo.

    python run_demo.py                    # default: seeded threat, ~92 min to TCA
    python run_demo.py --hero ISS         # protect a different asset
    python run_demo.py --organic          # skip the seed, fly whatever the catalog gives you
"""
from __future__ import annotations

import argparse
import time

from kessler.agents import run_resolution
from kessler.assurance import Authorization, EngagementLog, Mode, cross_check
from kessler.bus import Bus, Event
from kessler.catalog import load_demo_catalog
from kessler.conjunction import screen
from kessler.mission import (Constraints, MissionState, find_tca, requires_action,
                             synthesize_threat, altitude_shortlist)
from kessler.monitor import select_fleet, sweep_fleet
from kessler.physics import elements, teme_state, timescale

C = {"reset": "\033[0m", "dim": "\033[2m", "b": "\033[1m", "amber": "\033[38;5;214m",
     "ice": "\033[38;5;110m", "red": "\033[38;5;203m", "green": "\033[38;5;79m",
     "grey": "\033[38;5;245m"}
STYLE = {"status": C["grey"], "alert": C["red"], "agent1": C["ice"],
         "agent2": C["amber"], "tool": C["grey"], "verdict": C["green"],
         "error": C["red"], "clear": C["green"]}
LABEL = {"status": "SYS ", "alert": "!!!!", "agent1": "FDO ", "agent2": "MAD ",
         "tool": "PHYS", "verdict": ">>>>", "error": "ERR ", "clear": "OK  "}


def printer(ev: Event) -> None:
    col = STYLE.get(ev.kind, "")
    print(f"{C['dim']}{time.strftime('%H:%M:%S')}{C['reset']} "
          f"{col}{LABEL.get(ev.kind, ev.kind):<4}{C['reset']} {col}{ev.text}{C['reset']}",
          flush=True)


def rule(title: str = "") -> None:
    print(f"\n{C['dim']}{'-' * 78}{C['reset']}")
    if title:
        print(f"{C['b']}{title}{C['reset']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", default="STARLINK-1008")
    ap.add_argument("--limit", type=int, default=None, help="cap catalog size")
    ap.add_argument("--tca-min", type=float, default=92.0)
    ap.add_argument("--miss-km", type=float, default=0.412)
    ap.add_argument("--organic", action="store_true",
                    help="do not seed a threat; screen for a real one")
    ap.add_argument("--reload", action="store_true", help="force a fresh CelesTrak pull")
    ap.add_argument("--monitor", action="store_true",
                    help="fleet monitor: sweep many assets, auto-resolve breaches")
    ap.add_argument("--fleet", default="STARLINK", help="asset name filter for --monitor")
    ap.add_argument("--fleet-size", type=int, default=10)
    ap.add_argument("--supervised", action="store_true",
                    help="hold approved burns at a human authorization gate")
    args = ap.parse_args()

    bus = Bus(sink=printer)
    ts = timescale()
    elog = EngagementLog()
    op_mode = Mode.SUPERVISED.value if args.supervised else Mode.AUTONOMOUS.value

    rule("PHASE 1  INGEST")
    t_load = time.time()
    cat = load_demo_catalog(limit=args.limit, reload=args.reload)
    bus.emit("status", f"{len(cat)} objects from {cat.source} in {time.time()-t_load:.2f}s")

    if args.monitor:
        rule("FLEET MONITOR")
        fleet = select_fleet(cat.objects, args.fleet, args.fleet_size)
        if not fleet:
            bus.emit("error", f"no objects matching {args.fleet!r}")
            return 1
        c = Constraints()
        res = sweep_fleet(fleet, cat.objects, t0 := ts.now(), horizon_s=6 * 3600,
                          coarse_step_s=60.0, threshold_km=25.0, constraints=c)
        bus.emit("status", f"swept {len(res.statuses)} assets against {res.catalog_size:,} "
                           f"objects in {res.elapsed_s:.2f}s")
        bus.emit("status", f"one propagation: {res.states:,} states, "
                           f"{res.matrix_mb:.0f} MB resident, reused per asset")
        print(f"\n  {'STATUS':<8}{'ASSET':<20}{'ALT':>9}  {'CLOSEST':<26}{'MISS':>9}{'Pc':>11}")
        for s_ in res.statuses:
            w = s_.worst
            print(f"  {s_.severity:<8}{s_.name[:19]:<20}{s_.alt_km:8.1f}k  "
                  f"{(w.secondary[:25] if w else '—'):<26}"
                  f"{(f'{w.miss_km:.2f} km' if w else '—'):>9}"
                  f"{(f'{w.pc:.1e}' if w else '—'):>11}")
        print()
        bus.emit("status", f"{len(res.actionable)} asset(s) breach the "
                           f"{c.min_miss_km:.1f} km minimum · {len(res.watching)} on watch")
        if not res.actionable:
            bus.emit("clear", "NO ACTION REQUIRED across the fleet — detection is not "
                              "the same as needing to maneuver.")
        for s_ in res.actionable:
            rule(f"DISPATCHING AGENTS — {s_.name}")
            tgt = next(a for a in fleet if a.name == s_.name)
            hr, hv = teme_state(tgt, t0)
            hel = elements(hr, hv)
            obj = cat.objects[s_.worst.secondary_index]
            tr_, tv_ = teme_state(obj, t0)
            e2 = find_tca(hr, hv, tr_, tv_, horizon_s=s_.worst.tca_offset_s * 1.4)
            st_ = MissionState(tgt.name, hr, hv, obj.name, tr_, tv_, t0,
                               nominal_alt_km=(hel["perigee_alt_km"] + hel["apogee_alt_km"]) / 2,
                               constraints=c)
            al = {"primary": tgt.name, "secondary": obj.name,
                  "tca_offset_s": e2["tca_offset_s"], "miss_km": e2["miss_km"],
                  "pc": e2["pc"], "rel_speed_kms": e2["rel_speed_kms"]}
            run_resolution(st_, altitude_shortlist(cat.objects, hel, 50.0), al, bus)
        return 0

    hero = cat.by_name(args.hero)
    if hero is None:
        bus.emit("error", f"no object matching {args.hero!r}")
        return 1
    t0 = ts.now()
    r0, v0 = teme_state(hero, t0)
    el = elements(r0, v0)
    bus.emit("status", f"Protecting {hero.name} -- {el['alt_km']:.1f} km, "
                       f"inc {el['inc_deg']:.2f} deg, period {el['period_min']:.1f} min")

    rule("PHASE 2  CONJUNCTION SCREENING")
    if args.organic:
        t_s = time.time()
        found = screen(hero, cat.objects, t0, horizon_s=6 * 3600,
                       coarse_step_s=60, threshold_km=25.0)
        bus.emit("status", f"screened {len(cat)} objects x 6 h in {time.time()-t_s:.2f}s")
        if not found:
            bus.emit("error", "no natural conjunction inside the window; rerun without --organic")
            return 1
        c = found[0]
        threat_sat = cat.objects[c.secondary_index]
        tname = threat_sat.name
        tr, tv = teme_state(threat_sat, t0)
        enc = find_tca(r0, v0, tr, tv, horizon_s=c.tca_offset_s * 1.4)
    else:
        tca_s = args.tca_min * 60.0
        tname, tr, tv = synthesize_threat(r0, v0, tca_s, miss_km=args.miss_km)
        bus.emit("status", f"seeded threat {tname} (deterministic demo encounter)")
        t_s = time.time()
        enc = find_tca(r0, v0, tr, tv, horizon_s=tca_s * 1.3)
        bus.emit("status", f"engine measured the encounter in {time.time()-t_s:.2f}s")

    shortlist = altitude_shortlist(cat.objects, el, pad_km=50.0)
    bus.emit("status", f"altitude pre-filter: {len(cat)} -> {len(shortlist)} screenable objects")

    bus.emit("alert", f"CONJUNCTION -- {hero.name} vs {tname}")
    bus.emit("alert", f"  TCA T+{enc['tca_offset_s']/60:.1f} min   miss {enc['miss_km']:.3f} km   "
                      f"rel {enc['rel_speed_kms']:.2f} km/s   Pc {enc['pc']:.2e}")
    bus.emit("alert", f"  radial {enc['radial_km']:+.3f}  in-track {enc['in_track_km']:+.3f}  "
                      f"cross-track {enc['cross_track_km']:+.3f} km")

    state = MissionState(
        hero_name=hero.name, hero_r0=r0, hero_v0=v0,
        threat_name=tname, threat_r0=tr, threat_v0=tv, t0=t0,
        nominal_alt_km=(el["perigee_alt_km"] + el["apogee_alt_km"]) / 2.0,
        constraints=Constraints())

    alert = {
        "primary": hero.name, "secondary": tname,
        "tca_offset_s": enc["tca_offset_s"], "miss_km": round(enc["miss_km"], 4),
        "rel_speed_kms": round(enc["rel_speed_kms"], 3), "pc": enc["pc"],
        "radial_km": round(enc["radial_km"], 3),
        "in_track_km": round(enc["in_track_km"], 3),
        "cross_track_km": round(enc["cross_track_km"], 3),
    }

    cc = cross_check(hero, t0, enc["tca_offset_s"])
    bus.emit("status", f"consensus check -- SGP4 vs RK4+J2 agree to {cc.residual_km*1000:.1f} m "
                       f"at TCA ({'consistent' if cc.consistent else 'DIVERGENT'})")

    eng = elog.open(primary=hero.name, secondary=tname, seeded=not args.organic,
                    geometry=alert, consensus=cc.as_dict(), mode=op_mode)

    act, why = requires_action(enc, state.constraints)
    eng.action_required, eng.assessment = act, why
    if not act:
        eng.authorization = Authorization.NOT_REQUIRED.value
        bus.emit("clear", f"NO ACTION REQUIRED -- {why}")
        rule("RESULT")
        print(f"  {C['green']}Conjunction logged, no maneuver.{C['reset']}  {why}")
        print(f"  engagement {eng.engagement_id} closed as {eng.authorization}\n")
        return 0
    bus.emit("alert", f"ACTION REQUIRED -- {why}")

    rule("PHASE 3  AGENT RESOLUTION")
    t_a = time.time()
    out = run_resolution(state, shortlist, alert, bus)
    elapsed = time.time() - t_a
    eng.proposal = out.get("proposal")
    eng.engine_verdict = out.get("result")
    eng.agent_rounds = out.get("rounds", 0)
    eng.record_reasoning(bus.events)

    if out["approved"]:
        if op_mode == Mode.AUTONOMOUS.value:
            eng.authorize("agent (autonomous mode)")
            bus.emit("verdict", "Gate self-authorized -- AUTONOMOUS mode, decision recorded.")
        else:
            rule("AUTHORIZATION GATE")
            p_ = out["proposal"]; r_ = out["result"]
            print(f"  engagement {eng.engagement_id}")
            print(f"  burn {p_['delta_v_mps']:.4f} m/s -> miss {r_['new_miss_km']:.3f} km, "
                  f"Pc {r_['new_pc']:.1e}")
            try:
                ans = input(f"  {C['amber']}AUTHORIZE this burn? [y/N] {C['reset']}").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                eng.authorize("console operator")
                bus.emit("verdict", "AUTHORIZED by operator.")
            else:
                eng.halt("console operator", "halted at operator review")
                bus.emit("error", "HALTED by operator -- no command issued.")

    rule("RESULT")
    if out["approved"] and eng.authorization == Authorization.HALTED.value:
        print(f"  {C['red']}HALTED by {eng.authorized_by} — no command issued.{C['reset']}\n")
        return 0
    if out["approved"]:
        p, r = out["proposal"], out["result"]
        print(f"  {C['green']}APPROVED in {out['rounds']} round(s), {elapsed:.1f}s{C['reset']}")
        print(f"  burn        {p['delta_v_mps']:.4f} m/s  RIC {p['direction_ric']}  "
              f"at T+{p['burn_offset_s']:.0f}s")
        print(f"  miss        {alert['miss_km']:.3f} km  ->  {r['new_miss_km']:.3f} km")
        print(f"  Pc          {alert['pc']:.2e}  ->  {r['new_pc']:.2e}")
        print(f"  altitude    drift {r['altitude_drift_km']:.3f} km")
        print(f"  secondary   {len(r['secondary_conjunctions'])} new conjunctions")
        print(f"  consensus   SGP4 vs RK4+J2 within {cc.residual_km*1000:.1f} m")
        print(f"  engagement  {eng.engagement_id} · {eng.authorization} "
              f"by {eng.authorized_by}")
        print(f"  uplink      {eng.uplink}")
    else:
        print(f"  {C['red']}NO SAFE MANEUVER after {out['rounds']} rounds ({elapsed:.1f}s){C['reset']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
