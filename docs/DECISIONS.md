# Decision log

The chronology, including what was tried and abandoned. Written so someone
picking this up does not re-litigate settled questions or re-make known mistakes.

---

## 1. Pitch deck (before any code existed)

The project began as a hackathon pitch, not a codebase. The deck came first and
the code was built to make the deck true.

**Iterations, in order:**
1. An HTML deck published as a Claude artifact — dark radar-console aesthetic,
   live canvas globe, 11 slides.
2. Rebuilt "YC style" on request: **one big idea per slide**, huge type, no
   paragraphs.
3. Converted to PowerPoint. No `pptxgenjs`, no `python-pptx`, no LibreOffice on
   the machine and npm was blocked, so **the .pptx was hand-built as raw OOXML**
   with Python's `zipfile` — see the generator in the scratch history. It
   validates and imports cleanly into Google Slides.
4. Cut to **5 slides**, minimum type size **18 pt**, with an animated orbital GIF
   embedded on the title slide.

**Deck numbers that must stay consistent with the code:**

| Claim | Where it came from |
|---|---|
| 12-hour bottleneck → **4-second** autonomous loop | Cameron's framing, used verbatim |
| 128 GB unified memory, zero PCIe copies | GB10 spec; now measured by `accel.py` |
| Acer Veriton GN100 · NVIDIA GB10 Grace Blackwell | Named explicitly on slides 1 and 5 |
| 0.412 km → 2.48 km, 0.189 m/s | The seeded demo encounter |

**Unresolved tension:** slide 5's punchline is `requires_human_ack: false`
("nobody approved this burn"), but the current UI hands a human five costed
options to choose between. Both modes are built and switchable. **Pick one
before presenting and re-cut the slide to match.**

**Artifacts outside the repo:** `Project-Kessler.pptx` and `kessler-orbit.gif`
(760 px, 80 frames, seamless 4.8 s loop) are on Cameron's Desktop. The GIF loops
perfectly because every object's orbital period is an integer harmonic of the
loop length — regenerating it naively will produce a visible seam.

## 2. Demo architecture

Built from a technical plan Cameron supplied: skyfield + CelesTrak TLEs, a
conjunction engine, two LLM agents (a proposer and a critic with tool calling),
and a Streamlit UI. That plan was followed closely; the departures are in
`CLAUDE.md` under the non-obvious decisions.

**A second architecture was proposed mid-project** — two vision-based ground
station agents doing bearing calculation, triangulation and consensus, feeding a
human review dashboard and an authorisation gate. The instruction was "consider
it, but don't change a lot if you don't have to."

What was adopted from it:
- **Consensus cross-check** — mapped onto something real: the encounter geometry
  is computed by two independent propagators and the residual reported.
- **Engagement log** — auditable record per conjunction, persisted.
- **Human authorisation gate** — SUPERVISED vs AUTONOMOUS.

What was **not** adopted, deliberately: the two vision/bearing station agents.
The inputs here are TLEs, which are already position estimates from someone
else's radar. Simulating optical sensors on top would have made the demo less
honest, not more. If real multi-station observation is ever wanted, that needs a
different data source (optical or radar tracklets) and is a day of work.

## 3. The trade space (why the hardware is needed)

Originally the agent produced **one** avoidance burn. That needs a laptop, which
undercuts the entire GB10 argument.

Changed on request to generate **five physically distinct strategies**, each
fully costed: flown, then re-screened against ~6,900 real catalogued objects for
fresh conjunctions, across two calibration passes. **~99 million states per
run.** A human then compares the trade-offs and chooses.

The five: minimum fuel (early, cheap), balanced, latest commit (most decision
time, more fuel), drop back (retrograde), plane shift (holds altitude and timing,
costs ~10× — usually infeasible, and instructive precisely because it fails).

The agent then **probes beyond** that generated set through the physics tool.

## 4. LeoLabs

Initially assessed as unusable — their sharing terms bar commercial and
promotional use, and a hackathon deck with investor-meeting prizes looked
promotional.

**That was over-cautious and was corrected.** A hackathon build is squarely the
"educational" carve-out their terms permit. On re-examination their public
visualisations are served without authentication and carry no `X-Frame-Options`
or CSP `frame-ancestors`, so they embed directly.

Now embedded with the required attribution, which **must stay attached to the
frame**. The authenticated LeoLabs *API* still needs a key nobody has; nothing
depends on it.

## 5. The debris-count question

The pitch cites ~140 million pieces of debris. The instinct was to inflate the
rendered catalogue to match.

Resolved without fabricating anything: the number is real — ESA's figure for
fragments **larger than 1 mm** — but essentially none of them are tracked. Public
CelesTrak carries only ~2,600 debris TLEs; every debris group was probed and
that is all there is.

So the globe has four honest layers: real payloads, real tracked debris, a
**modeled** sub-centimetre cloud sampled from published shell distributions and
labelled as such, and Earth. **The modeled cloud never enters a computation.**

The correct phrasing for the pitch is *"larger than a millimetre, and 99.97% of
them are untracked"* — which is a stronger line than the bare number, and it is
also the answer when a judge asks whether those dots are real.

## 6. NVIDIA integration

Added late, after the submission form made clear the project had nothing NVIDIA
in it while the deck claimed GB10 advantages:

- **Nemotron** backend for both agents over an OpenAI-compatible endpoint.
- **CuPy** for the screening sweep, with transfer instrumentation so the
  unified-memory claim is measured rather than asserted.
- **OpenClaw/NemoClaw** deployment — MCP tool server plus SKILL.md guidance
  files, with `issue_burn` refusing anything the engine has not cleared.

All three are written and stub-tested. **None has run against live hardware or a
live endpoint.**

## 7. Abandoned or rejected

- **Browser-based visual QA of the deck.** Considerable time went into rendering
  and screenshotting slides through a headless browser; it was slow and the
  feedback was "you're making little progress and burning a lot of compute."
  Deterministic checks (font-metrics wrap simulation, overlap detection) proved
  far more useful than screenshots.
- **Embedding a webfont in the deck.** CSP and licensing made it impractical;
  system font stacks with strong typographic treatment were used instead.
- **Auto-rotating the Streamlit globe.** Plotly cannot do it without custom JS.
  The pre-rendered GIF covers that need for the deck.
