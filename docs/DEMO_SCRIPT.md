# 3-minute screen recording — script

Two cases that tell one story. Settings are exact; type them in before you hit
record so the recording is clean.

---

## The control panel, in plain language

Open **Advanced**. Six controls. Think of yourself as air traffic control, but
for satellites.

| Control | What it means | Turn it UP and… |
|---|---|---|
| **Assets protected** | How many of our satellites we're watching | We watch more satellites; the scan takes longer |
| **Look-ahead (h)** | How far into the future we check | We see trouble sooner, but check more of the sky |
| **Separation minimum (km)** | How much room we insist on between our satellite and the junk | We're pickier about safety, so a dodge is harder and costs more fuel |
| **Δv budget (m/s)** | How much fuel we'll allow for one dodge | We can afford a bigger shove |
| **Modeled debris (sample size)** | How much of the untracked junk we *draw* on screen | It looks more crowded. **Cosmetic only — never used in any calculation** |
| **Require human authorization** | Does a person have to say yes before the engine fires? | Ticked = a human approves. Unticked = it decides for itself |

**Two of these fight each other, and that fight is the whole demo:**
**Separation minimum** is how safe you insist on being. **Δv budget** is how much
fuel you'll spend. Demand more safety with the same fuel, and options start
dying.

One more thing about physics that makes the story work: shoving a satellite
forwards or backwards makes it drift further and further from where it would
have been — **the earlier you shove, the more distance you get for the same
fuel.** Waiting is expensive.

---

## Part 1 — the inputs (≈60 s)

Open on the globe. Don't touch anything for the first few seconds.

> "This is every satellite and every piece of tracked debris in low Earth orbit,
> right now, from the public catalogue. Green is a working satellite — about
> fifteen thousand of them. Red is debris we can actually track — about
> two and a half thousand. The faint cloud is the untracked junk, over a hundred
> million fragments bigger than a millimetre that nobody has a position for.
> That one's a model, and it never enters a calculation."

Open **Advanced**. Point at each slider as you say it:

> "These are the operator's controls. How many satellites we're protecting. How
> far ahead we look. How much clearance we insist on. And how much fuel we'll
> spend on one dodge."
>
> "Those last two are the ones that matter. Clearance is how safe you want to
> be. Fuel is what you can afford. The whole job is trading one against the
> other."

---

## Part 2 — the easy case (≈60 s)

**Set exactly:** Assets 12 · Look-ahead 6 h · **Separation minimum 2.0 km** ·
**Δv budget 0.35 m/s** · debris 45k · human authorization **off**

Press **Engage orbital traffic control**.

> "It's screening eighteen thousand objects. And there's the problem —
> a piece of Cosmos 1408 debris passing four hundred metres from one of our
> satellites in ninety-two minutes, closing at eleven kilometres a second."

When the trade space appears:

> "Now the useful part. It didn't generate *one* answer. It generated five
> different ways out, and it flew every one of them — re-checking each new orbit
> against the whole catalogue to make sure the dodge doesn't cause a *different*
> collision. That's about a hundred million position calculations."

Point at the table:

> "Four of the five work. They're all safe — they differ in what they cost you.
> This one is cheapest. This one lets you wait forty minutes before committing,
> which matters because better tracking data might arrive. This one drops the
> orbit instead of raising it."
>
> "The agent recommends one and explains why. But a human picks."

Choose the recommendation, press **Execute this maneuver**.

> "Four hundred metres becomes two kilometres, for about a fifth of a metre per
> second of fuel."

Scroll to the close-up:

> "That's the encounter from the satellite's point of view. The satellite is the
> diamond in the middle. Red is where the debris *would* have gone. Green is
> where it goes now."

---

## Part 3 — the hard case (≈60 s)

**Same threat. Change only two numbers:**
**Separation minimum 2.0 → 5.0 km** · **Δv budget 0.35 → 0.45 m/s**

> "Same debris, same ninety-two minutes. But now suppose this satellite is
> carrying something we really can't lose, so we demand five kilometres of
> clearance instead of two. We'll allow a bit more fuel for it."

Press **Engage** again.

> "Watch what happens to the options."

When it lands — **only 2 of 5 survive**:

> "Two survive. And look *which* two — both of them burn early. Five minutes
> from now, or fourteen."
>
> "The one that let us wait forty minutes is gone. It's not that waiting is
> unsafe — it's that if you wait, you can no longer afford enough of a shove to
> get five kilometres. The fuel needed grows as your time shrinks."
>
> "So the honest answer the system gives the operator is: **your decision time
> just disappeared.** You wanted to be safer, and the price wasn't fuel — it was
> the option to think about it."

> "That is why this has to run in seconds and not hours. By the time a human
> team finished the analysis by hand, the cheap options would already be gone."

Execute, and close:

> "Five kilometres of clearance. Nobody had to do orbital mechanics by hand."

---

## If something goes wrong on the day

- **Nothing feasible** — separation minimum is too high for the fuel budget.
  Lower separation or raise fuel. Never demo `Look-ahead` under about 40 min with
  a tight budget; physics runs out.
- **Too slow** — drop **Assets protected** to 6 and modeled debris to 15k.
- **Stale screen** — press **Reset**.
- **After editing any file under `kessler/`** — restart the server, or the
  browser shows a stale-import error.
