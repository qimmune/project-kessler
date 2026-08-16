# Screen recording script

Read the **SAY** lines out loud. Do the **DO** lines. Nothing else.

Total ≈ 3:00. The engine takes ~12 seconds to think, twice — the words to fill
that time are written in, so you never sit in silence.

**Before you hit record:** open the app, press **Reset** if anything is on
screen, and leave **Advanced** closed.

---

# PART 1 · What you're looking at  (0:00 – 0:55)

**DO** — Nothing. Let the globe sit there for three seconds.

> **SAY:** "This is low Earth orbit right now. Every dot is real, from the public
> tracking catalogue."

**DO** — Point the cursor at the green dots.

> **SAY:** "Green is a working satellite. There are about fifteen thousand."

**DO** — Point at the red dots.

> **SAY:** "Red is debris we can actually track. About two and a half thousand."

**DO** — Point at the faint dark cloud.

> **SAY:** "The faint cloud is the junk nobody can track. Over a hundred million
> fragments bigger than a millimetre. That layer is a model — it never touches
> any calculation."

**DO** — Click **Advanced** to open it.

> **SAY:** "These are the operator's controls."

**DO** — Point at each slider as you name it.

> **SAY:** "How many satellites we're protecting. How far ahead we look."

**DO** — Rest the cursor on **Separation minimum**.

> **SAY:** "How much clearance we insist on."

**DO** — Move to **Δv budget**.

> **SAY:** "And how much fuel we'll spend to get it. Those two fight each other,
> and that fight is the whole demo."

**DO** — Close **Advanced**.

---

# PART 2 · The easy case  (0:55 – 2:00)

Settings are already correct — these are the defaults.
*(Separation 2.0 km · Δv budget 0.35 m/s)*

**DO** — Click **ENGAGE ORBITAL TRAFFIC CONTROL**.

> **SAY:** "It's screening eighteen thousand objects against twelve of our
> satellites."

**DO** — Wait. The log streams. *(~12 seconds — keep talking)*

> **SAY:** "And it's found one. A piece of Cosmos 1408 debris — that's a satellite
> Russia blew up in a missile test — passing four hundred metres from one of ours
> in ninety-two minutes. Closing at eleven kilometres a second."
>
> "At that speed a fragment the size of a marble hits like a hand grenade."

**DO** — When the numbers appear, point at **States evaluated**.

> **SAY:** "Here's the part that needs the hardware. It didn't work out one
> answer. It worked out five different ways to dodge — and it actually flew every
> single one, then re-checked each new orbit against the whole catalogue to make
> sure the dodge doesn't cause a *different* collision."
>
> "A hundred and fifty million position calculations. Twelve seconds."

**DO** — Point at the comparison table, top row to bottom.

> **SAY:** "Four of the five work. They're all safe. They differ in what they cost
> you."

**DO** — Point at **Minimum fuel**, then at **Latest commit**.

> **SAY:** "This one's cheapest. This one lets you wait forty-one minutes before
> committing — which matters, because better tracking data might come in."

**DO** — Point at the agent recommendation box.

> **SAY:** "The agent recommends one and says why. But it doesn't fire anything."

**DO** — Point at the **Execute / Hold** buttons.

> **SAY:** "A person decides. Always."

**DO** — Click **EXECUTE THIS MANEUVER**.

> **SAY:** "Four hundred metres becomes two kilometres. For about a seventh of a
> metre per second of fuel — that's slower than walking pace."

**DO** — Scroll down to the close-up plot.

> **SAY:** "That's the encounter from the satellite's point of view. The satellite
> is the diamond in the middle. Red is where the debris would have gone. Green is
> where it goes now."

---

# PART 3 · The hard case  (2:00 – 3:00)

**DO** — Click **Reset**. Open **Advanced**.

> **SAY:** "Same debris. Same ninety-two minutes' warning. But now say this
> satellite is carrying something we really can't lose."

**DO** — Drag **Separation minimum** from `2.0` to **`5.0`**.

> **SAY:** "So we demand five kilometres of clearance instead of two."

**DO** — Drag **Δv budget** from `0.35` to **`0.45`**.

> **SAY:** "And we'll allow a bit more fuel for it."

**DO** — Close **Advanced**. Click **ENGAGE ORBITAL TRAFFIC CONTROL**.

> **SAY:** "Watch what happens to the options."

**DO** — Wait. *(~12 seconds — keep talking)*

> **SAY:** "Same five strategies. Same physics. The only thing that changed is
> how safe we're insisting on being."

**DO** — Point at **Feasible: 2/5**.

> **SAY:** "Two survive instead of four."

**DO** — Point at the two that passed — both commit early.

> **SAY:** "And look which two. Both of them burn early. Five minutes from now,
> or fourteen."

**DO** — Point at **Latest commit**, now failing.

> **SAY:** "The one that let us wait forty-one minutes is gone. It only reaches
> three point seven kilometres — not enough."
>
> "It's not that waiting became dangerous. It's that if you wait, you can't
> afford a big enough nudge anymore. Fuel needed goes up as your time goes down."

**DO** — Pause here. This is the line.

> **SAY:** "So we asked to be safer, and the price wasn't fuel. It was the option
> to think about it."

**DO** — Click **EXECUTE THIS MANEUVER**.

> **SAY:** "Five kilometres of clearance. A person made the call — with every
> option priced in front of them, inside a window where a call was still
> possible."
>
> "Doing this by hand takes about twelve hours. By then, the cheap options are
> already gone."

**STOP RECORDING.**

---

## Settings cheat sheet

| | Part 2 | Part 3 |
|---|---|---|
| Assets protected | 12 | 12 |
| Look-ahead | 6 h | 6 h |
| **Separation minimum** | **2.0 km** | **5.0 km** |
| **Δv budget** | **0.35 m/s** | **0.45 m/s** |
| Modeled debris | 45k | 45k |

Only two numbers change. Everything else is default.

## If it goes wrong

- **Press Reset between the two runs.** Otherwise you're looking at the old result.
- **Nothing feasible** — separation is too high for the fuel. Nudge Δv up to 0.5.
- **Too slow** — drop Assets protected to 6 and debris to 15k, then re-record.
- **Numbers slightly different from this script** — expected. The catalogue is
  live. What must hold: Part 2 gives **4 feasible**, Part 3 gives **2**, and
  *Latest commit* dies in Part 3. If that's not what you see, check the sliders.
