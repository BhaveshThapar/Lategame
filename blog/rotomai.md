# My bot beats the standard baseline 73% of the time. Humans beat it 65% of the time.

Both numbers are correct. That gap is the most useful thing I measured all year, and I only found it
because I made myself go play the humans.

---

## The setup

I spent a while building a competitive Pokémon Showdown agent: behaviour cloning on human replays,
then offline RL, then PPO self-play. 4.56M parameters, entity transformer over a 761-dimensional
observation of the current turn. It runs entirely on CPUs — at that size a GPU does nothing except
lengthen the queue, and the binding constraints turned out to be RAM and how fast a local Showdown
server can simulate battles (209–242 per minute at concurrency 20).

The rule I set at the start was that every published number had to be gated on an evaluation
declared *before* the run, and negative results had to be kept. That is now 251 committed result
files and a build log where about a third of the entries are things that did not work.

This post is about the three findings from that discipline that have nothing to do with Pokémon.

---

## Finding 1: the same weights don't score the same twice

Here is the thing nobody tells you about evaluating a self-play agent.

I re-scored a set of checkpoints — the *same files*, same opponent, same settings, same code — in two
separate cluster runs. They came back at **0.472** and **0.499**.

That is 0.027 apart, against a per-read standard error of 0.023. Nothing differed but the run.

So I adopted a rule: **a cross-run difference under about 0.03 is not a result.** Then I went back
through the build log and re-labelled several comparisons I had already computed and believed.

A later build re-measured the same phenomenon independently on a different opponent at a larger n.
Individual checkpoints swung by up to **0.028** across runs. The **seed-pooled** read of those same
arms spanned **0.0097** — six times tighter.

That is the actual lesson. Pooling across seeds is what buys you run-to-run stability. Raising n
within one seed does not, because the between-seed standard deviation here is about 0.0706 — roughly
2.5× the within-seed binomial. More battles per seed shrinks the wrong variance.

And these are two different claims that get conflated constantly:

- *"These checkpoints score X"* — a pooled read supports this.
- *"This training method produces agents that score X"* — this needs the seed-level analysis, and
  the interval is much wider.

The first is bigger and easier to compute. It is also usually not the one you meant.

## Finding 2: I threw away a diagnostic because it couldn't measure itself

I wanted to know whether a training stage was gradient-noise-limited, so I pre-registered a gate on
the squared gradient norm, sized to detect a 2.2× difference between two conditions.

Before running it, I probed a **fixed** checkpoint twice, changing only the probe seed.

It moved **2.7×**.

The instrument's own noise was larger than the effect it had been built to detect. So the diagnostic
was retired rather than reported, along with a rule: at this budget, a gradient-norm difference under
~3× is not measurable, and any future probe runs at least two probe seeds. Exactly one quantity
survived the probe-seed swing — a cosine similarity that moved 0.312 → 0.317 — and that is the only
one I kept.

I think this is a common failure and an invisible one. You adopt a diagnostic because it is cheap and
because a paper you respect used it. You never run the null. When the seed noise exceeds the effect,
you do not get weak evidence — you get *no* evidence that looks exactly like a measurement.

## Finding 3: the baseline number does not mean what everyone uses it to mean

Every agent in this space reports a win rate against a fixed rule-based opponent. For poke-env-based
work that is `SimpleHeuristicsPlayer`. I report it too, on purpose, because it is the one number an
outsider can situate:

> **0.7303** [0.7211, 0.7394] on gen9ou, n=9,000 — three independent runs across two nodes, pooled.

I also ran a control arm: the same protocol at the same n against an in-repo baseline whose value I
had already published at 0.7513. It came back at 0.7504 — **0.0009** off. That control is the thing
that lets me claim the 0.7303 is a property of the *agent* and not of the *run*. Without it, given
Finding 1, it isn't.

Then I played the public ranked ladder.

I pre-registered it like everything else: n=100, the exact command, the account, and the bands —
WEAK, DECENT, STRONG — all committed to a file before the first rated game, including an explicit
line saying a weak result would be published too.

> **Elo 1004 / GXE 30%.** Record 35–65. Pre-registered **WEAK**.

Same checkpoint. 0.7303 against the standard bot baseline; 0.350 against roughly 1055-Elo humans.

Both measurements are correct. They measure different things, and I could not have known how
differently without running the second one.

The reason isn't mysterious once you say it out loud. A rule-based baseline is a *fixed policy*. An
agent trained by self-play against a fixed heuristic will find that policy's stable weaknesses and
exploit them, and a win rate against it partly measures how thoroughly it has done so. Human ladder
opponents are nonstationary, adversarial and strategically diverse. Nothing about the first number
bounds the second.

**A bot-baseline win rate does not predict human-ladder performance.** And no instrument I had —
every one of them pointed at a field of bots — could have shown me that.

---

## What I actually did wrong (systems edition)

Two bugs, both silent, both of which corrupted comparisons rather than crashing.

**Two jobs shared a simulator.** poke-env hardcodes `ws://localhost:8000`. Slurm array tasks
routinely land on the same node — I watched tasks 1 and 2 of one array do exactly that — so two
training arms opened the *same* Showdown server and battled into each other's games. No error, no
warning, a comparison that looked completely fine. Fix: derive the port from the array index.

**Two jobs shared an output path.** A gate's `--out` is a global. Two overlapping jobs with the same
path both write it, the later one wins, and the earlier run's results file simply stops existing
while its number lives on only in a log. One build lost two of six reads that way.

Same shape both times: a default that is correct for one job and silently wrong for two. When the
effect you are chasing is 0.03 and your run-to-run noise is 0.028, a silent contamination isn't a
nuisance. It's the entire result.

---

## What I'd tell you if you're starting one of these

1. **Measure your instrument before you measure with it.** Same checkpoint, same opponent, different
   run. Whatever spread you get is your floor. Mine was 0.03 and it invalidated work I'd already
   done.
2. **Pool over seeds — and say which claim you're making.** "These weights" and "this method" have
   very different error bars.
3. **One small-n cell can't carry a comparison.** My own record held an n=300 cell at 0.8167 for a
   checkpoint that scored 0.741, 0.769 and 0.761 at n=1,000.
4. **Pre-register the bad band too.** The ladder result only means anything because WEAK was defined
   and committed before game one.
5. **Go play the real population.** 100 rated ladder games took an afternoon. The training they
   evaluate took 107 task-hours. That is an absurdly good trade and I put it off for months.

---

The weak ladder number is the one I was most tempted not to publish, and it is the only one that
taught me something I could not have gotten another way. Everything above is in the repo, including
the runs that didn't work:
**[github.com/BhaveshThapar/RotomAI](https://github.com/BhaveshThapar/RotomAI)**

*Longer version with the full methodology, the threats to validity, and the figures:
[`paper/rotomai.md`](https://github.com/BhaveshThapar/RotomAI/blob/main/paper/rotomai.md).*
