# Spectral Mixing vs. Matrix Multiplication — Two Explanations

Companion notes to the [Spectral Token Mixing whitepaper](spectral-mixing.md).
Part 1 is the rigorous category-by-category comparison; Part 2 is the same
story told for a general audience.

---

# Part 1 — Can spectral mixing have more quality than the matmul? (by category)

The honest headline first: **on raw accuracy for arbitrary inputs, nothing can
beat the exact matmul — it *defines* the correct answer. But there are five
categories where spectral mixing is mathematically superior, including one
surprising accuracy case.**

## 1. Speed / complexity — spectral wins, asymptotically and provably

The matmul mixing costs O(n²·d): n² pairwise interactions, each explicitly
computed. Spectral mixing costs O(n log n·d). This isn't an engineering claim,
it's a theorem (the FFT's divide-and-conquer structure). At n = 1M,
n² / (n·log₂n) = 1,048,576 / 20 ≈ **52,000× fewer operations for the mixing
step**. The only caveat is the crossover: below a few thousand tokens,
constants dominate and the matmul (which maps perfectly onto tensor cores) is
faster in practice. Spectral's speed win is real but *asymptotic* — it grows
with context.

## 2. Memory — spectral wins by construction

Matmul mixing has an n×n intermediate object. FlashAttention avoids
*materializing* it, but cannot avoid O(n²) computation traffic, and the KV
cache still grows O(n). Spectral mixing never has any n² object — activations
are O(n·d) — and its recurrent form has **O(1) decode state**. Mathematically:
a dense mixing matrix has n² degrees of freedom to touch; a convolution has n;
a distilled state-space has a constant. You cannot beat that by optimizing the
matmul — it's a property of the operator class, not the implementation.

## 3. Accuracy vs. arbitrary inputs — matmul wins, by definition

For an *arbitrary* data-dependent mixing matrix `softmax(QKᵀ)`, exact
evaluation is the ground truth; any structured operator approximates it. The
theory makes this precise: exact attention is quadratic under SETH, and
sub-quadratic approximation requires bounded entries. What's approximated is
specifically **content-based addressing** ("find the token that says X,
wherever it is") — a convolution knows only *relative position*, not
*content*. Gates and hybrid layers recover much of this, but mathematically, a
single spectral layer's function class is a strict subset of a single
attention layer's. No honest framing can invert this category.

## 4. Accuracy in floating point — the surprising case where spectral wins

Restrict to the operation *both* can compute exactly in infinite precision —
convolutional (position-relative) mixing. In finite precision (fp16/bf16,
where LLMs actually live), the two evaluations have *different roundoff
behavior*:

- Direct matmul evaluation: each output is a length-n dot product; roundoff
  error grows like **O(√n)** (stochastic) per output.
- FFT evaluation: roundoff grows like **O(√log n)** — because the DFT is a
  *unitary* transform (condition number exactly 1, Parseval's theorem: it
  preserves norms perfectly), and the computation is a log-depth tree instead
  of a length-n chain.

At n = 1M: √n ≈ 1000 vs √log₂n ≈ 4.5. So **for the convolutional component of
mixing, the spectral route is not just faster — it is genuinely more
numerically accurate than the matmul at the same precision.** Fewer
operations, better conditioned, provably.

## 5. Conditioning and stability — spectral wins structurally

A dense learned matrix can have arbitrarily bad condition number — it can
amplify noise in some directions by orders of magnitude. The spectral operator
is (unitary transform) × (diagonal) × (unitary transform): its conditioning is
exactly the ratio of its largest to smallest filter coefficient, directly
readable and directly controllable. Norm preservation through the transform
also means gradients neither explode nor vanish *through the mixing step
itself* — a structural stability property the dense matmul doesn't offer.

## 6. Quality per FLOP — spectral wins at long context

This is the fair fight for machine learning, because compute is the budget.
Hyena matches Transformer perplexity on The Pile with ~20% fewer total FLOPs;
at equal *compute* (not equal layers), the FLOPs a spectral mixer saves can
buy more layers, more width, or more context — all of which buy quality.
Per-layer, attention is more expressive; per-FLOP at long sequence lengths,
the evidence favors the structured operator.

## 7. Quality of the *reachable* regime — spectral wins where matmul can't compete at all

A subtle but decisive category: at 1M+ tokens, quadratic attention doesn't
produce a *lower-quality* answer — on realistic hardware budgets it produces
**no answer**. On Long Range Arena's hardest tasks, structured operators
(S4-class) solved problems like Path-X where Transformers scored at chance;
HyenaDNA beat Transformers outright on long genomic sequences. When the task's
essential dependencies live beyond the horizon that O(n²) can afford, the
"less expressive" operator has strictly higher quality — because 100% of an
approximate global view beats 0% of an exact one.

## Summary table

| category | winner | why, in one line |
|---|---|---|
| Asymptotic speed | **spectral** | O(n log n) vs O(n²) — theorem, not tuning |
| Memory / decode state | **spectral** | O(n) activations, O(1) recurrent state vs O(n²) traffic + O(n) cache |
| Accuracy, arbitrary inputs | **matmul** | exact is the definition of correct; content addressing is the gap |
| Accuracy, convolutional class, finite precision | **spectral** | unitary FFT: roundoff O(√log n) vs O(√n) |
| Conditioning / stability | **spectral** | condition number 1 transform; diagonal, inspectable spectrum |
| Expressivity per layer | **matmul** | dense data-dependent matrix ⊃ structured operator |
| Quality per FLOP (long ctx) | **spectral** | equal perplexity at fewer FLOPs; savings buy quality elsewhere |
| Quality at extreme lengths | **spectral** | an answer at 1M tokens beats no answer |
| Short context (< ~16k) | **matmul** | constants dominate; tensor cores love GEMM |

## The takeaway

Mathematically, spectral mixing is not a "worse but cheaper" version of the
matmul — it's a **different point on a rigorous trade-off surface**: it gives
up per-layer expressivity (one category, recoverable via gates + a thin hybrid
of exact layers) in exchange for wins in complexity, memory, conditioning,
finite-precision behavior, and the entire regime of context lengths the matmul
cannot reach. Never claim spectral is *more accurate* than exact attention on
arbitrary inputs (it can't be, by definition), but do claim — with math on
your side — that at long context it is faster, lighter, better conditioned,
and the only operator still standing at a million tokens.

---

# Part 2 — The high-school explanation

The whole idea rests on one trick you've already seen in math class — three
examples get you there.

## Step 1: What a matrix multiply actually does — "everyone talks to everyone"

Imagine 4 students in a row, each holding a number:

```
x = [5, 2, 8, 3]
```

A **matrix multiply** makes a new list where *every* student's new number is a
custom recipe using *all* the old numbers. For example, student 1's new number
might be:

```
new₁ = 0.5·(5) + 0.1·(2) + 0.3·(8) + 0.1·(3)
```

and student 2 has a *completely different* recipe, and so on. The recipes are
the matrix — a 4×4 grid of 16 weights:

- 4 students × 4 ingredients each = **16 multiplications = 4²**

Now scale up. With 1,000 students: 1,000,000 multiplications. With a million
students (a million-token document): **a trillion**. This is the "everyone
shakes hands with everyone" problem — n people means n² handshakes. That's
exactly what attention in an LLM does: every word holds a custom conversation
with every other word. Wonderful quality, brutal cost.

## Step 2: The special case — "listen by distance, not by name"

What if the recipe is the *same for everyone* and only depends on
**distance**? Like an echo rule: *"my new number = my number + ½ of my left
neighbor + ¼ of the one two seats left."*

```
weights by distance:  w = [1, 0.5, 0.25]

new₃ = 1·(8) + 0.5·(2) + 0.25·(5) = 10.25
new₄ = 1·(3) + 0.5·(8) + 0.25·(2) = 7.5
```

This "same rule slid along the row" is called a **convolution**. You already
know one intimately: **long multiplication of numbers is a convolution of
their digits** — when you compute 123 × 456 by hand, each digit of one number
gets multiplied against each digit of the other and results slide-and-add into
columns. Done naively it's still n² work (every digit × every digit). So even
with the simpler rule, the obvious method stays expensive.

## Step 3: The trick you already know — logarithms

Suppose you had to compute **37 × 42** but multiplication was expensive and
only addition was cheap. Old-school engineers (with slide rules!) did this:

```
1. Transform:      log(37) = 1.568,  log(42) = 1.623
2. Cheap step:     1.568 + 1.623 = 3.191
3. Transform back: 10^3.191 ≈ 1554  ✓ (37 × 42 = 1554)
```

They **changed representation** — hopped into "log world," where the hard
operation (multiply) becomes an easy one (add), did the easy thing, and hopped
back. The transforms cost something, but far less than what they saved.

**The Fourier transform is the exact same trick, one level up:**

| | transform in | hard becomes easy | transform back |
|---|---|---|---|
| slide rule | logarithm | multiply → **add** | power of 10 |
| spectral mixing | Fourier (FFT) | convolution → **multiply pointwise** | inverse FFT |

## Step 4: What "Fourier" means — the equalizer

The Fourier transform says: any signal — a sound, a list of numbers — can be
rewritten as a **recipe of waves**: so much slow wave + so much medium wave +
so much fast wave. Your music app's *equalizer* is literally this: bass
slider, mid slider, treble slider.

And here's the theorem (the *convolution theorem*): applying an
echo/convolution to a song is exactly the same as **turning each frequency's
volume knob**. So instead of the slide-and-add grind:

```
1. FFT the signal into its wave recipe        (cheap-ish)
2. Multiply each wave's volume by one number  (one multiply each!)
3. Inverse FFT back                           (cheap-ish)
```

Step 2 is the miracle: what was n × n sliding multiplications becomes just
**n knob turns** — because in wave-world, a convolution is a *diagonal*
operation: each frequency minds its own business, no crosstalk.

## Step 5: Count the cost

The FFT itself costs about **n·log₂(n)** operations (a clever halving trick,
like binary search's cousin). Compare:

| n (tokens) | matmul mixing: n² | spectral: n·log₂n | savings |
|---:|---:|---:|---:|
| 1,000 | 1,000,000 | ~10,000 | 100× |
| 100,000 | 10,000,000,000 | ~1,700,000 | ~6,000× |
| 1,000,000 | 1,000,000,000,000 | ~20,000,000 | ~50,000× |

Notice the pattern: the longer the document, the more ridiculous the gap. At
short lengths the trick barely matters (the two transforms have overhead); at
a million tokens it's the difference between possible and impossible.

## Step 6: The honest fine print — what the trick can't do

The condition in Step 2: the rule must be the *same for everyone, based only
on distance*. But real attention sometimes needs a **by-name** conversation:
*"find the word 'password', wherever it is, and listen to it hard."* That's
content-based, not distance-based — and no frequency knob can express it, just
like no equalizer setting can make your speaker play a different song.

So the practical design (this is our hybrid): let each word talk **directly to
its 500 nearest neighbors** the expensive exact way (cheap, because 500 ≪ n),
and handle all the **far-away, distance-flavored listening over the radio** —
FFT, knobs, inverse FFT. Everyone gets local conversations plus a global
broadcast, and nobody pays for a trillion handshakes.

## The one-sentence summary

**Matmul mixing is n² handshakes with custom rules; spectral mixing notices
that most of those rules only depend on distance, hops into wave-world where
distance-rules become independent volume knobs (the same "transform → easy
step → transform back" trick as logarithms), pays only n·log n for the round
trip — and keeps a few real handshakes for the by-name conversations the radio
can't do.**
