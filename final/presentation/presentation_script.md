# Presentation script (5 minutes)

For `Insurance_Pricing_Hackathon_2026.pptx`. All ten slides show the same one-page summary, so each
slide below says which panel to point at. The script is 622 words: about 4.4 minutes at a normal
pace and 5.0 at a slow one. The times are where you should be when each slide starts, at the slow pace.

| Slide | Panel | Starts at |
|---:|---|---:|
| 1 | Title | 0:00 |
| 2 | 1. Problem & Data | 0:40 |
| 3 | 2. Approach | 1:11 |
| 4 | 3. AI-Assisted Workflow | 1:35 |
| 5 | 4. Data Bugs Found | 2:13 |
| 6 | 5. Models Used | 2:40 |
| 7 | 6. Guardrails | 3:05 |
| 8 | 7. How the New Market Changed Our Solution | 3:30 |
| 9 | Key Takeaways | 4:26 |
| 10 | Close | 4:46 |

---

## Slide 1: Title (0:00)

*Point at the title.*

Hello. My name is Krishna Kumar Shrestha, and I am representing Actuarial Noob. First, thank you to
the organizers for this wonderful hackathon. Over these two weeks I did a lot of learning, and a lot
of unlearning. Everything you see here, from this script to the slides, was made with AI, Claude to
be precise, with a lot of prompting from me. Today I will show how I priced a developed market in
Week 1, and then an emerging one in Week 2.

## Slide 2: Problem and data (0:40)

*Point at panel 1.*

The task was the same both weeks: predict the 2025 claim cost of every motor, buildings and contents
policy, scored by Tweedie deviance D squared. Week 1 had four years of data and 163,150 claims. Week
2 had only two years and 9,258 claims. My Week 1 file scored 0.13830 on the private leaderboard, and
my best Week 2 public score is 0.17172.

## Slide 3: Approach (1:11)

*Point at panel 2, top to bottom.*

I followed four steps. Understand the data: read it, clean it, keep every row, log every change.
Build features, where each new column had to beat twenty random noise columns. Develop models, and
test them on years they had not seen. Then check the final file before it goes in.

## Slide 4: AI-assisted workflow (1:35)

*Point at panel 3.*

I worked alone, with an AI coding agent, Claude Code, at every step. Most of the work was a long
series of prompts, each one testing a single idea. Every prompt came with context:
the case study, the question, and a file called KNOWLEDGE.md that records what was already tried and
why it failed. So the AI model never changed, but its memory did. I questioned each result
separately, and nothing went in until I had checked the evidence myself.

## Slide 5: Data bugs found (2:13)

*Point at panel 4.*

Cleaning paid off. I found customer IDs that had lost their leading zeros, exposure recorded in days,
sums insured in units instead of thousands, a reserve typed with an extra digit, and 604 claims
stored twice, which had inflated 2022 claim counts by 1.4 percent. I dropped no rows and imputed
nothing, and every fix is logged.

## Slide 6: Models used (2:40)

*Point at panel 5.*

For Week 1 I built six experts per line: a Tweedie GBM as the baseline, a GLM, one GBM per peril,
and three frequency times claim-size models. Their geometric mean, recalibrated per line, beat every
single model in both test years: 0.1386 and 0.1375, against 0.1364 and 0.1338 for one LightGBM.

## Slide 7: Guardrails (3:05)

*Point at panel 6.*

Guardrails kept me honest. Leakage and governance columns, like the old written premium, came out
first. Large losses were capped and loaded back. A permutation test showed that most factor searches on two years of data were finding noise. Only
gender survived. Every file is versioned, with its evidence and a checksum.

## Slide 8: How the new market changed my solution (3:30)

*Point at panel 7. This is the most important slide, so slow down here.*

Two years of data were not enough to learn a pricing model from
scratch, so I started from what the developed market already knew. I fitted a model on all the Week
1 data, using only factors that mean the same in both markets, and put it in the offset. The local
models then only learn how this market differs. For buildings, that lifted the forward-test score from
0.012 to 0.075. Simpler models also won: tiny trees and fewer experts. One more lesson:
77 percent of 2025 is new business, so my back-tests could check the ranking but not the level of
each segment. I set those levels from the public scores, with a model of their noise.

## Slide 9: Key takeaways (4:26)

*Point at Key Takeaways.*

So, my takeaways. Clean data and a clear process make better models. AI helps you work smarter, but
it does not replace domain knowledge. Guardrails build trust. And a new market is not a new problem,
just a new perspective.

## Slide 10: Close (4:46)

*Point at the bottom line: "Same goal. New data. Smarter decisions."*

Everything is reproducible: one command, final pipeline dot py, rebuilds both submitted files from
the repository. Thank you again to the organizers, and thank you for listening.

---

## Before recording

* Two numbers on the slide image are wrong. Panel 1 shows 629,063 rows to price in Week 2; the
  real number is 209,063. Panel 4 shows 895 Week 2 exposure fixes; the real number is 595. The
  script avoids both, but fix the image if you can.
* If you run long, shorten slide 4 first (drop the sentence about questioning each result), then
  slide 7 (drop the last sentence).
* Say "D squared" for D². Read "0.13830" as "zero point one three eight three zero".
