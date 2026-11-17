# Trident

**A theoretical memory component for AI inference, plus a simulator to test it.**

Modern AI inference is mostly limited by moving data, not by doing math. When a model generates a reply token by token, it has to read basically all of its weights for every single token, and most of the time and energy goes into shuffling those bytes between memory and the processor. Trident is my attempt at a component that attacks that bottleneck directly: give each kind of data the memory that suits it, and put small processors inside the memory so the weights barely have to move at all.

This repo has the design and an analytical simulator (`trident.py`) that compares it against a GPU-with-HBM baseline and a regular DDR5 baseline.

> ⚠️ **All hardware numbers in the simulator are placeholders.** Results show directions and tradeoffs, not real-world performance. Calibration against published numbers is the next step (see [Roadmap](#roadmap)).

---

## The design

```
  ┌──────────────────────┐                 ┌──────────────────────────────┐
  │  stacked SRAM        │                 │  dense NVM layers  (weights) │
  │  (activations,       │                 ├──────────────────────────────┤
  │   newest KV cache)   │                 │  MRAM layer  (older KV cache │
  ├──────────────────────┤◄═══ the fork ══►│   + saved conversations)     │
  │  main core           │                 ├──────────────────────────────┤
  │                      │                 │  logic base layer            │
  └──────────────────────┘                 │  (N small "far units")       │
                                           └──────────────────────────────┘
        main core die                              memory stack
              └────────────── 2.5D interposer ──────────────┘
```

Each memory tier holds the data whose **write pattern** fits it:

| Tier | Holds | Why |
|---|---|---|
| **Stacked SRAM** | activations + newest KV cache | written constantly, needs max speed |
| **MRAM** | older KV cache + persistent prompt/conversation cache | written every token; MRAM handles frequent writes well and keeps data with the power off |
| **Dense NVM** | model weights only | read-only during inference, so dense memory's slow, wear-prone writes never matter |

**Far units.** The logic base layer of the memory stack has many small int8 matrix-vector units, roughly one per bank group. They do the weight math right where the weights live, so only small activation vectors cross the fork.

**Controller.** For every weight matrix, the controller splits the rows between the far units and the main core so both finish at the same moment. With one user almost everything goes to the far units; with many users more work shifts to the main core.

**Scope.** Inference only, focused on decode (generating tokens). Prefill runs on the main core. Training is out of scope, since it rewrites weights constantly and that's exactly what dense NVM is bad at.

**Showcase workload: MoE models.** All experts sit cheaply in the dense prong, and the far units compute *after* the router picks experts, so the design doesn't need expert prediction at all.

### Where it would be used
Data centers, most likely paired with GPUs: GPUs handle prefill (math-heavy), Trident handles decode (memory-heavy). Splitting those two phases across different hardware is already a trend in AI serving.

---

## Quick start

Needs Python 3, nothing else.

```bash
python3 trident.py                      # 7B model, 1 user
python3 trident.py --model mixtral      # MoE model
python3 trident.py --sweep batch --values 1,4,16,64
```

The script only prints results. It doesn't write files or change anything, and the output is deterministic, so the same command always gives the same numbers unless you edit the hardware specs.

---

## Usage

| Flag | Default | What it does |
|---|---|---|
| `--model` | `7b` | `7b`, `13b`, `70b`, `mixtral`, `moe-large` |
| `--bytes` | `2` | bytes per weight (2 = fp16, 1 = int8, 0.5 = int4) |
| `--kv-bytes` | `1` | bytes per KV cache value (1 = int8, 2 = fp16), applied to every system |
| `--batch` | `1` | users served at once |
| `--prompt` | `512` | context length already in the conversation |
| `--gen` | `128` | tokens to generate |
| `--units` | `32` | number of far units |
| `--acc` | `90` | MoE expert prediction accuracy (%), only affects the streaming path |
| `--sweep` | none | `batch`, `ctx`, `units`, or `acc` |
| `--values` | `1,4,16,64` | values for the sweep |

Some useful runs:

```bash
python3 trident.py --sweep units --values 4,8,16,32,64          # how many far units do you need?
python3 trident.py --sweep ctx --values 512,4096,16384,65536    # long-context behavior
python3 trident.py --model mixtral --sweep acc --values 50,70,90,99
python3 trident.py --prompt 32000                               # resume advantage on long conversations
python3 trident.py --model moe-large --bytes 1                  # a model too big for one GPU
```

### Systems compared

| System | What it is |
|---|---|
| `hbm_gpu` | GPU-style baseline, everything in HBM. Scales to several GPUs if the model doesn't fit (inter-GPU communication is **not** modeled, which flatters the baseline) |
| `ddr_cpu` | conventional DDR5 baseline |
| `tri_stream` | all weight math on the main core, weights streamed over the fork with prefetch |
| `tri_ndp` | all weight math on the far units, in place |
| `tri_adapt` | per layer, picks stream *or* ndp |
| `tri_split` | rows split between far units and main core so both finish together (the main design) |

### Output columns

- **tok/s**: tokens generated per second, across all users
- **ms/step**: time to generate one token for every user in the batch
- **mJ/token**: energy per generated token
- **fork GB/step**: data crossing the link between main core and memory stack
- **cold start**: time before the model can run after power-on
- **resume**: time to continue a saved conversation

---

## Results so far

Placeholder hardware, so read these as directions only.

- **7B, 1 user:** `tri_split` is about **2.8× faster** than `hbm_gpu` and uses about **1.9× less energy** per token.
- **Far units:** the design needs roughly **16 or more** to clearly beat the GPU. At 4 it loses.
- **Mixtral (MoE):** `tri_split` beats a **2-GPU** baseline at every batch size from 1 to 64 (about 1.3× at 64 users).
- **Expert prediction:** only the streaming path cares about it. `tri_ndp` gives the same speed at 50% and 99% accuracy.
- **Resume:** continuing a 32K-token conversation takes about **1.2 s** on the GPU and about **0** on Trident, since the KV cache never left MRAM.
- **Cold start:** about **0 s**, because weights already sit in nonvolatile memory. The GPU baseline has to load them from an SSD (around 2 s for 7B, around 10 s for 70B).

### Known weakness: long context
At 64K tokens of context, the KV cache spills from SRAM into MRAM and every read crosses the fork. Trident drops to roughly a tie with the GPU (104 vs 111 tok/s). Fixing this is on the roadmap.

---

## What the simulator doesn't model (yet)

- prefill throughput
- bank conflicts and queueing
- expert load imbalance across far units
- thermal limits
- communication between GPUs in the multi-GPU baseline
- embeddings and the LM head

It's an analytical (roofline-style) model, not cycle-accurate. That's intentional for this stage, since it's fast to iterate on. If the design keeps looking good, the plan is to validate it in an established C++ simulator.

---

## Roadmap

1. **Calibrate the baseline** against published real tokens/sec (7B and Mixtral on an H100)
2. **Replace placeholder device numbers** with cited figures (datasheets, ISSCC/VLSI papers)
3. **Fix long context**: try attention on the far units, or a faster MRAM-to-main-core path
4. **Model prefill**
5. **Charts**: tok/s and energy vs batch size
6. **Prior-art review** (see below) to pin down exactly what's new
7. **Validate in Ramulator 2 or gem5** if the analytical results hold up

---

## Related work to check

Pieces of this exist in research already. The goal is to be precise about what's different here.

- Samsung HBM-PIM, SK hynix AiM (processing-in-memory products)
- AttAcc and NeuPIMs (ASPLOS 2024), PIM for LLM inference
- Apple, *LLM in a Flash* (running LLMs from flash memory)
- MoE expert-prefetching papers
- Hybrid SRAM/MRAM caches, heterogeneous memory tiering

The parts most likely to be novel: the per-matrix row split between far units and main core, assigning memory tiers by write pattern (weights to dense, KV to MRAM), and the persistent conversation cache.

---

## Files

| File | What it is |
|---|---|
| `trident.py` | the simulator |
| `CLAUDE.md` | project context for Claude Code |
| `README.md` | this file |