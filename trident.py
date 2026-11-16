#!/usr/bin/env python3
"""
trident_sim.py v2 - simulator for the trident memory component on LLM inference (decode)

Layout (v2)
  [main core + stacked SRAM]  ==fork link (2.5D interposer)==  [memory stack]
                                                                 dense NVM layers -> weights only
                                                                 MRAM layer       -> KV cache tier 2 + saved prompts
                                                                 logic base layer -> N small far units (one per bank group)

Each memory holds the data whose write pattern fits it
  SRAM  : activations + newest KV cache (written constantly)
  MRAM  : older KV cache + persistent prompt cache (written every token, survives power-off)
  dense : model weights (read-only during inference)

Systems compared
  hbm_gpu     : GPU-style baseline, scales to several GPUs if the model doesn't fit
  ddr_cpu     : conventional DDR5 baseline
  tri_stream  : all weight math on the main core, weights streamed over the fork with prefetch
  tri_ndp     : all weight math on the far units, in place inside the dense stack
  tri_adapt   : per layer, pick stream OR ndp (v1 behavior)
  tri_split   : rows of every weight matrix split between far units and main core so both finish together

New in v2
  - many far units: compute and internal bandwidth scale with --units
  - row splitting (tri_split)
  - KV tiering SRAM -> MRAM -> dense, int8 KV by default (applied to baselines too, for fairness)
  - MoE models: only routed experts get read. The streaming path needs expert prediction (--acc)
    to prefetch; far units compute in place after routing, so they don't need prediction at all
  - "resume" column: time to continue a saved conversation
    (baseline recomputes the prompt or reloads KV from SSD, trident keeps KV in persistent MRAM)
  - v1 policies naive/lru dropped (v1 already showed LRU gets ~0% hits on cyclic weight reads)

!!! ALL DEVICE NUMBERS IN default_hardware() ARE PLACEHOLDERS - replace with cited figures !!!
Not modeled: prefill throughput, bank conflicts/queueing, expert load imbalance across far units,
thermal limits, multi-GPU communication (this favors the baseline), embeddings / LM head.

Usage
  python3 trident_sim.py
  python3 trident_sim.py --model mixtral
  python3 trident_sim.py --model moe-large --bytes 1
  python3 trident_sim.py --sweep batch --values 1,4,16,64
  python3 trident_sim.py --sweep ctx   --values 512,4096,16384,65536
  python3 trident_sim.py --sweep units --values 4,8,16,32,64
  python3 trident_sim.py --model mixtral --sweep acc --values 50,70,90,99
"""
import argparse
import math
from dataclasses import dataclass

GB = 1e9


# ---------------------------------------------------------------- hardware descriptions

@dataclass
class Mem:
    name: str
    capacity_gb: float
    bw_gbs: float          # bandwidth inside the device (GB/s)
    latency_ns: float      # first-byte latency per request
    read_pj_bit: float
    write_pj_bit: float
    persistent: bool = False


@dataclass
class Core:
    name: str
    tflops: float
    pj_per_flop: float


@dataclass
class Trident:
    fast: Mem               # stacked SRAM on the main core
    persist: Mem            # MRAM layer in the memory stack
    dense: Mem              # dense NVM layers in the memory stack
    fork_bw_gbs: float      # main core <-> memory stack link
    fork_pj_bit: float      # extra energy per bit to cross that link
    main: Core
    n_units: int            # far units on the logic base layer
    unit_tflops: float
    unit_pj_per_flop: float
    offload_us: float       # fixed cost to hand a job to the far units and get the answer back
    fast_reserve_gb: float  # SRAM kept for activations / stream buffers, not KV
    controller_state_mb: float

    @property
    def far_tflops(self):
        return self.n_units * self.unit_tflops


def default_hardware(units=32):
    """PLACEHOLDER numbers. Swap in real, cited values."""
    main = Core("main core", tflops=300, pj_per_flop=0.5)
    hbm = Mem("HBM3 GPU", capacity_gb=80, bw_gbs=3350, latency_ns=100, read_pj_bit=4.0, write_pj_bit=4.0)
    ddr = Mem("DDR5", capacity_gb=256, bw_gbs=100, latency_ns=90, read_pj_bit=15.0, write_pj_bit=15.0)
    unit_bw = 250  # GB/s each far unit pulls from its own bank group
    tri = Trident(
        fast=Mem("stacked SRAM", capacity_gb=1.0, bw_gbs=8000, latency_ns=5,
                 read_pj_bit=0.5, write_pj_bit=0.5),
        persist=Mem("MRAM", capacity_gb=8.0, bw_gbs=2000, latency_ns=10,
                    read_pj_bit=1.0, write_pj_bit=5.0, persistent=True),
        dense=Mem("3D dense NVM", capacity_gb=512, bw_gbs=units * unit_bw, latency_ns=200,
                  read_pj_bit=2.0, write_pj_bit=20.0, persistent=True),
        fork_bw_gbs=2000, fork_pj_bit=1.0,
        main=main,
        n_units=units, unit_tflops=4.0, unit_pj_per_flop=1.0,
        offload_us=1.0, fast_reserve_gb=0.25, controller_state_mb=64,
    )
    ssd_gbs = 7.0
    return main, hbm, ddr, tri, ssd_gbs


# ---------------------------------------------------------------- model + workload

@dataclass
class Model:
    name: str
    n_layers: int
    d_model: int
    d_ff: int               # per expert for MoE models
    n_heads: int
    n_kv_heads: int
    n_experts: int = 1
    top_k: int = 1
    bytes_per_param: float = 2.0   # weights: 2 = fp16, 1 = int8, 0.5 = int4
    kv_bytes: float = 1.0          # KV cache: 1 = int8 (default), 2 = fp16

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def is_moe(self):
        return self.n_experts > 1

    def attn_bytes(self):
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        return (2 * d * d + 2 * d * kv) * self.bytes_per_param  # Wq, Wo, Wk, Wv

    def expert_bytes(self):
        return 3 * self.d_model * self.d_ff * self.bytes_per_param  # gate, up, down

    def unique_experts(self, batch):
        """expected number of distinct experts touched in one layer by a batch"""
        if not self.is_moe:
            return 1.0
        e, k = self.n_experts, self.top_k
        return e * (1 - (1 - k / e) ** batch)

    def mlp_read_bytes(self, batch):
        return self.unique_experts(batch) * self.expert_bytes()

    def mlp_flops(self, batch):
        return 2 * self.top_k * (self.expert_bytes() / self.bytes_per_param) * batch

    def attn_proj_flops(self, batch):
        return 2 * (self.attn_bytes() / self.bytes_per_param) * batch

    def attn_flops(self, ctx, batch):
        return 4 * ctx * self.d_model * batch  # QK^T and attn*V

    def kv_per_token_layer(self):
        return 2 * self.n_kv_heads * self.head_dim * self.kv_bytes

    def total_weight_bytes(self):
        return self.n_layers * (self.attn_bytes() + self.n_experts * self.expert_bytes())

    def active_params(self):
        return self.n_layers * (self.attn_bytes() + self.top_k * self.expert_bytes()) / self.bytes_per_param


MODELS = {
    "7b": dict(name="7B dense", n_layers=32, d_model=4096, d_ff=11008, n_heads=32, n_kv_heads=32),
    "13b": dict(name="13B dense", n_layers=40, d_model=5120, d_ff=13824, n_heads=40, n_kv_heads=40),
    "70b": dict(name="70B dense (GQA)", n_layers=80, d_model=8192, d_ff=28672, n_heads=64, n_kv_heads=8),
    "mixtral": dict(name="Mixtral-class MoE (8 experts, top-2)", n_layers=32, d_model=4096, d_ff=14336,
                    n_heads=32, n_kv_heads=8, n_experts=8, top_k=2),
    "moe-large": dict(name="large MoE (128 experts, top-8)", n_layers=48, d_model=6144, d_ff=2048,
                      n_heads=48, n_kv_heads=8, n_experts=128, top_k=8),
}


@dataclass
class Workload:
    batch: int = 1
    prompt_len: int = 512
    gen_tokens: int = 128


def kv_total(m, wl, ctx):
    return m.n_layers * ctx * m.kv_per_token_layer() * wl.batch


# ---------------------------------------------------------------- cost helpers

def xfer(mem, nbytes, write=False, link_bw=None, link_pj=0.0):
    """time (s) and energy (J) to move nbytes in/out of mem, optionally through a narrower link"""
    if nbytes <= 0:
        return 0.0, 0.0
    bw = mem.bw_gbs if link_bw is None else min(mem.bw_gbs, link_bw)
    t = mem.latency_ns * 1e-9 + nbytes / (bw * GB)
    pj = (mem.write_pj_bit if write else mem.read_pj_bit) + link_pj
    return t, nbytes * 8 * pj * 1e-12


def compute(core, flops):
    return flops / (core.tflops * 1e12), flops * core.pj_per_flop * 1e-12


@dataclass
class Stats:
    name: str
    time_s: float = 0.0
    energy_j: float = 0.0
    fork_bytes: float = 0.0
    weight_bytes: float = 0.0
    far_bytes: float = 0.0
    cold_start_s: float = 0.0
    resume_s: float = 0.0
    steps: int = 0
    tokens: int = 0
    note: str = ""

    def add_note(self, text):
        self.note = f"{self.note}; {text}" if self.note else text

    @property
    def tok_s(self):
        return self.tokens / self.time_s if self.time_s else 0.0

    @property
    def ms_step(self):
        return 1e3 * self.time_s / self.steps if self.steps else 0.0

    @property
    def mj_token(self):
        return 1e3 * self.energy_j / self.tokens if self.tokens else 0.0

    @property
    def fork_gb_step(self):
        return self.fork_bytes / self.steps / GB if self.steps else 0.0


def session_kv_bytes(m, wl):
    return m.n_layers * wl.prompt_len * m.kv_per_token_layer()  # one saved conversation


# ---------------------------------------------------------------- baselines (one flat memory)

def sim_flat(name, m, wl, mem, core, ssd_gbs):
    s = Stats(name)
    need = m.total_weight_bytes() + kv_total(m, wl, wl.prompt_len + wl.gen_tokens)
    n = max(1, math.ceil(need / (mem.capacity_gb * GB)))
    bw, flops_peak = n * mem.bw_gbs * GB, n * core.tflops * 1e12
    if n > 1:
        s.add_note(f"needs {n}x {mem.name} (no inter-device comm modeled)")
    for step in range(wl.gen_tokens):
        ctx = wl.prompt_len + step
        for _ in range(m.n_layers):
            kv_r = ctx * m.kv_per_token_layer() * wl.batch
            kv_w = m.kv_per_token_layer() * wl.batch
            w = m.attn_bytes() + m.mlp_read_bytes(wl.batch)
            f = m.attn_proj_flops(wl.batch) + m.mlp_flops(wl.batch) + m.attn_flops(ctx, wl.batch)
            t_mem = mem.latency_ns * 1e-9 + (w + kv_r + kv_w) / bw
            s.time_s += max(t_mem, f / flops_peak)  # streaming overlapped with math
            s.energy_j += ((w + kv_r) * 8 * mem.read_pj_bit + kv_w * 8 * mem.write_pj_bit) * 1e-12
            s.energy_j += f * core.pj_per_flop * 1e-12
            s.weight_bytes += w
    s.steps, s.tokens = wl.gen_tokens, wl.gen_tokens * wl.batch
    s.cold_start_s = m.total_weight_bytes() / (n * ssd_gbs * GB)  # each device loads its share in parallel
    p = wl.prompt_len
    prefill_flops = 2 * m.active_params() * p + m.n_layers * 2 * p * p * m.d_model
    s.resume_s = min(prefill_flops / flops_peak, session_kv_bytes(m, wl) / (ssd_gbs * GB))
    return s


# ---------------------------------------------------------------- trident pieces

def kv_tiers(hw, kv_all):
    """fractions of the KV cache in SRAM / MRAM / dense, plus where the oldest entry gets demoted"""
    if kv_all <= 0:
        return 1.0, 0.0, 0.0, None
    sram_cap = max(0.0, (hw.fast.capacity_gb - hw.fast_reserve_gb) * GB)
    mram_cap = hw.persist.capacity_gb * GB
    s = min(kv_all, sram_cap)
    mm = min(kv_all - s, mram_cap)
    d = kv_all - s - mm
    demote = None if kv_all <= sram_cap else ("mram" if d == 0 else "dense")
    return s / kv_all, mm / kv_all, d / kv_all, demote


def attention_phase(hw, m, wl, ctx, tiers):
    """attention on the main core; KV read from SRAM (local) and MRAM/dense (over the fork)"""
    fs, fm, fd, demote = tiers
    kv_r = ctx * m.kv_per_token_layer() * wl.batch
    kv_w = m.kv_per_token_layer() * wl.batch
    fork = dict(link_bw=hw.fork_bw_gbs, link_pj=hw.fork_pj_bit)
    t_s, e_s = xfer(hw.fast, kv_r * fs)
    t_m, e_m = xfer(hw.persist, kv_r * fm, **fork)
    t_d, e_d = xfer(hw.dense, kv_r * fd, **fork)
    t_w, e_w = xfer(hw.fast, kv_w, write=True)  # newest KV always lands in SRAM
    e_dem, fb_dem = 0.0, 0.0
    if demote == "mram":
        _, e_dem = xfer(hw.persist, kv_w, write=True, **fork)
        fb_dem = kv_w
    elif demote == "dense":
        _, e_dem = xfer(hw.dense, kv_w, write=True, **fork)
        fb_dem = kv_w
    t_c, e_c = compute(hw.main, m.attn_flops(ctx, wl.batch))
    t = max(t_s + t_w, t_m + t_d, t_c)  # local path, fork path and math overlap
    return t, e_s + e_m + e_d + e_w + e_dem + e_c, kv_r * (fm + fd) + fb_dem


def weight_block(hw, m, wl, w, f, acc, policy):
    """one weight matrix-vector job (attention projections or MLP/experts).
    returns time, energy, fork bytes, bytes handled by far units"""
    fork_bw = min(hw.fork_bw_gbs, hw.dense.bw_gbs) * GB
    miss = 1.0 - acc
    # far units: read in place at internal bandwidth, no prediction needed (they run after routing)
    tf = max(w / (hw.dense.bw_gbs * GB), f / (hw.far_tflops * 1e12))
    ef = (w * 8 * hw.dense.read_pj_bit + f * hw.unit_pj_per_flop) * 1e-12
    # main core: weights streamed over the fork with prefetch.
    # mispredicted experts: the wrong prefetch is wasted and the right one stalls on demand
    tm = max(w / fork_bw, f / (hw.main.tflops * 1e12)) + miss * w / fork_bw
    em = (w * (1 + miss) * 8 * (hw.dense.read_pj_bit + hw.fork_pj_bit) + f * hw.main.pj_per_flop) * 1e-12
    act = 2 * wl.batch * m.d_model * 2  # fp16 activations in + results out
    t_act, e_act = xfer(hw.fast, act, link_bw=hw.fork_bw_gbs, link_pj=hw.fork_pj_bit)
    lat = hw.dense.latency_ns * 1e-9
    c = hw.offload_us * 1e-6 + t_act  # fixed cost whenever the far units are used

    opts = {
        "stream": (tm + lat, em, w * (1 + miss), 0.0),
        "ndp": (tf + c + lat, ef + e_act, act, w),
    }
    # split: give far units fraction x of the rows so x*tf + c == (1-x)*tm
    if tf + tm > 0:
        x = min(1.0, max(0.0, (tm - c) / (tf + tm)))
        if 0.0 < x < 1.0:
            opts["split"] = ((1 - x) * tm + lat, x * ef + (1 - x) * em + e_act,
                             (1 - x) * w * (1 + miss) + act, x * w)

    if policy == "stream":
        return opts["stream"]
    if policy == "ndp":
        return opts["ndp"]
    if policy == "adapt":
        return min((opts["stream"], opts["ndp"]), key=lambda o: o[0])
    if policy == "split":
        return min(opts.values(), key=lambda o: o[0])
    raise ValueError(policy)


def sim_trident(policy, m, wl, hw, acc):
    s = Stats(f"tri_{policy}")
    if m.total_weight_bytes() > hw.dense.capacity_gb * GB:
        s.add_note("weights don't fit in dense prong")
    mlp_acc = acc if m.is_moe else 1.0
    for step in range(wl.gen_tokens):
        ctx = wl.prompt_len + step
        tiers = kv_tiers(hw, kv_total(m, wl, ctx))
        for _ in range(m.n_layers):
            jobs = [
                weight_block(hw, m, wl, m.attn_bytes(), m.attn_proj_flops(wl.batch), 1.0, policy),
                attention_phase(hw, m, wl, ctx, tiers) + (None,),
                weight_block(hw, m, wl, m.mlp_read_bytes(wl.batch), m.mlp_flops(wl.batch), mlp_acc, policy),
            ]
            for t, e, fb, far in jobs:
                s.time_s += t
                s.energy_j += e
                s.fork_bytes += fb
                if far is not None:
                    s.far_bytes += far
            s.weight_bytes += m.attn_bytes() + m.mlp_read_bytes(wl.batch)
    s.steps, s.tokens = wl.gen_tokens, wl.gen_tokens * wl.batch
    # weights already sit in the nonvolatile dense prong; only controller state comes back from MRAM
    s.cold_start_s, _ = xfer(hw.persist, hw.controller_state_mb * 1e6)
    # saved conversation: KV already in persistent MRAM, just refill the SRAM-resident part
    sram_cap = (hw.fast.capacity_gb - hw.fast_reserve_gb) * GB
    s.resume_s, _ = xfer(hw.persist, min(session_kv_bytes(m, wl), sram_cap),
                         link_bw=hw.fork_bw_gbs, link_pj=hw.fork_pj_bit)
    if policy in ("adapt", "split", "ndp") and s.weight_bytes:
        s.add_note(f"far units did {100 * s.far_bytes / s.weight_bytes:.0f}% of weight work")
    final = kv_tiers(hw, kv_total(m, wl, wl.prompt_len + wl.gen_tokens))
    if final[1] > 0 or final[2] > 0:
        s.add_note(f"KV {100 * final[0]:.0f}% SRAM / {100 * final[1]:.0f}% MRAM / {100 * final[2]:.0f}% dense")
    return s


# ---------------------------------------------------------------- running + printing

POLICIES = ("stream", "ndp", "adapt", "split")


def run_all(m, wl, units=32, acc=0.9):
    main, hbm, ddr, tri, ssd = default_hardware(units)
    out = [sim_flat("hbm_gpu", m, wl, hbm, main, ssd), sim_flat("ddr_cpu", m, wl, ddr, main, ssd)]
    out += [sim_trident(p, m, wl, tri, acc) for p in POLICIES]
    return out


def print_table(m, wl, units, acc, results):
    print(f"\nmodel: {m.name}")
    print(f"  {m.total_weight_bytes() / GB:.1f} GB weights @ {m.bytes_per_param:g} B/param, "
          f"KV @ {m.kv_bytes:g} B/value, {m.active_params() / 1e9:.1f}B active params/token")
    print(f"workload: batch {wl.batch}, prompt {wl.prompt_len}, generate {wl.gen_tokens} tokens")
    moe = f", expert prediction {100 * acc:.0f}%" if m.is_moe else ""
    print(f"trident: {units} far units{moe}")
    print("(device numbers are placeholders - relative comparisons only)\n")
    hdr = (f"{'system':<12}{'tok/s':>9}{'ms/step':>9}{'mJ/token':>10}{'fork GB/step':>14}"
           f"{'cold start':>12}{'resume':>10}  notes")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<12}{r.tok_s:>9.1f}{r.ms_step:>9.2f}{r.mj_token:>10.1f}{r.fork_gb_step:>14.2f}"
              f"{r.cold_start_s:>11.3f}s{r.resume_s:>9.3f}s  {r.note}")
    base = results[0]
    best = max(results, key=lambda r: r.tok_s)
    green = min(results, key=lambda r: r.mj_token)
    print(f"\nfastest: {best.name} ({best.tok_s / base.tok_s:.2f}x vs hbm_gpu)   "
          f"most efficient: {green.name} ({base.mj_token / green.mj_token:.2f}x less energy vs hbm_gpu)")


def sweep(m, base_wl, units, acc, what, values):
    print(f"\nsweep over {what} - model {m.name}  (each cell: tok/s | mJ/token)\n")
    rows, names = [], []
    for v in values:
        wl = Workload(base_wl.batch, base_wl.prompt_len, base_wl.gen_tokens)
        u, a = units, acc
        if what == "batch":
            wl.batch = int(v)
        elif what == "ctx":
            wl.prompt_len = int(v)
        elif what == "units":
            u = int(v)
        elif what == "acc":
            a = v / 100.0
        res = run_all(m, wl, u, a)
        names = [r.name for r in res]
        rows.append((v, res))
    print(f"{what:>7} " + "".join(f"{n:>17}" for n in names))
    for v, res in rows:
        print(f"{v:>7g} " + "".join(f"{r.tok_s:>9.0f} |{r.mj_token:>5.0f}  " for r in res))


def main():
    ap = argparse.ArgumentParser(description="trident memory component simulator v2 (LLM decode)")
    ap.add_argument("--model", choices=MODELS.keys(), default="7b")
    ap.add_argument("--bytes", type=float, default=2.0, help="bytes per weight (2=fp16, 1=int8, 0.5=int4)")
    ap.add_argument("--kv-bytes", type=float, default=1.0, help="bytes per KV value (1=int8, 2=fp16)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--units", type=int, default=32, help="number of far units")
    ap.add_argument("--acc", type=float, default=90, help="MoE expert prediction accuracy, percent")
    ap.add_argument("--sweep", choices=("batch", "ctx", "units", "acc"))
    ap.add_argument("--values", default="1,4,16,64")
    a = ap.parse_args()

    m = Model(**MODELS[a.model], bytes_per_param=a.bytes, kv_bytes=a.kv_bytes)
    wl = Workload(batch=a.batch, prompt_len=a.prompt, gen_tokens=a.gen)
    acc = a.acc / 100.0
    if a.sweep:
        sweep(m, wl, a.units, acc, a.sweep, [float(x) for x in a.values.split(",")])
    else:
        print_table(m, wl, a.units, acc, run_all(m, wl, a.units, acc))


if __name__ == "__main__":
    main()