# Trident memory component: project context

## What this is
A theoretical computer component for AI **inference** (running trained models, not training them), plus a Python analytical simulator (`trident.py`) that tests it against GPU and CPU baselines. The goal is a credible write-up for PhD applications and faculty conversations, possibly a workshop paper.

## Current design (v2)
- **Main core die with stacked SRAM** (3D V-Cache style). The SRAM holds activations and the newest KV cache.
- **Separate memory stack** beside the main core on a 2.5D interposer, connected by the "fork" link:
  - **Dense NVM layers**: model weights only. Read-only during inference, nonvolatile, so cold start is about 0.
  - **MRAM layer**: older KV cache plus a persistent prompt/conversation cache. Write-friendly and survives power-off, so resuming a conversation is instant.
  - **Logic base layer**: N small int8 "far units", one per bank group, that do weight matrix-vector math in place (near-data processing).
- **Controller**: splits every weight matrix's rows between the far units and the main core so both finish at the same time (`tri_split`). The program declares what must persist; hardware handles placement.
- **Scope**: inference only, decode-focused. Prefill runs on the main core. Not suited to training (too many writes to dense memory).
- **Showcase workload**: MoE models. The far units compute after routing, so near-data processing makes expert prediction unnecessary.

## Simulator (`trident.py`, Python, stdlib only)
- Systems: `hbm_gpu`, `ddr_cpu`, `tri_stream`, `tri_ndp`, `tri_adapt`, `tri_split`
- Models: `7b`, `13b`, `70b`, `mixtral`, `moe-large`
- Flags: `--model --bytes --kv-bytes --batch --prompt --gen --units --acc --sweep {batch,ctx,units,acc} --values`
- **All device numbers in `default_hardware()` are placeholders** and need cited sources.
- Not modeled yet: prefill throughput, bank conflicts/queueing, expert load imbalance, thermal limits, multi-GPU communication (this favors the baseline), embeddings / LM head.

## Results so far (placeholder hardware, directional only)
- 7B, batch 1: `tri_split` about 4.1x faster than `hbm_gpu` (725 vs 179 tok/s), about 1.8x less energy per token (`tri_ndp` 1.9x). Was 2.8x before the GPU bandwidth factor was added.
- Needs about 8 far units to beat the GPU (153 tok/s at 4, 303 at 8).
- Mixtral: `tri_split` beats a 2-GPU baseline at every batch size (1 to 64): 2.0x at batch 1, 1.9x at batch 64.
- 70B dense (2-GPU baseline): `tri_split` about 2.1x faster.
- Large MoE (128 experts, 6-GPU baseline): `hbm_gpu` is faster at batch 1 (374 vs 256 tok/s); Trident uses about 1.7x less energy.
- Weakness: long context. At 64K tokens the KV cache moves to MRAM, crosses the fork, and Trident's lead drops to about 1.3x (104 vs 78 tok/s).
- Resuming a 32K-token conversation: about 1.2 s on the GPU vs about 0 on the trident.
- All GPU-baseline numbers use `bw_eff=0.7` (placeholder judgment, not calibrated).

## Next steps (priority order)
1. Calibrate `hbm_gpu` against published real tokens/sec (7B and Mixtral on an H100). `bw_eff=0.7` is a placeholder until this is done.
2. Replace the placeholder device numbers with cited figures.
3. Fix long context: try attention on the far units, or a faster MRAM-to-main-core path.
4. Add prefill modeling.
5. Add charts (tok/s and energy vs batch size); matplotlib is fine.
6. Prior-art check: Samsung HBM-PIM, SK hynix AiM, AttAcc and NeuPIMs (ASPLOS 2024), Apple's "LLM in a Flash", MoE expert-prefetching papers.
7. Later: validate in Ramulator 2 or gem5 (C++) if the analytical results hold up.

## Working preferences
- Be direct and push back on errors.
- Prefer complete, working outputs over partial drafts.
- Python stays the right language for this analytical stage.