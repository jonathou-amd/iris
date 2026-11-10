# PR 544 fused and concurrent capture notes

This branch is `pr-544` plus these notes. The code that supports both modes is `iris/concurrent/`. The captures already on disk were not edited after roccap wrote them.

This is not the Gluon fused all-gather GEMM (`iris/ops/gluon/all_gather_matmul.py`). That kernel pulls peer A inside a GEMM. This workload runs an independent local GEMM at the same time as an all-gather of a separate tensor.

## What the two modes are

`iris.concurrent.gemm.all_gather` takes `mode="fused"` or `mode="concurrent"`. The tile math is the same. The scheduling is not.

**Fused** is one kernel, `fused_ws_gemm_all_gather`. One launch of `num_wgs` persistent workgroups and two atomic queues. Workgroups `0 .. gemm_wgs-1` start on GEMM. The rest start on the all-gather. When a workgroup's home queue is empty it steals from the other queue. `GEMM_WGS` is the home split, not a second grid.

**Concurrent** is two kernels on two streams. `ws_gemm` is launched with `gemm_wgs` workgroups. `ws_all_gather` is launched with `comm_wgs` workgroups. Each kernel only consumes its own queue. The hardware scheduler shares the CUs. The host waits on both streams before returning.

The all-gather tile loads this rank's `comm_src` and publishes it through `DeviceContext.all_gather` into rows `[rank*Mc : (rank+1)*Mc]` of every rank's `comm_dst`. The GEMM is `C = A @ B` and stays local. GEMM does not take the device context. Only the all-gather does.

## How the recorded caps were generated

Host tooling, not in this repo:

- `/home/jonathou/concurrent/capture.sh`
- `/home/jonathou/concurrent/capture_workload.py`

Image `mkmhub.amd.com/sw-irisx-dev/gfx1260-ffm:latest`. One `docker run` per mode. Inside the container the command sources `/ffm/roccap_env.sh`, sets `IRIS_SIMULATION=1`, and points `PYTHONPATH` at this checkout and `PYTHONUSERBASE` at `/home/jonathou/concurrent/.pyuser`. `IRIS_SIMULATION=1` forces the torch allocator. `torchrun --nproc_per_node=2` runs `scripts/roccap_wrapper.py`.

`roccap_wrapper.py` sets `IRIS_HEAP_BASES_PREFIX` to the `-k` filter and writes `{filter}_rank_{RANK}.cap`. A nonzero torchrun status is still success when both the cap and the heap JSON exist. roccap often SIGSEGVs during teardown after the files are written.

Defaults in `capture.sh`, which are the caps on disk:

| Setting | Value |
|---|---|
| Ranks | 2 |
| GEMM | fp16 `512×256×512`, tile `256×64×64`, 128 workgroups |
| All-gather | fp16 `2048×2048` per rank, tile `256×64`, 128 workgroups |
| Fused grid | `num_wgs = 256` (`gemm_wgs + comm_wgs`) |
| Stages / warps | 2 stages, 8 warps |
| Heap | 1 GiB (`capture_workload.py` default; the shell script does not pass `--heap-size`) |

Tile counts at those sizes: GEMM has `(512/256)*(256/64) = 8` tiles. The all-gather has `(2048/256)*(2048/64) = 256` tiles.

Regenerate from `/home/jonathou/concurrent`:

```bash
./capture.sh
```

The script skips a mode when both rank caps already exist. Remove those files first to recapture. `NPROC`, `GEMM_M`, `GEMM_N`, `GEMM_K`, `COMM_M`, `COMM_N`, `GEMM_WGS`, and `COMM_WGS` override the defaults.

Filters passed to `-k`:

- fused: `fused_ws_gemm_all_gather`
- concurrent: `ws_gemm|ws_all_gather`

The concurrent filter is a regex so one capture contains both dispatches. The raw file name uses that string, including the `|`. `capture.sh` renames it afterward. Do not pass a descriptive stem as `-k`. The filter has to match the kernel name.

## Cap file names

Directory: `/home/jonathou/concurrent/captures/`.

```
{mode}_gemm_{M}x{N}x{K}_{gemm_wgs}wgp_allgather_{Mc}x{Nc}_{comm_wgs}wgp_{nproc}nproc_rank_{rank}.cap
```

The four files from 19 Oct 2026:

- `fused_gemm_512x256x512_128wgp_allgather_2048x2048_128wgp_2nproc_rank_0.cap`
- `fused_gemm_512x256x512_128wgp_allgather_2048x2048_128wgp_2nproc_rank_1.cap`
- `concurrent_gemm_512x256x512_128wgp_allgather_2048x2048_128wgp_2nproc_rank_0.cap`
- `concurrent_gemm_512x256x512_128wgp_allgather_2048x2048_128wgp_2nproc_rank_1.cap`

Each has a sibling `_heap_bases.json`. The fused rank-0 cap has one dispatch, `fused_ws_gemm_all_gather`. The concurrent rank-0 cap has two, `ws_gemm` then `ws_all_gather`.

`AQLPLAY(...)` names must not already end in `_cap`. The runner appends `_cap`.

## Heap bases

`Iris.__init__` writes the JSON immediately after `SymmetricHeap` is constructed, before any user tensor is allocated. The file is:

```json
{ "rank": 0, "num_ranks": 2, "heap_bases": ["0x...", "0x..."] }
```

Rank 0 of the fused cap recorded `0x7c0c00000` and `0x801000000`.

The all-gather does not take that live heap-bases tensor. It takes `ctx.get_device_context()`. That tensor is built once at the end of `Iris.__init__`, layout `[cur_rank, world_size, heap_base_0, heap_base_1, ...]`, and `DeviceContext.initialize` reads the bases at offset 2. Later `SymmetricHeap.allocate()` calls `refresh_peer_access()` and updates the live `heap_bases` tensor. It does not rebuild the device context. The JSON and the device context are the same init-time snapshot. They are not the post-allocation table.

That is the opposite of the Gluon all-gather GEMM, whose kernarg 0 is the live table and whose init-time JSON was stale. Do not run `insert_iris_peer_allocs.py` against these caps. That script rewrites peer packets to the live kernarg table, which these kernels do not read.

## Cap file modifications

None. The four caps are the roccap output after the rename in `capture.sh`. No peer packets were inserted, and no heap JSON was rewritten.

The fused rank-0 cap has 15 `AllocPacket`s. Their `globalFlags` are 2, 3, and 4. There is no `globalFlags == 8` peer allocation. The same is true of the generation path: nothing after capture adds one.

These caps have not been replay-checked. If a replay faults on a peer address, the address to map is the one inside the device-context tensor in the cap, which should match the heap JSON, not a later `heap_bases` value from a fresh process.

## Code map

- `iris/concurrent/gemm.py`: host launch. Fused calls `fused_ws_gemm_all_gather[(num_wgs,)]`. Concurrent calls `ws_gemm[(gemm_wgs,)]` and `ws_all_gather[(comm_wgs,)]` on separate streams.
- `iris/concurrent/_kernels.py`: the three kernels and the work-stealing loops.
- `iris/concurrent/_tiles.py`: `_gemm_tile` and `_all_gather_tile`. The all-gather tile is the one that constructs `DeviceContext`.
- `iris/host/iris.py`: JSON write in simulation, and `_build_device_context`.
- `iris/mem/triton/context.py`: `DeviceContext.initialize` and `all_gather`.

## Gluon backend (AM-decodable GEMM)

`iris.concurrent.gemm.all_gather(..., backend="gluon")` runs the same three kernels (`ws_gemm`, `ws_all_gather`, `fused_ws_gemm_all_gather`) written in Gluon, in `iris/concurrent/_gluon_kernels.py`. `mode` also accepts `"gemm_only"`, which launches `ws_gemm` alone as a baseline with no all-gather traffic. The default backend is still Triton.

Why: the Triton GEMM tile (`tl.dot`) compiles to `v_wmma_f32_16x16x32_f16` on gfx1260, which AM's SH decoder rejects. The Gluon GEMM uses WMMA version 4 at `[32, 64, 64]` through laneshared (`v_wmma_f32_32x64x64_f16`), which AM models. `tl.dot` has no switch to select that form, so the whole kernel has to be Gluon, including the all-gather tile (a `@triton.jit` kernel cannot call Gluon code).

Constraints:

- fp16 or bf16 only.
- `num_warps` must be 4 (128 threads per workgroup). `LaneSharedLayout` asserts warps == NUM_WAVE_GROUPS (4).
- The GEMM tile M must be a multiple of `32 * num_warps` (128), N a multiple of 64, K a multiple of 64. Shapes that do not divide the tile take a masked path.
- The comm tile (default 8 x 256 elements) must be a multiple of `32 * num_warps` elements.
- Work stealing uses `_grab()`: a lane-0 masked atomic plus a reduction. A scalar `ttgl.atomic_add` is per-thread and hangs FFM.
- The fused kernel inlines each tile function exactly once (one two-stage loop). Two inlined copies of the GEMM tile with the all-M warp layout hang FFM.

Layout and loop (see the kernel for detail): all four warps tile M, the accumulator is converted to a blocked layout for 16-byte C stores with no cache modifier, and the K loop prefetches the next tile before the current WMMA.

Capture driver: `examples/35_concurrent_capture/example.py` (`--backend gluon --mode fused|concurrent|gemm_only --check`). Use `-k fused_ws_gemm_all_gather` for fused, `-k 'ws_gemm|ws_all_gather'` for concurrent, `-k ws_gemm` for gemm_only. `--check` compares C with a CPU matmul. Inputs come from a seeded host RNG because the FFM device RNG returns all ones. aqltoolkit's `-s` sanity pass does not know the new opcode: use `gummi_setup.py --skip-shader-sanity-checks`.
