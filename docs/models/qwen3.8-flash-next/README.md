# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) · [ROCm 10.1 status](ROCM.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Eligible
requests draft over the first 98,304 token IDs plus a 1,024-row table containing
special tokens and other tokens already used by the request. The full head
remains the fallback when the anchor or two of the last 32 committed tokens
fall outside this vocabulary.

Single-session sampling, including the default thinking-on and thinking-off
presets, can use this smaller head with the existing acceptance/length
controller and AR backoff. Greedy requests, including penalties and
constraints, can use it in C1, when decoding alone and in jointly planned
batches; drafts ignore both, and verification applies them over the full
vocabulary. Greedy requests in unplanned multi-request batches retain the full
head.

The existing length controller and AR backoff apply to every request. The
target always verifies with its full vocabulary, including
exact sampled acceptance and residual correction. Sampled text can differ
across builds when proposals change. All-greedy C>1 batches retain their shared
planner and may use measured cycle costs. Each request keeps private caches,
rollback and RNG; snapshots preserve the learned acceptance history.
See [MTP qualification](QUALITY.md).

MTP decoding also proposes copies from the context, with one to eight sessions.
The context (committed tokens plus the chain so far) is matched against the
session's own earlier tokens by its last three tokens; when the best of the 16
most recent occurrences matches at least 12 tokens back for greedy requests
or 24 for sampled requests, the tokens that followed it fill the rest of the
chain and drafting stops. Greedy requests can
bypass the neural head when a match supplies at least three copies. Sampled
requests first make a neural proposal, then look for a copy extending it;
a repeated prefix alone can poorly predict new code or field values.
Their copy sources end at the request's starting context. Earlier parts of
the same new answer are not indexed for sampled copying; they become eligible
context on the next request. Greedy requests can also copy their current answer.
The request's first speculative step skips lookup to avoid delaying its first
token and records the start of generated text. Later matches cannot extend
back into the prompt's chat wrapper; a common tool header alone must not
qualify as a long matching answer. Copies
are verified like any draft: greedy text is byte-identical, and sampled
requests treat each copied token as a point-mass proposal under the same
rejection sampling, so outputs keep the target distribution (a seeded sampled
stream differs from one without copies, because random draws are consumed at
different points). Shorter matches
propose nothing; at temperature 1 an ungated 3-token rule displaced accepted
MTP drafts and was slower on every workload measured. A recently accepted
passage can resume after an edit with a five-token match; fewer than half of
the copied tokens being accepted disables that shorter gate, including after
a full match. Copies stop at EOS when the request enables it.
The index is private to the session and rebuilt after reset or snapshot
restore. Snapshots retain the request boundary and copy-resume cursor.
Lookup cycles and their following catch-up are excluded from the
batch planner's neural cost measurements. The planner measures short neural
chains before probing longer ones after copying. In all-greedy batches mixing
copies with new text, copied suffixes are bounded to half the verifier or the
planner's chosen width, whichever is larger, to limit pauses for other requests.
All-copy batches retain full width; sampled proposal streams are unchanged.
Timings report
`lookup_n` / `lookup_n_accepted`.

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144. A larger
context needs static YaRN and `--enable-yarn`: `gufo serve llm` refuses to
start with `--context` above 262144 without it, and `gufo bench` refuses a
workload whose depth plus prompt or generation tokens exceeds 262144. With
the flag, the factor is context / 262144 (`--context 409600` gives 1.5625).
The flag is an error at or below 262144, where it would have no effect.
Static YaRN applies to every request on that server, including short ones,
and changes the output distribution more as the factor grows; in the first
measurements, multiple-choice accuracy moved within noise at factors 1.5625
to 3.875. See the
[measurements](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323981940),
the
[opt-in proposal](https://github.com/gufo-org/gufo/pull/350#issuecomment-6323982386)
and the
[accuracy check](https://github.com/gufo-org/gufo/pull/350#issuecomment-6234093998).
No GGUF key enables YaRN.
Prefill chunks whose visible context exceeds 262144 tokens use the per-token
attention kernel rather than the fused WMMA kernel. Memory grows with used
context and selected rollback depth (about 27.5 KiB per context token with
MTP, measured on gfx1151); admission reserves the configured capacity before
creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.

## Expert access trace

Set `GUFO_EXPERT_STATS` to an output filename prefix before launching Gufo:

```sh
GUFO_EXPERT_STATS=/tmp/qwen-experts ./result/bin/gufo chat --model "$MODEL"
```

Use a binary built from this branch and your usual model arguments; the same
variable also works with `serve llm`. Tracing activates when a Qwen3.8
Flash Next GPU executor is created. Each executor writes a separate
`PREFIX.TIMESTAMP.SERIAL.csv`; the parent directory must exist. Disabled by
default, this diagnostic adds one GPU copy kernel per routing call and downloads
buffered records at existing stream synchronization points. CSV serialization
and disk writes happen synchronously; measure overhead before using traced
runs for performance comparisons.

```csv
routing_id,component,layer,token_row,expert_0,expert_1,expert_2,expert_3,expert_4,expert_5,expert_6,expert_7,expert_8,expert_9
0,trunk,0,0,17,203,4,88,9,16,21,42,73,91
```

There is one row per token row per layer routing invocation. `routing_id` is a
monotonically increasing **layer routing invocation** ID within that executor,
not a model-forward ID. Rows sharing it were routed together; `token_row` is
zero-based within that invocation, not an absolute context position or a stable
request ID. Layer indices are zero-based, with the MTP block at layer 48.
Expert columns are router rank, not execution order. Records preserve routing
invocation order and token-row order within each invocation, including GPU graph
replays. Multi-request batches do not currently label the originating request.

The trace includes prefill, decode, MTP drafts, rejected speculative work, and
any other executed routing. Prefix-cache hits that skip computation produce no
rows. Counts describe performed routing, not accepted output tokens. Trunk and
MTP expert identities are distinct. Analyze each routing invocation as a batch
of simultaneous expert requirements, rather than assuming token-row order is
physical GPU memory-access order.

Records append at normal synchronization points and on executor destruction.
The GPU buffer holds four full `max_batch` model passes; overflow fails inference
with an explicit incomplete-trace message rather than silently dropping rows.
A crash or forced kill can lose the last undrained records. Each CSV is an append
log: do not treat it as the former cumulative histogram format. Trace files can
be large (48 rows per trunk token, plus any MTP work).

Use `python3 tools/models/qwen38_flash_next/expert_usage.py TRACE.csv` to summarize
frequencies and the fraction of selections covered by the most-used experts in
each layer. This retrospective fixed-set coverage is not an LRU cache hit rate;
the ordered trace can support that analysis separately.

## SSD-backed expert LRU cache (experimental)

Set `GUFO_EXPERT_CACHE_MIB` to an integer memory budget in MiB. For example,
32 GiB including the expert workspace and disk transfer buffers:

```sh
GUFO_EXPERT_CACHE_MIB=32768 nix run --refresh github:fkz/gufo/expert-usage-stats#gfx1152 -- chat --model "$MODEL"
```

The setting works with `serve llm` too. Unset it or set it to `0` to use the
original eager upload. Invalid values and budgets too small for a complete
layer fail explicitly. The error reports the minimum budget for this model.
Keep the bound GGUF files available and unchanged throughout inference.
Startup also prints the MiB required for all experts. If the budget fits all
of them, loading uses the original resident layout and GPU graphs automatically;
there is no LRU, workspace, or routing download overhead in that case.

Each loaded model owns **one global LRU** keyed by `(layer, expert)`, shared by
all its sessions and including MTP when loaded. Routed gate/up/down payloads
stay in their GGUF encoding and are fetched only on demand. The router, shared
experts, attention and other non-routed weights remain resident. No prediction,
expert dropping, or additional quantization is performed.

The reservation includes:

- An arena of expert slots, each sized for the largest encoded expert among
  the loaded layers. Mixed-precision layers therefore leave some slot padding;
  startup prints the actual slot count, not an estimate from Q4 parameter bits.
- One reusable layer workspace, sized separately for the largest gate, up and
  down tensors, with quantized-kernel tail padding. This remains reserved for large prefill batches.
- The upload pipeline's roughly 256 MiB of pinned SSD transfer buffers.

KV cache, recurrent state, ordinary executor scratch, other model weights,
allocator overhead, and small CPU routing/LRU metadata are additional memory.
The reported resident model reservation includes the cache even while cold;
physical memory usage can increase as its pages are first populated.

After routing, the cache pins every expert needed by that invocation, evicts
least-recently-used unpinned entries, and reads misses through the bounded
GGUF upload pipeline (unbuffered reads where supported). Prefill can pin all
512 experts in one layer; the minimum budget guarantees this fits. Recency is
touched in token-row/router-rank order. GPU events order separate executors before slots or the shared workspace
can be reused, while a scheduled multi-session batch shares its selected experts.

Decode and verification vector kernels read weights **directly from cache
slots** through a small device pointer table. Hits transfer only routing IDs
and pointers; they do not copy expert payloads or clear per-expert workspace
tails. Slot padding is initialized only on misses. The existing quantized dot
products, grouping, original expert IDs and output order are retained. Large
prefill batches still copy selected weights into the reusable layer workspace
for the tiled matrix kernels.

Miss reads can overlap the independent shared expert; routed projections wait
for their uploads. GPU graph replay remains disabled for an active LRU. A
consumer event replaces the blocking synchronization at the end of each expert
invocation, and the next routing download waits for that event before allowing
eviction. The routing result still has to reach the CPU, so a partial LRU cache
can remain slower than resident weights even with many hits.

The focused GPU checks cover exact output equality with resident projections
for Q4_K/Q5_K/Q5_1/Q8_0 and prequantized gate/up inputs, plus eviction,
workspace fallback, peer-stream lifetime and failed I/O. These checks do not
establish full-model numerical parity or end-to-end throughput.

Normal model teardown logs cache hits, misses, and payload MiB read. Counts
include prefill, drafts and verification, not just accepted output tokens.
Every routed selection counts as an access; repeated selections within one
invocation count as hits after the first miss. Disk-read byte counts exclude
alignment and filesystem overhead. `GUFO_EXPERT_STATS` can be used alongside
caching; the trace still contains original model expert IDs.
