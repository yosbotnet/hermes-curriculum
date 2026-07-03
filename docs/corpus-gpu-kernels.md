# Corpus guide: GPU kernel optimization and performance engineering

This is a companion guide for assembling a `corpus.json` aimed at one goal: becoming
proficient at GPU kernel optimization and performance engineering, from CUDA fundamentals
through the internals of modern LLM inference engines (vLLM, FlashAttention). It does not
ship any course material. You supply your own plain-text extracts of legally obtained
sources (your own book purchases, publicly posted lecture transcripts, official vendor
documentation you have downloaded, blog posts you have saved) and point `corpus.json` at
them. Nothing from the sources named below is committed to this repository, and this guide
does not reproduce their text, chapter numbers, or URLs beyond what is needed to help you
find and organize them yourself.

## a. The spine-and-satellites model

The ingestion pipeline extracts concepts from every source you list and then has to decide
which pairs of concepts are prerequisites of each other. Two kinds of source play very
different roles in that decision:

- A **spine** source is one whose internal ordering you trust as a prerequisite chain. A
  textbook's chapter sequence is the canonical example: chapter 3 assumes chapter 2, which
  assumes chapter 1, by editorial design. When you mark a source as spine, the concepts
  extracted from it, taken in the order they appear in the source file, are chained directly
  into `PREREQUISITE` edges with full confidence and a provenance tag identifying them as
  spine-derived. The engine does not need to guess these edges or hedge on them; the author
  of the source already did the ordering work for you.

- A **satellite** source adds concepts and cross-links to the graph without any ordering you
  are willing to trust. Reference documentation, a metrics glossary, a task-oriented
  best-practices guide organized by topic rather than by dependency, a single blog post, or
  a research paper are typical satellites: they are valuable for filling in vocabulary and
  linking related ideas, but reading section 4 before section 2 of a reference manual does
  not imply that section 2 is a prerequisite of section 4. Edges touching satellite-derived
  concepts are inferred (by the linking pass, from co-occurrence and semantic similarity) and
  carry capped, lower confidence rather than the full trust given to spine edges. Inference
  never overwrites a prerequisite edge that a spine already established.

Practically, this means the single highest-leverage decision you make when building this
corpus is which source anchors each phase as the spine. Everything else in that phase is a
satellite that hangs concepts off the spine's backbone.

Note on chaining scope: the `spine` boolean on a `corpus.json` source entry is honored by
the ingestion pipeline (SpinePass; see `docs/superpowers/plans/2026-07-03-motivation-layer.md`),
but chaining happens PER SOURCE FILE -- each source is ingested through its own pipeline run,
so concepts are chained within one spine file and never across files. Practical consequence:
put an entire spine (for example, all the PMPP chapters you transcribe) into ONE plain-text
file, in order. Splitting a spine across several `spine: true` files produces several
disconnected chains with no trusted edges between them, which defeats the point of a spine.
Cross-file stitching is tracked as follow-up work; until it lands, one file per spine.

## b. Phase 1 -- CUDA foundations and the performance model

Goal of this phase: the CUDA execution and memory model (threads, warps, blocks, grids;
global/shared/register/constant memory; memory coalescing; occupancy; warp divergence) well
enough to read and reason about a kernel's performance before you touch a profiler.

**Spine, option A (paid): "Programming Massively Parallel Processors" (PMPP), 4th edition,
by Hwu, Kirk, and El Hajj.** Use the book's own table of contents as the spine order --
do not renumber or re-derive chapter numbers from this guide, since printings and editions
vary. The book's arc runs, broadly: an introduction to heterogeneous/data-parallel computing,
the core CUDA programming model (kernels, threads, blocks), the memory hierarchy and
coalescing, performance considerations (occupancy, divergence, tiling), a long run of
parallel-pattern chapters (convolution, stencil, histogram, reduction, prefix sum/scan,
merge, sorting, sparse matrix formats), and later chapters on graph traversal, deep-learning
primitives, and more advanced/dynamic-parallelism topics. Transcribe or extract the chapters
you want, in that order, into ONE plain-text file and mark that single source `spine: true`
(spine chaining is per source file; see the note in section a).

**Spine, option B (free alternative): GPU MODE lecture series plus the CUDA C++ Programming
Guide.** GPU MODE (formerly CUDA MODE) publishes a numbered lecture series covering the same
ground as PMPP -- the CUDA programming model, memory hierarchy, and profiling -- plus later
lectures on Triton and library internals that this guide's later phases also draw on. Use
the lecture numbering as published on the channel/playlist at the time you build your
corpus, since the series continues to grow and lecture numbers for a given topic can shift.
Interleave or follow it with the official CUDA C++ Programming Guide from NVIDIA, whose
sections run, in recent versions, from an introduction, through the programming model, the
programming interface, hardware implementation, and performance guidelines, with compute
capability and language-extension material in appendices; check the guide's own current
table of contents rather than trusting a specific section count here, since NVIDIA revises
it across CUDA releases. Either the lecture series or the Programming Guide alone can serve
as spine; combining them, treat whichever you extract text from first, in the order you
extract it, as the spine and add the other as a satellite if you are not confident chaining
the two together into one ordering.

**Satellites for this phase:**

- The **CUDA Best Practices Guide** (NVIDIA), organized around its own APOD cycle (Assess,
  Parallelize, Optimize, Deploy). It is meant to be consulted by topic as you optimize, not
  read start-to-front as a dependency chain, so treat it as a satellite even though it is an
  official, well-organized document.
- The **Nsight Compute documentation** (NVIDIA), covering the profiler's metrics and report
  sections (for example the "Speed of Light" summary, memory workload analysis, scheduler
  and warp state statistics). This is reference material you will return to constantly once
  you start profiling in Phase 2; it has no narrative order to trust as prerequisites.

## c. Phase 2 -- optimization craft

Goal of this phase: turn a correct-but-slow kernel into a fast one, and be able to explain,
with profiler evidence, why each rewrite helped. This phase is organized around a small set
of worked examples rather than a single spine textbook.

**Primary satellite: Simon Boehm's CUDA matmul worklog** ("How to Optimize a CUDA Matmul
Kernel for cuBLAS-like Performance," or search for Simon Boehm's blog post on optimizing
CUDA matrix multiplication). It walks a single SGEMM kernel through a sequence of concrete,
measured optimizations -- naive, global-memory coalescing, shared-memory tiling, 1D
blocktiling, 2D blocktiling, vectorized memory access, and warp-tiling -- each benchmarked
against the step before it, with 1D and 2D blocktiling built and measured as two separate
kernels rather than a single combined step. Boehm discusses double buffering (software
pipelining of shared-memory tile loads) only as proposed future work in the post's closing
section; no kernel implementing it is built or benchmarked there, so do not attribute a
measured double-buffering step to this source. For an implementation of the technique, see
the CUTLASS/CuTe documentation in the satellite list below. The measured portion of this
worklog is what most of the checkpoint ladder in section (e) is modeled on; checkpoint 4
(double buffering) extends that ladder one step past what Boehm actually measures. Because
it is a single blog post rather than an authored curriculum with editorially trusted chapter
dependencies, treat it as a satellite: its concepts should link into the graph, but do not
mark it spine.

**Other satellites for this phase:**

- **Horace He's "Making Deep Learning Go Brrr (From First Principles)"** -- search for that
  title on his personal blog. It builds the compute-bound / memory-bound / overhead-bound
  mental model (a roofline-style way of diagnosing what is actually limiting a kernel or
  model) that you will use to decide which optimization from Boehm's worklog, or from
  tensor-core programming, is worth applying next.
- **Tensor cores and CUTLASS/CuTe documentation** -- NVIDIA's CUTLASS is a templated CUDA
  library for GEMM and related kernels; CuTe is the layout-algebra library underlying
  CUTLASS 3.x. Their documentation (in the CUTLASS GitHub repository's docs and README
  material) explains warp-level matrix-multiply-accumulate (MMA) primitives and tile
  scheduling, including the software-pipelined (double-buffered) global-to-shared-memory
  loads that Boehm's worklog above leaves as unimplemented future work. Treat this as
  reference/satellite material you dip into once you are ready to move a kernel from plain
  CUDA cores onto tensor cores.
- **Triton tutorials** -- OpenAI's Triton ships an official tutorial sequence (vector add,
  fused softmax, matrix multiplication, low-memory dropout, layer norm, fused attention).
  Useful both as a second, higher-level language for the same optimization ideas and as a
  bridge toward the fused-attention material in Phase 3.
- **Citadel's Volta and Turing microbenchmarking papers** -- search for "Dissecting the
  NVIDIA Volta GPU Architecture via Microbenchmarking" and the corresponding Turing paper
  from Citadel (Securities) researchers. These reverse-engineer instruction latencies,
  memory-hierarchy behavior, and tensor-core characteristics that vendor documentation does
  not spell out, and are useful satellites once you need ground truth below the level the
  Programming Guide describes.

## d. Phase 3 -- inference systems

Goal of this phase: understand how the kernel-level techniques from Phases 1 and 2 compose
into a real LLM inference engine, specifically attention kernels and request scheduling.

**Satellites, read in roughly this order:**

- **The FlashAttention papers, in sequence: FlashAttention, FlashAttention-2, and
  FlashAttention-3.** These are IO-aware exact-attention algorithms: the original paper
  establishes tiling and recomputation to avoid materializing the full attention matrix,
  FlashAttention-2 improves work partitioning and parallelism, and FlashAttention-3 targets
  newer hardware (warp specialization, low-precision execution). Search for each by name and
  version number; read them in that order since each explicitly builds on the last.
- **The vLLM paper** -- "Efficient Memory Management for Large Language Model Serving with
  PagedAttention" -- which introduces PagedAttention (applying an OS-style paging idea to the
  KV cache) and discusses continuous batching. Read this before the vLLM documentation and
  well before the vLLM source, since the paper motivates the design decisions the code
  implements.
- **The vLLM documentation** -- the official docs site, covering the engine's architecture,
  scheduler, and configuration. Use it to connect the paper's abstractions to the concrete
  system.
- **Continuous batching** as a concept is covered by both the vLLM paper and docs and by
  independent write-ups; if you extract a separate piece on it, treat it as another
  satellite rather than a spine, since none of these sources has an editorially trusted
  chapter order the way a textbook does.

**vLLM source code, deliberately last.** Reading the actual vLLM source (scheduler, paged
KV-cache allocator, attention kernel bindings) is the capstone of this phase, once the paper
and docs have given you the vocabulary and the design rationale to read the code as an
implementation of ideas you already understand, rather than reverse-engineering the ideas
from the code cold. Source code has no chapter order to trust as a spine either; if you
extract notes from it into the corpus at all, add them as a satellite, and expect this to be
the smallest and last-added source in the corpus.

## e. Checkpoint-concept ladder

Across Phases 1 through 3, six checkpoints mark the spine of practical skill, independent of
which textbook or lecture series you used to learn the underlying theory:

1. **Naive matmul.** A straightforward CUDA kernel, one thread per output element, correct
   against a reference implementation.
2. **Coalesced.** The same kernel rewritten so that global memory accesses within a warp are
   coalesced, with a measured throughput improvement over the naive version.
3. **Tiled.** Shared-memory blocking (tiling) added to cut redundant global memory traffic,
   with occupancy and shared-memory usage understood as a tradeoff, not just a knob.
4. **Double-buffered.** Software pipelining of shared-memory tiles (loading the next tile
   while computing on the current one) added to hide memory latency behind compute.
5. **Tensor-core.** A kernel that issues tensor-core matrix-multiply-accumulate operations
   (via WMMA/MMA intrinsics, PTX, or a minimal CUTLASS/CuTe example), with tensor-core
   utilization confirmed in the profiler rather than assumed from the source code.
6. **Flash-attention forward.** A fused, IO-aware attention forward pass that never
   materializes the full attention matrix, verified for correctness against a naive
   attention implementation and for reduced memory traffic in the profiler.

Each of these is phrased as a concept in the graph, but none of them can be marked mastered
by answering questions about it. The curriculum engine can quiz you on why coalescing helps
or what double buffering hides, and you can answer every question correctly without having
written a single line of the kernel. These six checkpoints are self-gated: you, the learner,
mark a checkpoint concept mastered only after you have actually built, run, and (for
checkpoints 2 through 6) profiled the corresponding artifact and confirmed it does what the
checkpoint claims. The engine has no way to compile your code, run it on your GPU, or read
your profiler output, so at these checkpoints it extends the same kind of trust to your
self-report that a spine source extends to its own chapter ordering: trust placed
deliberately, at a specific and named point, rather than inferred and hedged everywhere.

## f. Example corpus.json

The example below follows the schema in `corpus.example.json` at the repository root, with
one addition: a per-source `"spine"` boolean (`true` for a source whose ordering should be
chained into trusted prerequisite edges, omitted or `false` for a satellite). As noted in
section (a), this flag depends on a sibling ingestion change; if it is not yet present in
your checkout, including it is harmless but has no effect until the change lands.

Every path below is a placeholder for a plain-text file you create yourself from material
you already legally own or that is freely and legitimately available (for example, your own
notes transcribed from a lecture, or text you extracted from a PDF you purchased). None of
that material is included in or committed to this repository; only the manifest shape is.

```json
{
  "_comment": "GPU kernel optimization corpus. Paths point at plain-text files you create from your own legally obtained materials; none of that material is committed to this repository.",
  "course": "GPUKernelOptimization",
  "chunk_lines": 150,
  "sources": [
    { "path": "materials/pmpp-chapters-in-order.txt", "token": "pmpp", "spine": true },
    { "path": "materials/cuda-best-practices-guide.txt", "token": "cuda-best-practices" },
    { "path": "materials/nsight-compute-docs.txt", "token": "nsight-compute-docs" },
    { "path": "materials/simon-boehm-cuda-matmul-worklog.txt", "token": "boehm-matmul-worklog" },
    { "path": "materials/horace-he-making-dl-go-brrr.txt", "token": "brrr-essay" },
    { "path": "materials/cutlass-cute-docs.txt", "token": "cutlass-cute-docs" },
    { "path": "materials/triton-tutorials.txt", "token": "triton-tutorials" },
    { "path": "materials/citadel-volta-microbenchmarking.txt", "token": "citadel-volta" },
    { "path": "materials/citadel-turing-microbenchmarking.txt", "token": "citadel-turing" },
    { "path": "materials/flashattention-1.txt", "token": "flashattention-1" },
    { "path": "materials/flashattention-2.txt", "token": "flashattention-2" },
    { "path": "materials/flashattention-3.txt", "token": "flashattention-3" },
    { "path": "materials/vllm-paper-pagedattention.txt", "token": "vllm-paper" },
    { "path": "materials/vllm-docs.txt", "token": "vllm-docs" },
    { "path": "materials/vllm-source-notes.txt", "token": "vllm-source-notes" }
  ]
}
```

If you use the free alternative for Phase 1 instead of PMPP, replace the `pmpp` entry with
one concatenated file of your GPU MODE lecture transcripts or CUDA C++ Programming Guide
sections, in the order you extracted them, and mark that single file `spine: true`; add the
other body of material as a satellite unless you are confident interleaving both into one
single trusted sequence in one file.
