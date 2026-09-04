# Qwen3.6 to Qwen3.8-Flash-Next compatibility matrix

This matrix is the implementation boundary. `Reuse` means the control or TTNN
mechanism can be carried forward with shape/topology assertions. `Adapt` means
the mechanism is useful but Qwen4Exp equations, tensor names, or placement must
replace the old contract. `Reject` means executing the old path would be a
semantic error even if its shapes happened to fit.

The reusable TP4 baseline is the in-tree `models/demos/blackhole/qwen36`
implementation at this run's starting revision. The separately pinned
Qwen3.6-A3B bundle is also audited because it contains newer single-card GDN,
MoE, paged-cache, and MTP experiments; its demos explicitly open a `1x1` mesh,
so it is not evidence for TP4 placement.

| Area | Decision | Reuse boundary and required Qwen4Exp change |
|---|---|---|
| Mesh/demo lifecycle | Reuse | Reuse the in-tree text-demo/vLLM lifecycle, trace cleanup, paged slot ownership, and TP request plumbing. Force `MeshShape(1, 4)`, verify four distinct physical devices and topology metadata, and remove any fallback to one device or replicated weights. |
| TP tensor helpers and CCL | Reuse | Reuse `qwen36/tt/tp_common.py` sharding helpers and the proven CCL call sites only where the Qwen4Exp dimension divides by four. All returned tensor topology is asserted. QSA KV grouping and expert parallelism get explicit new placement. |
| Configuration | Reject | Qwen3.6 layer schedules, dimensions, expert counts, residual contract, and MTP flags are not accepted. Load the pinned `Qwen4ExpForConditionalGeneration` config and fail closed on all architecture facts and the 48-layer `3xGDN+1xQSA` schedule. |
| Lazy checkpoint loader/cache | Adapt | Reuse the safetensors-index lazy thunk and per-tensor cache pattern. Replace every prefix/shape table with the exact 1,658-tensor Qwen4Exp manifest, retain `mtp.*`, keep PLE tables on host, omit only vision execution, and encode TP/EP placement plus precision in cache keys. |
| Zero-centered RMSNorm | Reuse | The `add_unit_offset=True` implementation matches Qwen4Exp's `(1 + weight)` convention. Preserve FP32 variance and apply grouping per GR branch where the weight is `4H`; ordinary Qwen3.6 whole-stream norm calls cannot stand in for grouped GR norms. |
| Residual path | Reject | Qwen3.6's two ordinary residual adds are invalid. Implement the four-branch GR read/write modules: branchwise RMSNorm, sigmoid read gate averaged over branches, rank-320 bottleneck, and one sigmoid-derived scalar write coefficient per branch. The persistent hidden representation is `4H`. |
| PLE / n-gram embedding | Reject | Qwen3.6 has no equivalent. Implement exact SplitMix64 multipliers, successive-prime per-head moduli, EOS-aware bigram/trigram history, signed-square-root gate, dilation-3/kernel-4 causal convolution, and host-resident 128-shard lookup at checkpoint layer 1. Transfer only the selected `4x640` contribution per token. |
| GDN recurrence | Adapt | Reuse the in-tree TP GDN projection/CCL, causal-convolution, recurrent-kernel, and state-lifetime machinery after numerical comparison. Replace shapes with 16 Q/K heads and 48 V heads at 128 dimensions, preserve FP32 recurrent state and exact decay/beta equations, replace Qwen3.6's SiLU output gate with the checkpoint-configured **sigmoid** gate, and integrate GR/PLE around the block. |
| Full attention / QSA | Reject core; adapt cache shell | Qwen3.6 dense attention is not final QSA. Reuse RoPE, output projection, paged KV allocation, and overwrite mechanics where contracts match. Add the four-head/one-KV-head indexer, raw-key block pooling before RoPE, complete-block top-k selection, causal incomplete tail, 24Q/2KV attention, and the explicit two-pair KV placement. A dense short-context oracle is test-only. |
| Embeddings and LM head | Adapt | Reuse vocabulary-parallel mapping and distributed top-token selection because 248,320 divides by four. Qwen4Exp input/output hidden state is `4H` around GR, PLE is additive at one layer only, embeddings are shared with MTP, and no replicated-LM-head fallback is permitted. |
| Router | Adapt | Reuse FP32 softmax, top-k selection, top-k renormalization, and dynamic shared-expert gate arithmetic. Change to 512 routed experts/top-10 on every target and MTP layer and preserve real scores/indices through dispatch and weighted reduction. |
| Routed experts | Reject old placement | The pinned bundle stores all expert weights as `[1,E,...]` for single-card indexed `sparse_matmul`; that does not scale memory across four cards. The exact oracle now enforces 128 experts/device with BF4_B accounting and complete score-weighted reduction. Current-tree all-to-all dispatch cannot express true global-B=1 across four devices, and current-tree `sparse_matmul` has no expert-index argument, so neither is silently promoted to the required decode path. A device implementation must retain EP4 ownership, exact top-10 scores, and a final reduction of all four partials without padding or replication. |
| Shared expert | Adapt | Reuse fused SwiGLU and sigmoid gate sequencing, but shard its divisible projections across TP4 and combine with the exactly weighted routed output at `H=2560`, `I=640`. It remains dynamic on every token. |
| MTP input/head | Reject | Qwen3.6 concatenates one embedding and one `H` hidden vector through a `[H,2H]` projection. Qwen4Exp normalizes the `4H` target branches, applies shared `fc_hidden[H,H]` to each branch, projects the embedding through `fc_embedding[H,H]`, broadcasts/adds it to all four branches, then runs the checkpoint QSA+GR+MoE layer. |
| Speculative control | Adapt | Reuse fixed-shape K+1 target batching, acceptance accounting, handoff, and trace-buffer concepts. Replace all model-state handling: QSA draft steps reuse the frozen target selection plus causal tail; target verification records five GDN/conv/PLE state candidates and commits only `current + accepted_drafts`. Validate rejection depths 0--4 before tracing. |
| Chat/server shell | Reuse | Reuse loopback serving, streaming, persistent turns, lifecycle logging, and safe cleanup. Rendering and parsing come from the pinned official template, including thinking preservation, reasoning-effort controls, and XML-like tool calls. |
| Vision | Reject for this milestone | The multimodal tensors remain in the manifest and license provenance but are not loaded or exposed. The demo reports vision as unsupported. |

## Primary semantic pins

- Transformers `dabae5fcb924a8eece0e727b627ca5f050b40d40` defines target GR, PLE,
  GDN, QSA, router, and cache semantics.
- SGLang Qwen4Exp integration PR head
  `73a255206f916366c8d26d4022f82ddfb0ab558d` defines the checkpoint MTP
  input fusion, shared-index drafting, and verifier commit behavior not exposed
  by the pinned Transformers model class.
- Qwen3.6-A3B bundle `42a71032659a9d0e3b860b6104c59cfa6ca82149`
  is a mechanism reference only. Its single-card mesh and generic MTP
  architecture are not promoted to model semantics.
