// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_nanobind.hpp"
#include "chunk_gated_delta_rule.hpp"
#include "device/chunk_gdn_phased.hpp"

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn-nanobind/bind_function.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ttnn/device.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::transformer {

namespace {

// The composite's own kernel-config default (chunk_gated_delta_rule.cpp): HiFi4, no approximations,
// fp32 accumulate, no L1 accumulate. Shared by both phase prims so a Python caller that passes
// compute_kernel_config=None reproduces the composite's call exactly.
ttnn::DeviceComputeKernelConfig phase_kernel_config(
    const ttnn::Tensor& reference, const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::init_device_compute_kernel_config(
        reference.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/false);
}

// Thin forwarders: they only apply the composite's two defaults (DRAM output, the kernel config
// above) and hand every argument straight to the prim. No shape work, no op logic.
std::vector<ttnn::Tensor> chunk_gdn_prep_py(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& eye,
    const ttnn::Tensor& tril,
    const ttnn::Tensor& ones,
    const ttnn::Tensor& masks,
    uint32_t chunk_size,
    float scale,
    bool v_flat,
    uint32_t HV,
    bool qk_flat,
    uint32_t Hk,
    bool qk_norm,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::chunk_gdn_prep(
        q,
        k,
        v,
        g,
        beta,
        eye,
        tril,
        ones,
        masks,
        chunk_size,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        phase_kernel_config(q, compute_kernel_config),
        v_flat,
        HV,
        qk_norm,
        scale,
        qk_flat,
        Hk);
}

std::vector<ttnn::Tensor> chunk_gdn_scan_py(
    const ttnn::Tensor& v_beta,
    const ttnn::Tensor& kd,
    const ttnn::Tensor& q_decay,
    const ttnn::Tensor& intra,
    const ttnn::Tensor& k_dec_t,
    const ttnn::Tensor& dl,
    const ttnn::Tensor& t_inv,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::chunk_gdn_scan(
        v_beta,
        kd,
        q_decay,
        intra,
        k_dec_t,
        dl,
        t_inv,
        initial_state,
        chunk_size,
        output_final_state,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        phase_kernel_config(v_beta, compute_kernel_config));
}

}  // namespace

void bind_chunk_gated_delta_rule(nb::module_& mod) {
    const auto* doc =
        R"doc(
        Standalone chunked Gated Delta Rule forward (flash-linear-attention algorithm).

        Args:
            q (ttnn.Tensor):    [B, T, H,  K]
            k (ttnn.Tensor):    [B, T, H,  K]
            v (ttnn.Tensor):    [B, T, HV, V]
            g (ttnn.Tensor):    [B, T, HV]   log-space decay
            beta (ttnn.Tensor): [B, T, HV]

        Keyword Args:
            scale (float, optional): defaults to K**-0.5.
            initial_state (ttnn.Tensor, optional): [B, HV, K, V].
            output_final_state (bool): default False.
            chunk_size (int): default 64.
            use_qk_l2norm (bool): default False.
            output_head_major (bool): default False. When True, o is returned head-major as
                [B*HV, T, V] in TILE layout (skips the token<->head permute round-trip);
                otherwise token-major [B, T, HV, V] ROW_MAJOR.
            memory_config (ttnn.MemoryConfig, optional).
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional).
            eye, tril, ones (ttnn.Tensor, optional): [1,1,C,C] fp32 TILE constant tiles (identity,
                lower-triangular ones, all-ones). Caller-supplied so they are device-resident before
                trace capture and their lifetime is device-scoped. Traced callers MUST pass these
                (an internal build does a host upload, illegal under trace); if omitted they are
                built eagerly.
            masks (ttnn.Tensor, optional): [1,1,32,96] fp32 TILE quadrant masks; supplied with eye/
                tril/ones.

        Returns:
            tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
                o [B, T, HV, V] (or [B*HV, T, V] if output_head_major),
                final_state [B, HV, K, V] (if output_final_state).

        The two phase prims this op runs internally are also bound, as ttnn.prim.chunk_gdn_prep and
        ttnn.prim.chunk_gdn_scan, for callers that lay the per-chunk inputs out themselves.
        )doc";

    ttnn::bind_function<"chunk_gated_delta_rule", "ttnn.transformer.">(
        mod,
        doc,
        &ttnn::transformer::chunk_gated_delta_rule,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("output_final_state") = false,
        nb::arg("chunk_size") = 64,
        nb::arg("use_qk_l2norm") = false,
        nb::arg("output_head_major") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("eye") = nb::none(),
        nb::arg("tril") = nb::none(),
        nb::arg("ones") = nb::none(),
        nb::arg("masks") = nb::none());

    // ---------------------------------------------------------------------------------------------
    // The two phase prims of the same kernel, exposed as ttnn.prim.chunk_gdn_prep / chunk_gdn_scan.
    // The composite above is unchanged and still the supported entry point; these bindings add no
    // kernel and no op logic, they are the same two device operations the composite launches.
    // ---------------------------------------------------------------------------------------------
    const auto* prep_doc =
        R"doc(
        PREP phase of the chunked Gated Delta Rule (state-independent, parallel over head x chunk).

        This is the first of the two device operations ttnn.transformer.chunk_gated_delta_rule runs.
        It takes the per-chunk inputs already laid out head-major, so a caller that keeps its own
        chunk layout can call the two phases directly and skip the composite's relayout. The
        composite's own call is reproduced exactly by passing the tensors it builds (below) and
        leaving memory_config / compute_kernel_config unset.

        C = chunk_size, NC = number of chunks, BH = B * HV, K = key/head dim, V = value/head dim.

        Args:
            q (ttnn.Tensor): [BH, NC, C, K] BFLOAT16 TILE, head-major, GQA-expanded to HV heads, and
                (unless qk_norm) already L2-normalized with ``scale`` folded in. Under qk_flat this
                is instead the flat token-major [B, T, Hk*K] BFLOAT16 TILE tensor and the reader
                addresses key head hk = hv/G itself.
            k (ttnn.Tensor): same layout as q, without the scale fold.
            v (ttnn.Tensor): [BH, NC, C, V] BFLOAT16 TILE head-major, or, under v_flat, the flat
                token-major [B, T, HV*V] BFLOAT16 TILE tensor (the reader tile-addresses head hv's
                chunk out of it). Only the read changes: the prep always WRITES head-major v_beta.
            g (ttnn.Tensor): [BH, NC, C, 1] FLOAT32 TILE, the log-space decay as a column.
            beta (ttnn.Tensor): [BH, NC, C, 1] FLOAT32 TILE column.

        Keyword Args:
            eye (ttnn.Tensor): [1, 1, C, C] FLOAT32 TILE identity.
            tril (ttnn.Tensor): [1, 1, C, C] FLOAT32 TILE lower-triangular ones.
            ones (ttnn.Tensor): [1, 1, C, C] FLOAT32 TILE all-ones.
            masks (ttnn.Tensor): [1, 1, 32, 96] FLOAT32 TILE — the three 32x32 WY-inverse quadrant
                masks (top-left | bottom-right | bottom-left) packed into one tile row.
                All four are required here (the composite's eager fallback build is host-side and
                illegal under trace); they are the same tensors the composite takes as eye/tril/
                ones/masks.
            chunk_size (int): C, a multiple of 32. The composite passes its own chunk_size.
            scale (float): folded into q's in-kernel L2 norm when qk_norm, otherwise unused (the
                caller already folded it into q). The composite passes scale or K**-0.5.
            v_flat (bool): default False. v is the flat token-major [B, T, HV*V] tensor. Requires
                HV and T % C == 0. The composite sets it when v arrives rank-3.
            HV (int): default 0. The value-head count, required by v_flat (the flat row stride).
            qk_flat (bool): default False. q/k are flat token-major [B, T, Hk*K]. Requires Hk and
                qk_norm, and C == 32. The composite sets it when q/k arrive rank-3.
            Hk (int): default 0. The key-head count, required by qk_flat. The composite passes the
                q head count H (= HV when q/k were GQA-expanded on the host).
            qk_norm (bool): default False. The prep compute L2-normalizes q/k over K in-kernel and
                folds ``scale`` into q's norm; only valid at C == 32. The composite sets it exactly
                when q/k arrived flat.
            memory_config (ttnn.MemoryConfig, optional): defaults to DRAM, as in the composite.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to the
                composite's own — HiFi4, no approximations, fp32 accumulate, no L1 accumulate.

        Returns:
            list[ttnn.Tensor]: the 7 per-chunk hand-off tensors, all FLOAT32 TILE, in the order the
            scan takes them:
                0 v_beta  [BH, NC, C, V]  (= v * beta)
                1 kd      [BH, NC, C, K]  (= k_beta * decay_exp)
                2 q_decay [BH, NC, C, K]
                3 intra   [BH, NC, C, C]
                4 k_dec_t [BH, NC, K, C]
                5 dl      [BH, NC, 1, 1]  (one scalar per chunk, in tile element [0, 0])
                6 t_inv   [BH, NC, C, C]  (the WY inverse, un-premultiplied)
        )doc";

    ttnn::bind_function<"chunk_gdn_prep", "ttnn.prim.">(
        mod,
        prep_doc,
        &chunk_gdn_prep_py,
        nb::arg("q").noconvert(),
        nb::arg("k").noconvert(),
        nb::arg("v").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("beta").noconvert(),
        nb::kw_only(),
        nb::arg("eye").noconvert(),
        nb::arg("tril").noconvert(),
        nb::arg("ones").noconvert(),
        nb::arg("masks").noconvert(),
        nb::arg("chunk_size"),
        nb::arg("scale"),
        nb::arg("v_flat") = false,
        nb::arg("HV") = 0,
        nb::arg("qk_flat") = false,
        nb::arg("Hk") = 0,
        nb::arg("qk_norm") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());

    const auto* scan_doc =
        R"doc(
        SCAN phase of the chunked Gated Delta Rule (sequential over chunk, parallel over head).

        The second of the two device operations ttnn.transformer.chunk_gated_delta_rule runs: it
        consumes the seven tensors ttnn.prim.chunk_gdn_prep returned plus the initial state, carries
        the recurrent state S [K, V] on-core across the chunks, and produces o and the final state.
        The composite passes the prep outputs positionally in the order they were returned.

        C = chunk_size, NC = number of chunks, BH = B * HV, K = key/head dim, V = value/head dim.
        Every input below is FLOAT32 TILE and comes straight from the prep.

        Args:
            v_beta (ttnn.Tensor): [BH, NC, C, V]  prep output 0.
            kd (ttnn.Tensor): [BH, NC, C, K]      prep output 1.
            q_decay (ttnn.Tensor): [BH, NC, C, K] prep output 2.
            intra (ttnn.Tensor): [BH, NC, C, C]   prep output 3.
            k_dec_t (ttnn.Tensor): [BH, NC, K, C] prep output 4.
            dl (ttnn.Tensor): [BH, NC, 1, 1]      prep output 5.
            t_inv (ttnn.Tensor): [BH, NC, C, C]   prep output 6 (the WY inverse; the scan applies it
                AFTER the v_beta - kd @ S subtraction, so the inverse's fp error is not amplified).
            initial_state (ttnn.Tensor, optional): [BH, K, V] FLOAT32 TILE. Default None = an
                implicit zero state. The composite always passes a tensor: the caller's initial
                state reshaped to [BH, K, V] (cast to FLOAT32), or a device-side zeros() when the
                caller gave none.

        Keyword Args:
            chunk_size (int): C, a multiple of 32; the same value the prep was given.
            output_final_state (bool): default False. When False the final-state output tensor is
                still allocated and returned, but the kernel does not write it. The composite
                passes its own output_final_state.
            memory_config (ttnn.MemoryConfig, optional): defaults to DRAM, as in the composite.
            compute_kernel_config (ttnn.DeviceComputeKernelConfig, optional): defaults to the
                composite's own — HiFi4, no approximations, fp32 accumulate, no L1 accumulate.

        Returns:
            list[ttnn.Tensor]: 2 tensors, both FLOAT32 TILE, in the C++ order:
                0 o           [BH, NC, C, V] — head-major; the composite folds NC,C -> T into
                              [BH, T, V] (a metadata-only reshape when T % C == 0) for
                              output_head_major, and otherwise permutes it to [B, T, HV, V].
                1 final_state [BH, K, V] — the composite reshapes it to [B, HV, K, V]. Written
                              only when output_final_state.
        )doc";

    ttnn::bind_function<"chunk_gdn_scan", "ttnn.prim.">(
        mod,
        scan_doc,
        &chunk_gdn_scan_py,
        nb::arg("v_beta").noconvert(),
        nb::arg("kd").noconvert(),
        nb::arg("q_decay").noconvert(),
        nb::arg("intra").noconvert(),
        nb::arg("k_dec_t").noconvert(),
        nb::arg("dl").noconvert(),
        nb::arg("t_inv").noconvert(),
        nb::arg("initial_state") = nb::none(),
        nb::kw_only(),
        nb::arg("chunk_size"),
        nb::arg("output_final_state") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none());
}

}  // namespace ttnn::operations::transformer
