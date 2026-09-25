# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused decode kernels over ``ttnn.generic_op``.

``program`` builds and runs one program from Python (kernels, CBs, semaphores, the rows contract); ``registry`` names
each fused kernel with the composed ttnn chain it replaces; the proven kernels serve by default (``QWEN38_FUSED_OFF``
falls back to the chains, ``QWEN38_FUSED`` switches an opt-in kernel on).  Each kernel
is a sub-package ``<name>/`` with its ``kernels/*.cpp`` and registers itself on import; add new ones to the import
list below.  Gate and accounting: FUSED-KERNEL-HOWTO.md under the dev tools.
"""

from . import program
from .registry import (
    ALL,
    BITWISE,
    COMPONENT,
    DEFAULT_ON,
    ENV,
    OFF_ENV,
    TOLERANCE_CLASSES,
    ULP,
    AdmittedStep,
    FusedKernel,
    GateSpec,
    default_names,
    enabled,
    enabled_names,
    kernel,
    kernels,
    register,
    resolve,
    resolve_admitted,
)
from . import (
    final_mixer,
    gdn_step,
    gr_read,
    gr_write,
    greedy_tail,
    moe_post,
    ple,
    position_derive,
    qsa_block,
    router_tail,
    sampler_tail,
    shared_expert,
    untilize_rows,
)
from . import gr_fold  # after gr_read: it composes gr_read's programs

__all__ = [
    "ALL",
    "BITWISE",
    "COMPONENT",
    "DEFAULT_ON",
    "ENV",
    "OFF_ENV",
    "TOLERANCE_CLASSES",
    "ULP",
    "FusedKernel",
    "GateSpec",
    "default_names",
    "enabled",
    "enabled_names",
    "final_mixer",
    "gdn_step",
    "gr_read",
    "gr_write",
    "greedy_tail",
    "kernel",
    "kernels",
    "moe_post",
    "ple",
    "position_derive",
    "program",
    "qsa_block",
    "register",
    "resolve",
    "router_tail",
    "sampler_tail",
    "shared_expert",
    "untilize_rows",
    "AdmittedStep",
    "resolve_admitted",
]
