#!/usr/bin/env bash
# Start the Qwen3.8-Flash-Next chat server on a Blackhole QuietBox.
#
#   tools/run_qwen38_chat_server.sh --profile tt-quietbox --checkpoint DIR --cache-root DIR [options]
#
#   --profile tt-quietbox   4x p150b QuietBox (verified)
#   --profile qb2           p300-based box: QuietBox 2 (2x p300c = 4 dies) or a 4x p300 host  -- UNTESTED, see README
#   --instance 0|1          qb2 only: which group of four dies (0 = nodes 0-3; 1 = nodes 4-7 on an 8-die host)
#   --checkpoint DIR        the ModelScope checkpoint directory
#   --cache-root DIR        where the converted weights, the BF4 scratch, the model I/O cache, the JIT cache and the
#                           run directories go (about 10 GB per allocated context; created on the first start)
#   --allocated-context N   32768 (default) | 65536 | 131072 | 262144
#   --mtp K                 multi-token-prediction drafting depth (needs a server that carries the MTP path)
#   --port N --host ADDR    default 8000 on 0.0.0.0 (the QuietBox profiles serve the LAN)
#   --acceptance-prompts D  replay the CPU greedy records in D at startup (--require-json-96 makes a miss fatal)
#   --serve-seconds N       stop after N seconds (default: until SIGTERM)
#   --python PATH           the interpreter that imports ttnn (default: $TT_METAL_HOME/python_env/bin/python)
#   --validate-only         run the server's checks and CPU preparation, do not open the mesh
#
# Requires TT_METAL_HOME: a tt-metal checkout built with the runtime patches this model needs (the launcher compares the
# checkout's commit with the expected base and lists the patches it cannot find).  No locks, no archives, no seals: the
# server records the identity of the ttnn it imports.
set -euo pipefail

readonly HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly MODEL_DIR="$(cd -- "$HERE/.." && pwd -P)"
readonly REPO_ROOT="$(cd -- "$MODEL_DIR/../../../.." && pwd -P)"
readonly SERVER="$HERE/qwen38_chat_server.py"

# tt-metal main this tree was developed on, and the runtime patches the model needs on top of it.  A checkout whose
# history lacks a patch (by subject) is reported; the server's own admission decides whether to run.
readonly EXPECTED_TT_METAL_BASE=d04395ed862b4c65eb6877000c40200f456cb74e
readonly REQUIRED_PATCHES=(
    "Fix empty-rank moe compute metadata ownership"
    "skip idle-expert combine sync in moe_compute B=1"
    "Add exact TP4 TTNN component path"
    "Fix fused MoE source buffer double counting"
    "Guard all-gather scatter state initialization"
    "Preserve-ring-connections-in-all-gather-endpoint-guard"
    "Fix all-gather endpoint no-target connection access"
    "Clear-fabric-router-packet-tags-on-teardown"
    "#23023: Bind BF4 cache loads to verified file descriptors"
    "#23023: Reject lexical aliases for descriptor loads"
    "#23023: Fail closed BF4 cache shape and cleanup"
    "#23023: Bind BF4 tensorbin payload and exact types"
)

die() { printf 'run_qwen38_chat_server: %s\n' "$*" >&2; exit 2; }
usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

profile= instance=0 checkpoint= cache_root= allocated_context=32768 mtp= port=8000 host=0.0.0.0
acceptance_prompts= require_json_96= serve_seconds= python= validate_only= prefill_mode=chunked
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile=${2-}; shift 2 ;;
        --instance) instance=${2-}; shift 2 ;;
        --checkpoint) checkpoint=${2-}; shift 2 ;;
        --cache-root) cache_root=${2-}; shift 2 ;;
        --allocated-context) allocated_context=${2-}; shift 2 ;;
        --mtp) mtp=${2-}; shift 2 ;;
        --port) port=${2-}; shift 2 ;;
        --host) host=${2-}; shift 2 ;;
        --acceptance-prompts) acceptance_prompts=${2-}; shift 2 ;;
        --require-json-96) require_json_96=1; shift ;;
        --serve-seconds) serve_seconds=${2-}; shift 2 ;;
        --python) python=${2-}; shift 2 ;;
        --prefill-mode) prefill_mode=${2-}; shift 2 ;;
        --validate-only) validate_only=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
[[ -n "$profile" && -n "$checkpoint" && -n "$cache_root" ]] || usage
[[ -d "$checkpoint" ]] || die "--checkpoint $checkpoint is not a directory"
[[ "$allocated_context" =~ ^(32768|65536|131072|262144)$ ]] || die "--allocated-context must be 32768, 65536, 131072 or 262144"
[[ "$instance" =~ ^[01]$ ]] || die "--instance must be 0 or 1"
[[ -n "${TT_METAL_HOME:-}" && -d "$TT_METAL_HOME" ]] || die "TT_METAL_HOME must name a built tt-metal checkout"
python=${python:-$TT_METAL_HOME/python_env/bin/python}
[[ -x "$python" ]] || die "no python at $python (pass --python)"

case "$profile" in
    tt-quietbox)
        [[ "$instance" == 0 ]] || die "--instance applies to --profile qb2"
        hardware_profile=tt-quietbox visible_devices=0,1,2,3
        descriptor="$HERE/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto" ;;
    qb2)
        printf 'run_qwen38_chat_server: the qb2 profile is UNTESTED (no QuietBox 2 was available); the first run prints the derived route to pin\n' >&2
        if [[ "$instance" == 0 ]]; then hardware_profile=tt-quietbox-2 visible_devices=0,1,2,3; else hardware_profile=tt-quietbox-2-instance-1 visible_devices=4,5,6,7; fi
        descriptor="$HERE/qb2_p300_1x4_line_mesh_graph_descriptor.textproto" ;;
    *) die "--profile must be tt-quietbox or qb2" ;;
esac

# -- the runtime: the checkout's commit against the expected base and patches -----------------------------------------
head=$(git -C "$TT_METAL_HOME" rev-parse HEAD 2>/dev/null || echo unknown)
missing=()
if [[ "$head" == unknown ]]; then
    missing=("${REQUIRED_PATCHES[@]}")
else
    git -C "$TT_METAL_HOME" merge-base --is-ancestor "$EXPECTED_TT_METAL_BASE" HEAD 2>/dev/null \
        || printf 'run_qwen38_chat_server: tt-metal %s does not descend from the expected base %s\n' "$head" "$EXPECTED_TT_METAL_BASE" >&2
    for subject in "${REQUIRED_PATCHES[@]}"; do
        git -C "$TT_METAL_HOME" log --format=%s "$EXPECTED_TT_METAL_BASE..HEAD" 2>/dev/null | grep -qF -- "$subject" || missing+=("$subject")
    done
fi
if [[ ${#missing[@]} -gt 0 ]]; then
    {
        printf 'run_qwen38_chat_server: tt-metal at %s (%s) lacks these runtime patches (expected base %s):\n' "$TT_METAL_HOME" "$head" "$EXPECTED_TT_METAL_BASE"
        printf '    %s\n' "${missing[@]}"
        printf 'the server checks the runtime it imports and refuses one it does not admit\n'
    } >&2
fi
extension=$("$python" -c 'import ttnn._ttnn as m; print(m.__file__)') || die "$python cannot import ttnn"
extension_sha=$(sha256sum "$extension" | cut -c1-64)
source_head=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || die "the model tree must be a git checkout")
source_tree=$(git -C "$REPO_ROOT" rev-parse 'HEAD^{tree}')

# -- caches, run directory, environment ---------------------------------------------------------------------------------
label="c$allocated_context-$hardware_profile"
caches="$cache_root/caches/$label"
run_dir="$cache_root/runs/q38-chat-server-$hardware_profile-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$caches/components" "$caches/bf4-scratch" "$caches/model-io" "$cache_root/jit-cache/$label" "$run_dir/tmp"

export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_MESH_GRAPH_DESC_PATH="$descriptor"
export TT_VISIBLE_DEVICES="$visible_devices"
export QWEN38_HARDWARE_MODE=diagnostic_non_promoting
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_CACHE="$cache_root/jit-cache/$label"
export TT_METAL_LOGS_PATH="$run_dir/metal-logs"
export TMPDIR="$run_dir/tmp"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/ttnn${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

args=(
    --hardware-profile "$hardware_profile"
    --checkpoint "$checkpoint"
    --component-cache-root "$caches/components"
    --routed-bf4-scratch-root "$caches/bf4-scratch"
    --model-io-cache-root "$caches/model-io"
    --phase-log "$run_dir/phase-markers.jsonl"
    --evidence "$run_dir"
    --runtime-extension "$extension"
    --runtime-sha256 "$extension_sha"
    --tt-metal-sha "$head"
    --source-head "$source_head"
    --source-tree "$source_tree"
    --allocated-context "$allocated_context"
    --prefill-mode "$prefill_mode"
    --host "$host" --port "$port"
)
if [[ -n "$mtp" ]]; then
    grep -qF -- '"--mtp"' "$SERVER" || die "this server carries no MTP drafting path (--mtp); the q38-serve-mtp branch does"
    args+=(--mtp "$mtp")
fi
[[ -z "$acceptance_prompts" ]] || args+=(--acceptance-prompts "$acceptance_prompts")
[[ -z "$require_json_96" ]] || args+=(--require-json-96)
[[ -z "$validate_only" ]] || args+=(--validate-only)

printf 'run_qwen38_chat_server: profile %s devices %s context %s run %s\n' "$hardware_profile" "$visible_devices" "$allocated_context" "$run_dir" >&2
if [[ -n "$serve_seconds" ]]; then
    exec timeout --signal=TERM "$serve_seconds" "$python" "$SERVER" "${args[@]}"
fi
exec "$python" "$SERVER" "${args[@]}"
