#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import sys
from pathlib import Path

import pytest

import tracy
from tracy import __main__ as tracy_cli
from tracy import process_ops_logs


# class for mocking creation of npe data
class _FakeNpeResult:
    def __init__(self, noc_util, mcast_noc_util, dram_bw_util, cong_impact):
        self.overall_avg_link_util = noc_util
        self.overall_avg_mcast_write_link_util = mcast_noc_util
        self.dram_bw_util = dram_bw_util
        self._cong_impact = cong_impact

    def getCongestionImpact(self):
        return self._cong_impact


class _FakeNpeDatapoint:
    def __init__(self, result):
        self.result = result


class _FakeNpeStats:
    def __init__(self, op_to_result):
        self._op_to_result = op_to_result

    def getDatapointByID(self, op_id):
        result = self._op_to_result.get(op_id)
        if result is None:
            return None
        return _FakeNpeDatapoint(result)


@pytest.mark.skip(reason="Missing mock for device log file; needs fix to properly stub _enrich_ops_from_device_logs")
def test_append_device_data_populates_multicast_noc_util(monkeypatch, tmp_path):
    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
        }
    }
    trace_replays = {}

    fake_stats = _FakeNpeStats(
        {
            1: _FakeNpeResult(
                noc_util=91.24,
                mcast_noc_util=44.44,
                dram_bw_util=38.88,
                cong_impact=12.345,
            )
        }
    )
    monkeypatch.setattr(process_ops_logs, "analyzeNoCTraces", lambda _log_folder: fake_stats)

    process_ops_logs.append_device_data(
        ops=ops,
        traceReplays=trace_replays,
        logFolder=tmp_path,
        analyze_noc_traces=True,
        device_analysis_types=[],
    )

    assert ops[1]["NOC UTIL (%)"] == 91.2
    assert ops[1]["MULTICAST NOC UTIL (%)"] == 44.4
    assert ops[1]["DRAM BW UTIL (%)"] == 38.9
    assert ops[1]["NPE CONG IMPACT (%)"] == 12.35


def test_generate_reports_writes_sub_device_id_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    device_log = log_folder / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,,,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":1;"sub_device_manager_id":7}',
            ]
        )
    )

    ops = {
        42: {
            "global_call_count": 42,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
        }
    }

    sub_device_lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    host_ops_by_device = {0: [ops[42].copy()]}
    process_ops_logs.attach_sub_device_ids_to_ops(host_ops_by_device, sub_device_lookup)
    ops[42]["sub_device_id"] = host_ops_by_device[0][0]["sub_device_id"]

    process_ops_logs.generate_reports(
        ops=ops,
        deviceOps={},
        traceOps={},
        signposts={},
        logFolder=log_folder,
        outputFolder=report_folder,
        date=False,
        nameAppend=None,
    )

    report_csv = Path(report_folder) / "ops_perf_results.csv"
    assert report_csv.is_file()

    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "SUB DEVICE ID" in reader.fieldnames
        assert row["SUB DEVICE ID"] == "1"
        assert "SUB DEVICE MANAGER ID" not in reader.fieldnames


def test_get_op_sub_device_lookup_key_prefers_device_perf_row():
    op = {
        "global_call_count": 1,
        "device_id": 0,
        "metal_trace_id": None,
        "_device_perf_row": {
            "GLOBAL CALL COUNT": 2048,
            "DEVICE ID": 0,
            "METAL TRACE ID": "",
            "METAL TRACE REPLAY SESSION ID": "",
        },
    }
    assert process_ops_logs.get_op_sub_device_lookup_key(op, 0) == (0, 2048, -1, -1)


def test_build_sub_device_id_lookup_ignores_manager_id_only_rows(tmp_path):
    device_log = tmp_path / "profile_log_device.csv"
    device_log.write_text(
        "\n".join(
            [
                "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64",
                "PCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data",
                '0,0,0,BRISC,1,100,0,42,0,1,BRISC-FW,ZONE_START,1,k.cpp,{"sub_device_id":0;"sub_device_manager_id":7}',
            ]
        )
    )

    lookup = process_ops_logs.build_sub_device_id_lookup_from_device_csv(device_log)
    assert lookup[(0, 42, 0, 1)] == 0


def test_generate_reports_writes_multicast_noc_util_column(tmp_path):
    log_folder = tmp_path / "logs"
    report_folder = tmp_path / "reports"
    log_folder.mkdir(parents=True, exist_ok=True)

    ops = {
        1: {
            "global_call_count": 1,
            "device_id": 0,
            "host_time": {"ns_since_start": 10, "exec_time_ns": 20},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
            "NOC UTIL (%)": 50.0,
            "MULTICAST NOC UTIL (%)": 25.0,
            "DRAM BW UTIL (%)": 75.0,
            "NPE CONG IMPACT (%)": 1.25,
        }
    }

    process_ops_logs.generate_reports(
        ops=ops,
        deviceOps={},
        traceOps={},
        signposts={},
        logFolder=log_folder,
        outputFolder=report_folder,
        date=False,
        nameAppend=None,
    )

    report_csv = Path(report_folder) / "ops_perf_results.csv"
    assert report_csv.is_file()

    with report_csv.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)
        assert "MULTICAST NOC UTIL (%)" in reader.fieldnames
        assert row["MULTICAST NOC UTIL (%)"] == "25.0"


def _host_only_log_directory(tmp_path):
    log_folder = tmp_path / ".logs"
    log_folder.mkdir()
    for name in (
        process_ops_logs.TRACY_FILE_NAME,
        process_ops_logs.TRACY_OPS_TIMES_FILE_NAME,
        process_ops_logs.TRACY_OPS_DATA_FILE_NAME,
    ):
        (log_folder / name).write_text(f"fresh {name}\n")
    return log_folder


def _host_only_op():
    return {
        7: {
            "op_code": "ttnn::multiply",
            "op_type": "tt_dnn_device",
            "global_call_count": 7,
            "device_id": 0,
            "host_time": {"ns_since_start": 100, "exec_time_ns": 25},
            "metal_trace_id": None,
            "input_tensors": [],
            "output_tensors": [],
            "op_hash": 1234,
            "program_cache_hit": True,
        }
    }


def test_process_ops_host_only_emits_flat_labeled_report_without_device_join(monkeypatch, tmp_path):
    _host_only_log_directory(tmp_path)
    monkeypatch.setattr(
        process_ops_logs,
        "import_tracy_op_logs",
        lambda _log_folder: (_host_only_op(), {}, {}),
    )

    process_ops_logs.process_ops(tmp_path, None, False, host_only=True)

    report_folder = tmp_path / "reports"
    assert sorted(path.name for path in report_folder.iterdir()) == [
        "host_only_report.json",
        "ops_perf_results.csv",
        process_ops_logs.TRACY_FILE_NAME,
    ]
    with (report_folder / "ops_perf_results.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        row = next(reader)
        assert row["REPORT MODE"] == "host_only_no_device_join"
        assert row["DEVICE ID"] == "0"
        for field in reader.fieldnames:
            if field.startswith("DEVICE ") and field not in {"DEVICE ID", "DEVICE ARCH"}:
                assert row[field] == ""
        assert row["OP TO OP LATENCY [ns]"] == ""
    manifest = json.loads((report_folder / "host_only_report.json").read_text())
    assert manifest == {
        "schema": "ttnn-host-only-ops-report/v1",
        "report_mode": "host_only_no_device_join",
        "device_data_joined": False,
        "stale_device_artifacts_rejected": True,
        "unexpected_profiler_artifacts_rejected": True,
        "allowed_log_artifacts": sorted(
            {
                process_ops_logs.TRACY_FILE_NAME,
                process_ops_logs.TRACY_OPS_TIMES_FILE_NAME,
                process_ops_logs.TRACY_OPS_DATA_FILE_NAME,
            }
        ),
        "operation_count": 1,
        "signpost_count": 0,
    }


@pytest.mark.parametrize("unexpected_name", ["profile_log_device.csv", "cpp_device_perf_report.csv", "other.log"])
def test_process_ops_host_only_rejects_every_unexpected_log_artifact(monkeypatch, tmp_path, unexpected_name):
    log_folder = _host_only_log_directory(tmp_path)
    (log_folder / unexpected_name).write_text("stale device or unrelated data\n")
    monkeypatch.setattr(
        process_ops_logs,
        "import_tracy_op_logs",
        lambda _log_folder: (_host_only_op(), {}, {}),
    )

    with pytest.raises(RuntimeError, match="unexpected profiler artifacts"):
        process_ops_logs.process_ops(tmp_path, None, False, host_only=True)


def test_process_ops_host_only_rejects_nonempty_report_directory(monkeypatch, tmp_path):
    _host_only_log_directory(tmp_path)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "stale.csv").write_text("stale\n")
    monkeypatch.setattr(
        process_ops_logs,
        "import_tracy_op_logs",
        lambda _log_folder: (_host_only_op(), {}, {}),
    )

    with pytest.raises(RuntimeError, match="nonempty pre-existing report"):
        process_ops_logs.process_ops(tmp_path, None, False, host_only=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device_only": True},
        {"analyze_noc_traces": True},
        {"device_analysis_types": ("device_kernel_duration",)},
        {"force_legacy_device_logs": True},
    ],
)
def test_process_ops_host_only_rejects_device_and_noc_modes(tmp_path, kwargs):
    with pytest.raises(ValueError, match="host-only"):
        process_ops_logs.process_ops(tmp_path, None, False, host_only=True, **kwargs)


@pytest.mark.parametrize(("name_append", "date"), [("named", False), (None, True)])
def test_process_ops_host_only_rejects_nonflat_output_layout(tmp_path, name_append, date):
    with pytest.raises(ValueError, match="flat undated unnamed"):
        process_ops_logs.process_ops(tmp_path, name_append, date, host_only=True)


def test_generate_report_host_only_refuses_preexisting_export_symlink(tmp_path):
    log_folder = tmp_path / ".logs"
    log_folder.mkdir()
    (log_folder / process_ops_logs.TRACY_FILE_NAME).write_text("fresh trace\n")
    stale = tmp_path / "stale.csv"
    stale.write_text("stale\n")
    (log_folder / process_ops_logs.TRACY_OPS_TIMES_FILE_NAME).symlink_to(stale)

    with pytest.raises(RuntimeError, match="exactly one nonempty regular nonsymlink Tracy capture"):
        tracy.generate_report(tmp_path, tmp_path / "bin", None, None, host_only=True)


@pytest.mark.parametrize(
    "conflict",
    [
        "--collect-noc-traces",
        "--profile-dispatch-cores",
        "--dump-device-data-mid-run",
        "--disable-device-data-dump-to-files",
    ],
)
def test_process_logs_only_host_only_rejects_capture_conflicts_before_generate_report(
    monkeypatch, conflict
):
    generated = []
    monkeypatch.setattr(sys, "argv", ["python", "--process-logs-only", "--no-device", conflict])
    monkeypatch.setattr(tracy_cli, "generate_report", lambda *_args, **_kwargs: generated.append(True))

    with pytest.raises(SystemExit) as raised:
        tracy_cli.main()

    assert raised.value.code == 2
    assert generated == []
