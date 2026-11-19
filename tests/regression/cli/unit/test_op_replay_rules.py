# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Pure CLI, discovery, and constant rules for operator replay."""

import importlib
import sys
from pathlib import Path

import pytest

from tools.perf_data_collection.op_replay import common


OP_REPLAY_DIR = Path(__file__).resolve().parents[4] / "tools" / "perf_data_collection" / "op_replay"
if str(OP_REPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(OP_REPLAY_DIR))

dispatch_ffn = importlib.import_module("DispatchFFNCombine_run")
run_all_op = importlib.import_module("run_all_op")
op_common = importlib.import_module("common")


class TestCommonModule:
    def test_module_imports_without_npu(self):
        assert common.torch is None
        assert common.torch_npu is None

    def test_data_dir_points_to_profiling_database(self):
        assert common.DATA_DIR.parts[-2:] == ("profiling_database", "data")

    def test_get_replay_repeat_count_uses_cli_value_or_default(self):
        assert op_common.get_replay_repeat_count(7) == 7
        assert op_common.get_replay_repeat_count(None) == op_common.DEFAULT_REPLAY_REPEAT_COUNT

    @pytest.mark.parametrize("repeat_count", [0, -1])
    def test_get_replay_repeat_count_rejects_non_positive_values(self, repeat_count):
        with pytest.raises(ValueError, match="--repeat-count must be positive"):
            op_common.get_replay_repeat_count(repeat_count)

    def test_build_host_tensor_uses_empty_for_float_dtypes(self, monkeypatch):
        class FakeTorch:
            bool = object()
            int32 = object()
            int64 = object()
            float16 = object()
            bfloat16 = object()
            float32 = object()
            float64 = object()

            def __init__(self):
                self.empty_calls = []
                self.randint_calls = []

            def empty(self, shape, dtype):
                self.empty_calls.append((shape, dtype))
                return ("empty", shape, dtype)

            def randint(self, *args, **kwargs):
                self.randint_calls.append((args, kwargs))
                return ("randint", args, kwargs)

        fake_torch = FakeTorch()
        monkeypatch.setattr(op_common, "get_runtime_modules", lambda: (fake_torch, None))

        tensor = op_common.build_host_tensor((2, 3), fake_torch.bfloat16)

        assert tensor == ("empty", (2, 3), fake_torch.bfloat16)
        assert fake_torch.empty_calls == [((2, 3), fake_torch.bfloat16)]
        assert fake_torch.randint_calls == []


class TestRunAllOp:
    def test_argparser_parses_replay_options(self, tmp_path):
        args = run_all_op.build_argparser().parse_args(
            [
                "--database-path",
                str(tmp_path),
                "--device",
                "TEST_DEVICE",
                "--update-mode",
                "missing-only",
                "--execution-mode",
                "subprocess",
                "--ops",
                "MatMulV2",
                "PadV3",
                "--continue-on-error",
            ]
        )

        assert args.database_path == tmp_path
        assert args.device == "TEST_DEVICE"
        assert args.update_mode == "missing-only"
        assert args.execution_mode == "subprocess"
        assert args.ops == ["MatMulV2", "PadV3"]
        assert args.continue_on_error is True

    def test_discover_run_scripts(self):
        scripts = run_all_op.discover_run_scripts()
        assert scripts
        assert run_all_op.SELF_NAME not in [script.name for script in scripts]

    def test_filter_run_scripts_exact_match(self):
        scripts = [Path("MatMulV2_run.py"), Path("PadV3_run.py"), Path("RmsNorm_run.py")]
        assert run_all_op.filter_run_scripts(scripts, {"MatMulV2"}) == [Path("MatMulV2_run.py")]

    def test_filter_run_scripts_none_returns_all(self):
        scripts = [Path("MatMulV2_run.py"), Path("PadV3_run.py")]
        assert run_all_op.filter_run_scripts(scripts, None) == scripts

    def test_get_csv_name(self):
        assert run_all_op.get_csv_name(Path("MatMulV2_run.py")) == "MatMulV2.csv"
        assert run_all_op.get_csv_name(Path("PadV3_run.py")) == "PadV3.csv"

    def test_has_operator_csv(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "MatMulV2.csv").write_text("x", encoding="utf-8")
        assert run_all_op.has_operator_csv(tmp_path, "MatMulV2.csv")
        assert not run_all_op.has_operator_csv(tmp_path, "Nonexistent.csv")


class TestDispatchFfnConstants:
    def test_default_ep_size(self):
        assert dispatch_ffn.DEFAULT_EP_SIZE == 16
        assert dispatch_ffn.DEFAULT_DFC_REPEAT_COUNT > 0

    def test_default_max_output_size(self):
        assert dispatch_ffn.DEFAULT_DFC_MAX_OUTPUT_SIZE > 0

    def test_build_argparser(self):
        parser = dispatch_ffn.build_standard_argparser(
            description="test",
            usage_examples=["python test.py"],
            version_help="test",
        )
        assert parser.parse_args(["--database-path", "test_dir"]).database_path == Path("test_dir")
