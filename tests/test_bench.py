"""Tests for benchmarking module."""

import pytest
from quant_bench.bench import run_benchmark
from pathlib import Path
import tempfile

def test_run_benchmark_invalid_model():
    """Test that invalid model path raises error."""
    with pytest.raises(SystemExit):
        run_benchmark("/nonexistent/path", ["mmlu"], [1])

def test_run_benchmark_creates_output():
    """Test benchmark execution with mock model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test-model"
        model_path.mkdir()
        metrics = run_benchmark(str(model_path), ["mmlu"], [1, 4])
        assert isinstance(metrics, dict)
        assert "throughput_tokens_per_sec" in metrics
