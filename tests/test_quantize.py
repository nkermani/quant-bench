"""Tests for quantization module."""

import pytest
from quant_bench.quantize import quantize_model
from pathlib import Path
import tempfile

def test_quantize_model_creates_output():
    """Test that quantization creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        quantize_model("test-model", "awq", 4, tmpdir)
        output_path = Path(tmpdir) / "test-model-awq-4bit"
        assert output_path.exists()

def test_quantize_invalid_format():
    """Test that invalid format raises error."""
    with pytest.raises(ValueError, match="Unsupported format"):
        quantize_model("test-model", "invalid", 4)

def test_quantize_fp8_format():
    """Test FP8 quantization format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        quantize_model("test-model", "fp8", 8, tmpdir)
        output_path = Path(tmpdir) / "test-model-fp8-8bit"
        assert output_path.exists()
