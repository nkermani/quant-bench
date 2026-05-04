"""Metrics calculation utilities for quantization evaluation."""

import numpy as np

def calculate_perplexity(logits, labels):
    """Calculate perplexity from model logits and labels."""
    # Simplified perplexity calculation
    # In practice, use proper tokenizer and model outputs
    return np.exp(np.mean(-np.log(np.array(logits) + 1e-10)))

def calculate_accuracy(predictions, ground_truth):
    """Calculate accuracy for benchmark tasks."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    return sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)

def calculate_compression_ratio(original_bits, quantized_bits):
    """Calculate memory compression ratio."""
    return original_bits / quantized_bits

def calculate_quality_score(accuracy_pct, throughput_lift):
    """Calculate quality-to-throughput Pareto score."""
    return (accuracy_pct / 100) * throughput_lift
