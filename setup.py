from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="quant-bench-2026",
    version="0.1.0",
    description="Comparative performance profiling of LLM quantization (FP8, INT8, AWQ, GPTQ) using Marlin kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nkermani/quant-bench-2026",
    author="Nathan Kermani",
    author_email="nkermani@42lyon.fr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "quantization": [
            "autoawq>=0.2.5",
            "auto-gptq>=0.7.1",
            "bitsandbytes>=0.43.0",
            "vllm>=0.7.0",
        ],
        "eval": [
            "lm-evaluation-harness>=0.4.3",
            "deepeval>=0.9.0",
            "huggingface-hub>=0.23.0",
        ],
        "bench": [
            "psutil>=5.9.0",
            "nvidia-ml-py3>=7.352.0",
            "tqdm>=4.66.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quant-bench-quantize=quant_bench.quantize:main",
            "quant-bench-bench=quant_bench.bench:main",
            "quant-bench-eval=quant_bench.eval:main",
        ],
    },
)
