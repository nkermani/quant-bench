{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "quant-bench-2026";

  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.numpy
    python311Packages.pandas
    python311Packages.pyyaml
    git
    cudaPackages.cudatoolkit
  ];

  shellHook = ''
    echo "Quant-Bench 2026 - Quantization Benchmarking Environment"
    echo "Python: $(python --version)"
    if [ ! -d "venv" ]; then
      echo "Creating virtual environment..."
      python -m venv venv
    fi
    source venv/bin/activate
    echo "Virtual environment activated."
    if [ -f "requirements.txt" ]; then
      pip install -r requirements.txt
    fi
    echo ""
    echo "To install the project in editable mode:"
    echo "  pip install -e .[quantization,eval,bench,dev]"
  '';
}
