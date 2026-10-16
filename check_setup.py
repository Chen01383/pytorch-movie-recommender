"""Check Python, PyTorch and an actual matrix multiplication on the selected device."""
import sys

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
try:
    import torch
except ImportError:
    print("PyTorch is missing. Install requirements.txt using this interpreter.")
    raise SystemExit(1)

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
try:
    matrix = torch.randn(128, 128, device=device)
    value = (matrix @ matrix.T).mean().item()
except RuntimeError as error:
    print("Device computation failed:", error)
    print("Check the PyTorch CUDA wheel and driver; use --device cpu while troubleshooting.")
    raise SystemExit(1)
print(f"Matrix multiplication OK on {device}: {value:.6f}")
