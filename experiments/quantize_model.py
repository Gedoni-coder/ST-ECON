import os
import torch
import torch.nn as nn

class LinearAttentionRegressor(nn.Module):
    def __init__(self, d_model: int = 128):
        """
        Simple linear prediction layer mapping GNN stress vectors to asset risk metrics.
        This target model will be quantized from FP32 to INT8 to satisfy low-power edge deployment requirements.
        """
        super(LinearAttentionRegressor, self).__init__()
        # Standard FP32 operators
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = nn.Linear(d_model, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 32)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize inputs, compute, and dequantize outputs
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

def run_quantization():
    print("[Quantization Module] Instantiating FP32 base model...")
    model_fp32 = LinearAttentionRegressor()
    model_fp32.eval() # Quantization requires evaluation mode

    # 1. Configure Static Quantization
    # Uses fbgemm for x86 CPUs or qnnpack for ARM edge devices
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    # 2. Prepare model for calibration
    print("[Quantization Module] Preparing model for static calibration...")
    model_prepared = torch.ao.quantization.prepare(model_fp32, inplace=False)

    # 3. Calibrate with representative dummy data
    print("[Quantization Module] Running calibration passes...")
    calibration_data = torch.randn(100, 128)
    with torch.inference_mode():
        for _ in range(10):
            _ = model_prepared(calibration_data)

    # 4. Convert model to quantized INT8 format
    print("[Quantization Module] Compiling FP32 weights to quantized INT8...")
    model_int8 = torch.ao.quantization.convert(model_prepared, inplace=False)

    # 5. Save and compare model footprints on disk
    os.makedirs("data", exist_ok=True)
    fp32_path = os.path.join("data", "model_fp32.pt")
    int8_path = os.path.join("data", "model_int8.pt")

    torch.save(model_fp32.state_dict(), fp32_path)
    torch.save(model_int8.state_dict(), int8_path)

    fp32_size = os.path.getsize(fp32_path)
    int8_size = os.path.getsize(int8_path)

    print("\n============================================================")
    print("QUANTIZATION COMPARISON REPORT")
    print("============================================================")
    print(f"FP32 Model Size: {fp32_size} bytes")
    print(f"INT8 Model Size: {int8_size} bytes")
    print(f"Memory Reduction: {((fp32_size - int8_size) / fp32_size) * 100:.2f}%")
    print("============================================================\n")

if __name__ == "__main__":
    run_quantization()
