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
    model_fp32.eval() # Static Quantization requires evaluation mode

    # ── 1. STATIC POST-TRAINING QUANTIZATION (PTQ) ──
    # Uses fbgemm for x86 CPUs or qnnpack for ARM edge devices
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    
    print("[Quantization Module] Preparing model for static calibration...")
    model_prepared = torch.ao.quantization.prepare(model_fp32, inplace=False)

    print("[Quantization Module] Running static calibration passes...")
    calibration_data = torch.randn(100, 128)
    with torch.inference_mode():
        for _ in range(10):
            _ = model_prepared(calibration_data)

    print("[Quantization Module] Converting static model to quantized INT8...")
    model_int8_static = torch.ao.quantization.convert(model_prepared, inplace=False)

    # ── 2. DYNAMIC QUANTIZATION-AWARE TRAINING (QAT) ──
    print("\n[Quantization Module] Instantiating model for Quantization-Aware Training (QAT)...")
    model_qat = LinearAttentionRegressor()
    model_qat.train() # QAT requires training mode for backprop
    
    # Configure QAT defaults
    model_qat.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare model for QAT by inserting fake-quantization operators
    print("[Quantization Module] Preparing model for QAT by inserting fake-quant nodes...")
    model_prepared_qat = torch.ao.quantization.prepare_qat(model_qat, inplace=False)
    
    # Simulated economic stress data training loop
    print("[Quantization Module] Running QAT optimization epochs under simulated economic stress distributions...")
    optimizer = torch.optim.SGD(model_prepared_qat.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    
    # Run 10 epochs of QAT
    for epoch in range(10):
        # Generate representative stress inputs (heavy-tailed distributions)
        inputs = torch.randn(32, 128) * 1.5 # Inject volatility scaling
        targets = torch.randn(32, 32)
        
        optimizer.zero_grad()
        outputs = model_prepared_qat(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    print(f" -> QAT Epoch 10 Complete. Final Loss: {loss.item():.4f}")
    
    # Convert QAT model to quantized INT8
    print("[Quantization Module] Converting QAT model to evaluation mode and converting to INT8...")
    model_prepared_qat.eval()
    model_int8_qat = torch.ao.quantization.convert(model_prepared_qat, inplace=False)

    # ── 3. SAVE AND COMPARE FOOTPRINTS ──
    os.makedirs("data", exist_ok=True)
    fp32_path = os.path.join("data", "model_fp32.pt")
    int8_static_path = os.path.join("data", "model_int8_static.pt")
    int8_qat_path = os.path.join("data", "model_int8_qat.pt")

    torch.save(model_fp32.state_dict(), fp32_path)
    torch.save(model_int8_static.state_dict(), int8_static_path)
    torch.save(model_int8_qat.state_dict(), int8_qat_path)

    fp32_size = os.path.getsize(fp32_path)
    int8_static_size = os.path.getsize(int8_static_path)
    int8_qat_size = os.path.getsize(int8_qat_path)

    print("\n============================================================")
    print("QUANTIZATION MASTER COMPARISON REPORT")
    print("============================================================")
    print(f"Base FP32 Model Size:        {fp32_size} bytes")
    print(f"Static PTQ INT8 Model Size:  {int8_static_size} bytes (Reduction: {((fp32_size - int8_static_size) / fp32_size) * 100:.2f}%)")
    print(f"Dynamic QAT INT8 Model Size: {int8_qat_size} bytes (Reduction: {((fp32_size - int8_qat_size) / fp32_size) * 100:.2f}%)")
    print("============================================================\n")

if __name__ == "__main__":
    run_quantization()
