import torch
import torch.nn as nn
import time

class MASAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, sram_limit_bytes: int = 65536):
        """
        Memory-Aware Stream (MAS) Attention Module.
        Subclasses nn.Module.
        
        Designed to process massive sequence lengths on resource-constrained edge NPUs
        by streaming attention blocks through a small on-chip SRAM buffer.
        """
        super(MASAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sram_limit_bytes = sram_limit_bytes
        
        # Projection Layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, use_mixed_precision: bool = True) -> torch.Tensor:
        """
        Forward pass with custom block-based stream processing.
        
        Arguments:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            use_mixed_precision: Run with FP16 autocast to optimize edge registers
        """
        batch_size, seq_len, _ = x.size()
        
        # Auto-detect hardware device for execution
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        self.to(device)
        
        # Define run context based on precision flag
        precision_ctx = torch.cuda.amp.autocast(enabled=use_mixed_precision) if device == "cuda" else torch.autocast(device_type="cpu", enabled=use_mixed_precision)
        
        with precision_ctx:
            # 1. Project Q, K, V
            Q = self.q_proj(x) # [B, S, D]
            K = self.k_proj(x)
            V = self.v_proj(x)
            
            # Reshape for multi-head attention: [B, H, S, d_k]
            Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # 2. Block-Wise Streaming Attention Loop
            # Calculates peak SRAM activation bounds to avoid memory thrashing.
            # Block size B_q and B_k are dynamically calculated based on the hardware SRAM limit.
            element_size = 2 if use_mixed_precision else 4 # FP16 vs FP32 size in bytes
            block_size = max(8, self.sram_limit_bytes // (self.d_k * element_size * 2))
            
            output = torch.zeros_like(Q) # [B, H, S, d_k]
            
            # Process queries in blocks to maintain constant peak memory overhead
            for q_start in range(0, seq_len, block_size):
                q_end = min(q_start + block_size, seq_len)
                Q_block = Q[:, :, q_start:q_end, :] # [B, H, B_q, d_k]
                
                # Accumulators for this query block
                accum_num = torch.zeros(batch_size, self.n_heads, q_end - q_start, self.d_k, device=device)
                accum_den = torch.zeros(batch_size, self.n_heads, q_end - q_start, 1, device=device)
                
                # Stream keys and values through memory buffers
                for k_start in range(0, seq_len, block_size):
                    k_end = min(k_start + block_size, seq_len)
                    K_block = K[:, :, k_start:k_end, :] # [B, H, B_k, d_k]
                    V_block = V[:, :, k_start:k_end, :] # [B, H, B_k, d_k]
                    
                    # Compute raw attention scores for this block
                    scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (self.d_k ** 0.5) # [B, H, B_q, B_k]
                    
                    # Log-sum-exp numerical stability adjustment
                    exp_scores = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True)[0])
                    
                    # Accumulate numerator and denominator
                    accum_num += torch.matmul(exp_scores, V_block) # [B, H, B_q, d_k]
                    accum_den += torch.sum(exp_scores, dim=-1, keepdim=True) # [B, H, B_q, 1]
                
                # Normalize and assign to output
                output[:, :, q_start:q_end, :] = accum_num / (accum_den + 1e-9)
            
            # 3. Final projection
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.out_proj(output)

if __name__ == "__main__":
    print("[MAS-Attention Experiment] Instantiating PyTorch Model...")
    
    # Target scale modeling: 128 embedding dimension, 4 attention heads
    model = MASAttention(d_model=128, n_heads=4, sram_limit_bytes=4096)
    
    # Generate dummy input tensor: [batch_size=2, seq_len=64, d_model=128]
    dummy_input = torch.randn(2, 64, 128)
    
    # Run test execution
    print("[MAS-Attention Experiment] Running mixed-precision (FP16) forward pass...")
    start_time = time.time()
    result = model(dummy_input, use_mixed_precision=True)
    latency = (time.time() - start_time) * 1000
    
    print(f" -> Execution complete. Shape: {list(result.shape)}")
    print(f" -> Latency: {latency:.3f} ms")
