"""
LLM Integration Layer for the Universal Engine.
Wires the TaCIE prompt output from the Orchestrator directly to a hosted LLM.

Supports three modes (in priority order):
  1. Ollama   — local model (Llama3, Mistral, etc.) at http://localhost:11434
  2. OpenAI   — cloud API (gpt-4o, gpt-3.5-turbo, etc.)
  3. Fallback — structured expert rule engine (no model needed, always works)

The platform is model-agnostic. Swap the backend by setting LLM_BACKEND env var.
"""

import os
import json
import httpx
import asyncio
from typing import Optional

# ── BACKEND SELECTION ──
BACKEND = os.environ.get("LLM_BACKEND", "fallback").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ── HUGGING FACE SETTINGS ──
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "")
HF_ADAPTER_ID = os.environ.get("HF_ADAPTER_ID", "")

# ── LLAMA.CPP GGUF SETTINGS ──
LLAMACPP_MODEL_PATH = os.environ.get("LLAMACPP_MODEL_PATH", "")
LLAMACPP_N_GPU_LAYERS = int(os.environ.get("LLAMACPP_N_GPU_LAYERS", "0"))  # Default to CPU execution
# Cap default threads to CPU count minus 1 to prevent core starvation
default_threads = max(1, (os.cpu_count() or 4) - 1)
LLAMACPP_THREADS = int(os.environ.get("LLAMACPP_THREADS", str(default_threads)))


class UniversalLLMClient:
    """
    Model-agnostic LLM client. Sends TaCIE prompts to whichever backend
    is configured and returns the advisory text.
    """
    def __init__(self):
        self._hf_model = None
        self._hf_tokenizer = None
        self._llamacpp_model = None
        self._executor_hf = None
        self._executor_llamacpp = None

    def shutdown(self):
        """Gracefully shuts down running executors on application exit."""
        if self._executor_hf:
            print("[LLM Client] Shutting down Hugging Face ThreadPoolExecutor...")
            self._executor_hf.shutdown(wait=True)
            self._executor_hf = None
        if self._executor_llamacpp:
            print("[LLM Client] Shutting down LlamaCpp ThreadPoolExecutor...")
            self._executor_llamacpp.shutdown(wait=True)
            self._executor_llamacpp = None

    async def generate(self, tacie_prompt: str, target_node: str,
                       threat_level: float, actions: dict) -> dict:
        """Routes the prompt to the correct backend."""
        if BACKEND == "ollama":
            return await self._call_ollama(tacie_prompt)
        elif BACKEND == "openai":
            return await self._call_openai(tacie_prompt)
        elif BACKEND == "huggingface":
            return await self._call_huggingface(tacie_prompt)
        elif BACKEND == "llamacpp":
            return await self._call_llamacpp(tacie_prompt)
        else:
            return self._structured_fallback(target_node, threat_level, actions)

    async def _call_ollama(self, prompt: str) -> dict:
        """Calls a locally hosted Ollama model (Llama3, Mistral, etc.)."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
                )
                r.raise_for_status()
                text = r.json().get("response", "")
                return {"advisory": text, "model": OLLAMA_MODEL, "backend": "ollama"}
        except Exception as e:
            return {"advisory": f"[Ollama Error: {e}] Falling back to rule engine.",
                    "backend": "ollama_error", "model": OLLAMA_MODEL}

    async def _call_openai(self, prompt: str) -> dict:
        """Calls OpenAI API (requires OPENAI_API_KEY env var)."""
        if not OPENAI_KEY:
            return {"advisory": "[No OPENAI_API_KEY set] Falling back to rule engine.",
                    "backend": "openai_error", "model": OPENAI_MODEL}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                    json={"model": OPENAI_MODEL,
                          "messages": [{"role": "user", "content": prompt}]}
                )
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                return {"advisory": text, "model": OPENAI_MODEL, "backend": "openai"}
        except Exception as e:
            return {"advisory": f"[OpenAI Error: {e}] Falling back to rule engine.",
                    "backend": "openai_error", "model": OPENAI_MODEL}

    async def _call_huggingface(self, prompt: str) -> dict:
        """Loads and executes a Hugging Face model/adapter locally via transformers."""
        if not HF_MODEL_ID:
            return {"advisory": "[HF_MODEL_ID not set] Please configure the model ID environment variable.",
                    "backend": "huggingface_error", "model": "none"}
        
        try:
            # Lazy import to avoid loading heavy modules if backend is not selected
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from concurrent.futures import ThreadPoolExecutor
            
            if self._hf_model is None:
                print(f"[Hugging Face] Loading tokenizer for {HF_MODEL_ID}...")
                self._hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
                
                print(f"[Hugging Face] Loading base model {HF_MODEL_ID} (auto device map)...")
                self._hf_model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                
                if HF_ADAPTER_ID:
                    from peft import PeftModel
                    print(f"[Hugging Face] Applying adapter weights from {HF_ADAPTER_ID}...")
                    self._hf_model = PeftModel.from_pretrained(self._hf_model, HF_ADAPTER_ID)
            
            if self._executor_hf is None:
                self._executor_hf = ThreadPoolExecutor(max_workers=1)
            
            # Non-blocking executor to keep async loop free during generation
            def _run_inference():
                inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
                with torch.no_grad():
                    outputs = self._hf_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True
                    )
                # Slice output to remove prompt part if model repeats it
                input_len = inputs["input_ids"].shape[1]
                advisory_ids = outputs[0][input_len:]
                return self._hf_tokenizer.decode(advisory_ids, skip_special_tokens=True)
            
            loop = asyncio.get_running_loop()
            advisory_text = await loop.run_in_executor(self._executor_hf, _run_inference)
            return {
                "advisory": advisory_text,
                "model": HF_MODEL_ID,
                "backend": "huggingface",
                "adapter": HF_ADAPTER_ID or "none"
            }
            
        except Exception as e:
            return {
                "advisory": f"[Hugging Face Loader Error: {e}] Check your environment / dependencies.",
                "backend": "huggingface_error",
                "model": HF_MODEL_ID
            }

    async def _call_llamacpp(self, prompt: str) -> dict:
        """Loads and executes a GGUF model locally using llama-cpp-python."""
        if not LLAMACPP_MODEL_PATH:
            return {
                "advisory": "[LLAMACPP_MODEL_PATH not set] Please point to a valid local .gguf file.",
                "backend": "llamacpp_error",
                "model": "none"
            }
            
        try:
            # Lazy import to avoid loading C++ libraries if backend not in use
            from llama_cpp import Llama
            from concurrent.futures import ThreadPoolExecutor
            
            if self._llamacpp_model is None:
                print(f"[LlamaCpp] Loading model from {LLAMACPP_MODEL_PATH}...")
                print(f"[LlamaCpp] Threads limit set to: {LLAMACPP_THREADS} cores (Prevents starvation)")
                print(f"[LlamaCpp] Requesting offload layers: {LLAMACPP_N_GPU_LAYERS}")
                
                # Check for GPU compiler flag warning
                if LLAMACPP_N_GPU_LAYERS > 0:
                    print("[LlamaCpp INFO] If GPU utilization is 0% during generation, make sure you compiled llama-cpp-python using CUDA bindings:")
                    print("  Powershell: $env:CMAKE_ARGS=\"-GGGUF_CUDA=on\"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
                
                self._llamacpp_model = Llama(
                    model_path=LLAMACPP_MODEL_PATH,
                    n_ctx=4096,
                    n_gpu_layers=LLAMACPP_N_GPU_LAYERS,
                    n_threads=LLAMACPP_THREADS
                )
            
            if self._executor_llamacpp is None:
                self._executor_llamacpp = ThreadPoolExecutor(max_workers=1)
                
            def _run_inference():
                res = self._llamacpp_model(
                    prompt,
                    max_tokens=256,
                    temperature=0.7,
                    stop=["[SYSTEM", "[USER"]
                )
                return res["choices"][0]["text"]
                
            loop = asyncio.get_running_loop()
            advisory_text = await loop.run_in_executor(self._executor_llamacpp, _run_inference)
            
            return {
                "advisory": advisory_text,
                "model": os.path.basename(LLAMACPP_MODEL_PATH),
                "backend": "llamacpp"
            }
            
        except Exception as e:
            return {
                "advisory": f"[LlamaCpp Loader Error: {e}] Ensure llama-cpp-python is installed and compiled correctly for Windows.",
                "backend": "llamacpp_error",
                "model": os.path.basename(LLAMACPP_MODEL_PATH) if LLAMACPP_MODEL_PATH else "none"
            }

    def _structured_fallback(self, target_node: str,
                              threat_level: float, actions: dict) -> dict:
        """
        Expert rule engine fallback. Always works with zero external dependencies.
        Generates structured advisory directly from the GNN threat data.
        """
        severity = (
            "CRITICAL" if threat_level > 4.0 else
            "HIGH" if threat_level > 2.0 else
            "MODERATE" if threat_level > 0.5 else "LOW"
        )
        best_action = max(actions, key=actions.get) if actions else "monitor_situation"
        best_score = round(actions.get(best_action, 1.0), 3)

        advisory = f"""
[STRUCTURED ADVISORY — Neuro-Symbolic Rule Engine]

TARGET: {target_node}
THREAT SEVERITY: {severity} (Score: {threat_level:.3f})

STAGE 1 — THREAT TRANSLATION:
The GNN propagation model has detected a {severity.lower()} threat level
at {target_node}. This indicates upstream disruptions are actively
flowing through the trade network toward this entity.

STAGE 2 — ACTION EVALUATION:
Based on counterfactual simulations across all available mitigations,
the mathematically optimal action is: '{best_action}'
(Learned confidence score: {best_score})

STAGE 3 — SELF-CRITIQUE:
While '{best_action}' offers the highest simulated threat reduction,
secondary risks may include increased operational overhead or capital
allocation constraints. Monitor adjacent nodes for cascading effects.

STAGE 4 — FINAL RECOMMENDATION:
Immediately implement '{best_action}' to counter the {severity} threat.
Continue streaming live signals to update threat propagation in real-time.
Submit outcome feedback after 7 days to improve future recommendations.
""".strip()

        return {
            "advisory": advisory,
            "model": "rule_engine_v1",
            "backend": "fallback",
            "severity": severity,
            "recommended_action": best_action
        }


# ── SINGLETON CLIENT ──
llm_client = UniversalLLMClient()


if __name__ == "__main__":
    # Test with simulated TaCIE prompt
    mock_prompt = """
[SYSTEM: UNIVERSAL NEURO-SYMBOLIC ADVISOR]
Target Entity: STORE
Current Threat Level: 3.800
Local Network: FACTORY affects STORE. WAREHOUSE distributes to STORE.
"""
    mock_actions = {"stockpile_goods": 1.37, "diversify_suppliers": 1.0}

    async def demo():
        result = await llm_client.generate(
            tacie_prompt=mock_prompt,
            target_node="STORE",
            threat_level=3.8,
            actions=mock_actions
        )
        print(f"\n--- LLM Backend: {result['backend']} | Model: {result['model']} ---")
        print(result["advisory"])

    asyncio.run(demo())
