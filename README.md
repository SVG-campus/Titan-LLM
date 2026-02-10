⚡ Titan-LLM: High-Performance Llama-3 from Scratch
Titan-LLM is a fully custom implementation of the Llama-3 architecture (RoPE, SwiGLU, RMSNorm) designed to run at maximum speed on older hardware (Pascal P100).
🚀 Performance Journey
We didn't just train a model; we engineered an inference engine.


Baseline (Naive): 10 tokens/sec


Optimization 1 (KV-Cache): 25 tokens/sec (2.5x speedup)


Optimization 2 (Static Allocation): 115 tokens/sec (Linear Scaling)


Optimization 3 (CUDA Graphs): 168 tokens/sec (Hardware Limit)


🧠 The Model


Architecture: Llama-3 Clone (8 layers, 8 heads, 512 dim)


Training: Trained on TinyStories for 2B tokens.


Capabilities: Fluent storytelling, coherent grammar.


🛠️ How to Run
pip install torch
python inference.py