# Awesome Inference Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> Inference engineering is the practice of running trained models in production so that they are fast, cheap and reliable, from GPU kernels to serving fleets.

Sections run from fundamentals to production. Each one is split into tiers:

- **Tier 1** - No prior inference knowledge needed.
- **Tier 2** - Builds on Tier 1.
- **Tier 3** - Specialist or research-level.

**New to inference?** Start with Tier 1 in each section, top to bottom. Then go to Tier 2 and Tier 3 for the topics you work on.

The focus is on engineering: the tools, systems and operational practice of serving models. For a complete index of inference research papers, see [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference).

## Contents

- [Fundamentals](#fundamentals)
- [Inference Engines](#inference-engines)
- [Attention and KV Cache](#attention-and-kv-cache)
- [Batching and Scheduling](#batching-and-scheduling)
- [Speculative Decoding](#speculative-decoding)
- [Quantization](#quantization)
- [Structured Output](#structured-output)
- [Distributed Inference](#distributed-inference)
- [Deployment and Orchestration](#deployment-and-orchestration)
- [Gateways and Routing](#gateways-and-routing)
- [Observability and Profiling](#observability-and-profiling)
- [Benchmarking](#benchmarking)
- [Capacity and Cost](#capacity-and-cost)
- [GPU Programming and Kernels](#gpu-programming-and-kernels)
- [Hardware](#hardware)
- [Multimodal Inference](#multimodal-inference)
- [Courses and Books](#courses-and-books)

## Fundamentals

### Tier 1

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual introduction to the transformer architecture.
- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works/) - Walkthrough from prompt to response, including tokenization and autoregressive decoding.
- [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) - Back-of-the-envelope compute and memory math for transformer inference.
- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) - Defines TTFT, TPOT, throughput and the trade-offs between them.

### Tier 2

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - Overview of batching, KV cache, parallelism and quantization.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) - Survey of distillation, quantization, pruning and sparsity for inference.

### Tier 3

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Why identical requests give different outputs and how batch-invariant kernels fix it.

## Inference Engines

### Tier 1

- [Ollama](https://github.com/ollama/ollama) - Local model runner with a model library and an OpenAI-compatible API.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++ across CPUs, GPUs and Apple Silicon, with the GGUF format.
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving engine built around PagedAttention and continuous batching.
- [SGLang](https://github.com/sgl-project/sglang) - Serving framework with RadixAttention prefix caching and a structured generation frontend.
- [Understanding LLM Inference Engines: Inside Nano-vLLM](https://neutree.ai/blog/nano-vllm-part-1) - Guided walkthrough of scheduling, batching and memory management in nano-vllm.

### Tier 2

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) - Tour of a production engine's scheduler, KV cache manager and model runner.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA's optimized LLM inference library for its GPUs.
- [LMDeploy](https://github.com/InternLM/lmdeploy) - Toolkit for compressing, deploying and serving LLMs with the TurboMind engine.
- [MLX LM](https://github.com/ml-explore/mlx-lm) - LLM inference and fine-tuning on Apple Silicon with MLX.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) - Compiler-based deployment of LLMs to GPUs, mobile and browsers.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - LLM inference engine written in Rust.
- [ExecuTorch](https://github.com/pytorch/executorch) - On-device inference runtime for PyTorch models.
- [Triton Inference Server](https://github.com/triton-inference-server/server) - Multi-framework model server with dynamic batching and model ensembles.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Cross-platform inference runtime for ONNX models.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel's toolkit for optimizing and running models on CPUs, GPUs and NPUs.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Fast inference engine for transformer models on CPU and GPU.

### Tier 3

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) - Lightweight vLLM reimplementation in about 1,200 lines of Python.
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - Compact reimplementation of SGLang for studying engine design.

## Attention and KV Cache

### Tier 1

- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - Builds up PagedAttention from the memory problem it solves.
- [Prompt Caching](https://ngrok.com/blog/prompt-caching) - How prompt caching works and when it pays off.

### Tier 2

- [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) - From prefix caching in vLLM to cache-aware routing in llm-d.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Paged KV cache allocation; the vLLM paper (SOSP 2023).
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - Grouped-query attention to shrink the KV cache (2023).
- [LMCache](https://github.com/LMCache/LMCache) - KV cache layer for reuse across requests, engines and storage tiers.

### Tier 3

- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware exact attention; FlashAttention-2 and -3 extend it to better parallelism and Hopper GPUs (2022).
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - Introduces Multi-head Latent Attention (MLA) for compressed KV cache (2024).
- [kvcached](https://github.com/ovg-project/kvcached) - Virtualized, elastic KV cache for sharing GPUs between models.

## Batching and Scheduling

### Tier 1

- [How Continuous Batching Enables 23x Throughput in LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) - Static versus continuous batching, explained with benchmarks.

### Tier 2

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - Iteration-level scheduling, the origin of continuous batching (OSDI 2022).
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) - Chunked prefill to balance throughput and latency (OSDI 2024).

## Speculative Decoding

### Tier 1

- [An Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) - Beginner-friendly explanation of draft-and-verify, EAGLE-3 and multi-token prediction.

### Tier 2

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - The original draft-and-verify method (ICML 2023).
- [Medusa](https://arxiv.org/abs/2401.10774) - Extra decoding heads that propose multiple tokens per step (ICML 2024).
- [EAGLE](https://github.com/SafeAILab/EAGLE) - Feature-level drafting; EAGLE-3 is widely supported in engines.
- [Speculators](https://github.com/vllm-project/speculators) - Library for training speculative decoding draft models that run in vLLM.

### Tier 3

- [SuffixDecoding](https://suffix-decoding.github.io/) - Model-free drafting with suffix trees for agentic workloads (NeurIPS 2025).
- [Spec-Bench](https://github.com/hemingkx/Spec-Bench) - Unified benchmark for speculative decoding methods.

## Quantization

### Tier 1

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Illustrated introduction to number formats and quantization methods.

### Tier 2

- [GPTQ](https://arxiv.org/abs/2210.17323) - One-shot post-training weight quantization using second-order information (2022).
- [AWQ](https://arxiv.org/abs/2306.00978) - Activation-aware weight quantization for low-bit LLMs (MLSys 2024).
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - Migrates activation outliers into weights for W8A8 inference (2022).
- [LLM Compressor](https://github.com/vllm-project/llm-compressor) - Quantization and sparsity recipes that produce vLLM-ready checkpoints.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - 8-bit and 4-bit quantization for PyTorch.
- [TorchAO](https://github.com/pytorch/ao) - PyTorch-native quantization and sparsity for training and inference.

### Tier 3

- [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) - Quantizing the KV cache to FP4 for long contexts and large batches.
- [GPTQModel](https://github.com/ModelCloud/GPTQModel) - Maintained GPTQ toolkit with multi-backend kernels.
- [AutoRound](https://github.com/intel/auto-round) - Signed-gradient weight rounding for low-bit quantization.

## Structured Output

### Tier 1

- [Outlines](https://github.com/dottxt-ai/outlines) - Structured generation with JSON Schema, regex and grammars.

### Tier 2

- [XGrammar](https://github.com/mlc-ai/xgrammar) - Grammar-constrained decoding engine used by vLLM and SGLang.
- [LLGuidance](https://github.com/guidance-ai/llguidance) - Fast constrained decoding library in Rust.
- [Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang) - Benchmark of XGrammar and LLGuidance backends across engines.
- [Tool Calling in Inference](https://www.baseten.co/blog/tool-calling-in-inference/) - Why tool calling quality varies across providers and how to make it reliable.

## Distributed Inference

### Tier 1

- [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) - Diagrams of how parallelism strategies shard a transformer.

### Tier 2

- [How To Scale Your Model: Inference](https://jax-ml.github.io/scaling-book/inference/) - Rigorous treatment of latency, throughput and sharding for transformer inference.
- [DistServe](https://arxiv.org/abs/2401.09670) - Disaggregates prefill and decode onto separate GPUs to meet goodput targets (OSDI 2024).
- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - Prefill/decode disaggregation in production.
- [The vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) - Combining TP, DP, PP and EP for MoE models in vLLM.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) - Datacenter-scale serving framework with disaggregation and KV-aware routing.
- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-native distributed inference built on vLLM.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA's collective communication library.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Interactive guide to data, tensor, context, pipeline and expert parallelism; written for training, but the concepts carry over to serving.
- [Disaggregated Inference: 18 Months Later](https://haoailab.com/blogs/distserve-retro/) - How prefill/decode disaggregation became standard across engines.

### Tier 3

- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) - Production disaggregated serving at Moonshot AI; a version appeared at FAST 2025 (2024).
- [vLLM Large Scale Serving: DeepSeek with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving) - Wide expert parallelism, dual-batch overlap and expert load balancing for DeepSeek on multi-node H200 clusters.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KV-cache-centric disaggregated serving platform from Moonshot AI.
- [NIXL](https://github.com/ai-dynamo/nixl) - Point-to-point transfer library for moving KV cache between nodes.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Communication library for expert-parallel MoE dispatch and combine.
- [UCCL](https://github.com/uccl-project/uccl) - GPU communication library for collectives, P2P and expert parallelism.

## Deployment and Orchestration

### Tier 1

- [vLLM Production Stack](https://github.com/vllm-project/production-stack) - Reference Kubernetes stack for vLLM with routing and observability.
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Scalable model serving on Ray with LLM-specific APIs.
- [BentoML](https://github.com/bentoml/BentoML) - Framework for packaging and serving models as APIs.

### Tier 2

- [AIBrix](https://github.com/vllm-project/aibrix) - Control plane for LLM serving on Kubernetes.
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) - Kubernetes API for multi-host inference groups.
- [KEDA](https://github.com/kedacore/keda) - Event-driven autoscaling for Kubernetes, often driven by queue depth or engine metrics.
- [Model Streamer](https://github.com/dsx-ai-factory/model-streamer) - Streams SafeTensors weights to GPU memory concurrently to cut model load time.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) - How a distributed serving stack splits scheduling, routing and model servers.
- [SkyPilot](https://github.com/skypilot-org/skypilot) - Runs and serves AI workloads across clouds and Kubernetes.
- [dstack](https://github.com/dstackai/dstack) - Orchestration layer for training and inference across GPU clouds, Kubernetes and bare metal.
- [KServe](https://github.com/kserve/kserve) - Kubernetes model serving platform with autoscaling and inference graphs.
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) - Manages GPU drivers, runtime and device plugins on Kubernetes.

### Tier 3

- [HAMi](https://github.com/Project-HAMi/HAMi) - GPU sharing and virtualization middleware for Kubernetes.

## Gateways and Routing

### Tier 1

- [LiteLLM](https://github.com/BerriAI/litellm) - OpenAI-compatible proxy for many providers with budgets and fallbacks.

### Tier 2

- [Agent Router](https://github.com/theagentrouter/agent-router) - Envoy-based control plane for AI and agent traffic, formerly Envoy AI Gateway.
- [Portkey Gateway](https://github.com/Portkey-AI/gateway) - Lightweight AI gateway with retries, fallbacks and load balancing.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) - Decision layer that picks a model for each request based on its content.
- [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Kubernetes Gateway API extension for model-aware, KV-cache-aware routing.

## Observability and Profiling

### Tier 1

- [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) - Exports NVIDIA GPU metrics to Prometheus.
- [Langfuse](https://github.com/langfuse/langfuse) - Request-level tracing and analytics for LLM applications.
- [A High-Level Guide to GPU Utilization](https://modal.com/blog/gpu-utilization-guide) - Why nvidia-smi utilization misleads for inference, and which metrics to watch instead.

### Tier 2

- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - System-wide profiler for CPU/GPU timelines.
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) - Operator- and kernel-level profiling for PyTorch workloads.
- [NVTX](https://github.com/NVIDIA/NVTX) - Annotations to label code ranges in Nsight timelines.
- [OpenTelemetry GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai) - Standard attribute names for tracing model calls.
- [Host Overhead is Killing Your Inference Efficiency](https://modal.com/blog/host-overhead-inference-efficiency) - Finding and removing CPU-side bottlenecks that idle the GPU.
- [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics.html) - Design of vLLM's Prometheus metrics, such as queue time, KV cache usage and request latency.

### Tier 3

- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - Kernel-level profiler for CUDA applications.
- [The GPU Observability Gap](https://eunomia.dev/blog/2025/10/14/the-gpu-observability-gap-why-we-need-ebpf-on-gpu-devices/) - Why eBPF-style visibility is needed inside GPU workloads.

## Benchmarking

### Tier 1

- [LLM Inference Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) - What TTFT, ITL and throughput actually measure, and common pitfalls.
- [GuideLLM](https://github.com/vllm-project/guidellm) - Evaluates LLM deployments under realistic traffic patterns.
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Checks that optimizations like quantization have not hurt model quality.

### Tier 2

- [AIPerf](https://github.com/ai-dynamo/aiperf) - Load generator for measuring inference latency and throughput.
- [inference-perf](https://github.com/kubernetes-sigs/inference-perf) - Kubernetes SIG tool for benchmarking GenAI inference servers.
- [Artificial Analysis Hardware Benchmarks](https://artificialanalysis.ai/hardware-inference-stack/) - Independent comparisons of accelerator systems for LLM serving.
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) - Continuously updated open benchmarks across accelerators and engines.

### Tier 3

- [MLPerf Inference](https://github.com/mlcommons/inference) - Industry-standard inference benchmark suite.

## Capacity and Cost

### Tier 1

- [GPU (In)efficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Why production GPUs sit underused and what to do about it.
- [LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/) - Step-by-step method from benchmark results to instance count and cost per million tokens.

### Tier 2

- [The Next 1000x Cost Saving of LLM](https://ralphmao.github.io/token-cost/) - Where past per-token cost reductions came from, and where the next 1000x could come from.

### Tier 3

- [Vidur](https://github.com/microsoft/vidur) - Simulator for exploring LLM serving configurations without GPUs.

## GPU Programming and Kernels

### Tier 1

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - Compute-bound, memory-bound and overhead-bound regimes explained.
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) - Official tutorials, starting with vector add and fused softmax.
- [Triton](https://github.com/triton-lang/triton) - Python-based language and compiler for writing GPU kernels.

### Tier 2

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step matmul optimization from naive to near-cuBLAS.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - NVIDIA's official reference for the CUDA programming model.
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA C++ and Python templates for high-performance GEMM and related kernels.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Kernel library for LLM serving: attention, GEMM, sampling and MoE.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Reference implementation of the FlashAttention kernels.

### Tier 3

- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - From GPU architecture and PTX/SASS to Hopper tensor core pipelines.
- [Outperforming cuBLAS on H100: A Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Hopper-specific matmul optimizations, step by step.
- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) - Writing Hopper GEMMs with CUTLASS and WGMMA.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) - Decoding kernels for Multi-head Latent Attention.
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - Tile primitives for writing fast deep learning kernels.
- [TileLang](https://github.com/tile-ai/tilelang) - Tile-level DSL for writing high-performance kernels.
- [Helion](https://github.com/pytorch/helion) - Python-embedded DSL from PyTorch for authoring ML kernels.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - Compact tensor core kernel library for FP8, FP4 and BF16 GEMMs and fused MoE.
- [AITER](https://github.com/ROCm/aiter) - AMD's library of inference operators for ROCm.
- [FlashInfer-Bench](https://bench.flashinfer.ai/) - Benchmark platform for LLM serving kernels.

## Hardware

### Tier 1

- [Domain-Specific Architectures for AI Inference](https://fleetwood.dev/posts/domain-specific-architectures) - Why AI inference accelerators look the way they do.

### Tier 2

- [ROCm](https://rocm.docs.amd.com/) - Documentation for AMD's open software stack for GPU computing.
- [MLX](https://github.com/ml-explore/mlx) - Array framework for machine learning on Apple Silicon.
- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - How GPUs work, how they are networked and what it means for LLMs.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - How TPUs work, how they are networked and how they differ from GPUs.

### Tier 3

- [A Close Look at SRAM for Inference](https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference) - SRAM versus HBM trade-offs for inference chips; the cost section is for paid subscribers.

## Multimodal Inference

### Tier 1

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper speech recognition on CTranslate2.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Portable Whisper inference in C/C++.
- [Diffusers](https://github.com/huggingface/diffusers) - Diffusion pipelines with attention backends and caching options.
- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) - Serving engine for embedding and reranker models.

### Tier 2

- [SGLang Diffusion](https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/) - SGLang's serving optimizations applied to image and video generation.
- [Optimizing Diffusion for Production-Ready Speeds](https://a-r-r-o-w.github.io/blog/3_blossom/00001_productionizing_diffusion-1/) - Step-by-step FLUX inference speedups with torch.compile, QKV fusion and CUDA Graphs, with benchmarks.
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - vLLM extension for serving omni-modality models.

### Tier 3

- [xDiT](https://github.com/xdit-project/xDiT) - Parallel inference engine for diffusion transformers.

## Courses and Books

### Tier 1

- [Inference Engineering](https://www.baseten.co/inference-engineering/book/) - Philip Kiely's book on serving generative AI in production, from GPUs to Kubernetes; free to read online.
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures) - Community lecture series on GPU programming and ML systems.

### Tier 2

- [ML Hardware and Systems (Cornell ECE 5545)](https://abdelfattah-class.github.io/ece5545/) - Course connecting ML algorithms to hardware and systems.
- [Build an LLM from Scratch with MAX](https://llm.modular.com/) - Interactive tutorial that builds an LLM with Modular's MAX.
- [AI Infrastructure Book](https://github.com/bojieli/ai-infra-book) - Quantitative treatment of LLM inference and training system design (in Chinese).
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) - Course on quantization, pruning, distillation and efficient inference.
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - Book on TPU/GPU architecture, communication and parallelism for LLMs.

## Related Lists

- [AI Inference Resources](https://github.com/aerlabsAI/ai-inference-resources#readme) - Tiered learning guide to AI inference engineering.
- [GPU Performance Engineering Resources](https://github.com/wafer-ai/gpu-perf-engineering-resources#readme) - Curated path from GPU fundamentals to production inference.
- [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference#readme) - Papers and code on LLM inference.
- [Awesome LLM Inference Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine#readme) - Survey of LLM inference engines.
- [Awesome vLLM Plugins](https://github.com/BudEcosystem/Awesome-vLLM-plugins#readme) - Plugins that extend vLLM.

## Contributing

Contributions are welcome. Read the [contribution guidelines](contributing.md) first.
