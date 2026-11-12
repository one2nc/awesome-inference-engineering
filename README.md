# Awesome Inference Engineering [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated guide to running models efficiently and reliably: inference fundamentals, engines, research papers, benchmarks and production practices.

The focus is LLM inference, with dedicated coverage of embeddings, speech, vision and diffusion.

<p align="center">
  <a href="assets/inference-landscape.png"><img src="assets/inference-landscape.png" alt="Inference Engineering landscape" width="100%"></a>
</p>

<a id="start-here"></a>

## Start here

Choose a path, then use the topic index for deeper reading. The order within each path is intentional.

| Your goal | Read in this order | Practical outcome |
|---|---|---|
| Learn the fundamentals | [How inference works](#inference-basics) -> [KV cache](#kv-cache-basics) -> [prefill and decode](#prefill-and-decode) -> [batching](#batching-and-scheduling) -> [latency metrics](#latency-metrics) -> [bottleneck analysis](#bottleneck-analysis) | Explain where time and memory go during generation. |
| Deploy a first model | [Model selection and evaluation](#model-strategy) -> [engine selection](#inference-engines) -> [sizing](#instances-and-interconnects) -> [containers](#containers-and-gpus) -> [benchmarking methodology](#benchmarking-and-profiling) | Serve one model and measure its quality and latency against a representative workload. |
| Improve production serving | [Benchmarking and profiling](#benchmarking-and-profiling) -> [optimization choices](#optimization-overview) -> [quantization](#quantization) -> [structured output](#structured-output) -> [reliability](#serving-reliability) -> [cost](#cost-and-efficiency) | Identify a bottleneck, change one variable and verify latency, quality and cost. |

<a id="contents"></a>

## Table of contents

- [Start here](#start-here)
- [1. Inference fundamentals](#fundamentals)
  - [1.1 What inference is](#inference-basics)
  - [1.2 Transformers and attention](#transformers-and-attention)
  - [1.3 The KV cache](#kv-cache-basics)
  - [1.4 Prefill, decode and serving modes](#prefill-and-decode)
  - [1.5 Batching and request scheduling](#batching-and-scheduling)
  - [1.6 Latency and throughput metrics](#latency-metrics)
  - [1.7 Bottleneck analysis](#bottleneck-analysis)
  - [1.8 Inference optimization overview](#optimization-overview)
  - [1.9 Model selection, evaluation and fine-tuning](#model-strategy)
  - [1.10 Mixture of Experts and state-space models](#model-architectures)
- [2. Hardware](#hardware)
  - [2.1 GPU architecture](#gpu-architecture)
  - [2.2 Accelerators by vendor](#accelerators-by-vendor)
  - [2.3 Instances, interconnects and sizing](#instances-and-interconnects)
- [3. Software](#software)
  - [3.1 Frameworks, runtimes and model formats](#frameworks-and-formats)
  - [3.2 Inference engines](#inference-engines)
  - [3.3 Benchmarking and profiling](#benchmarking-and-profiling)
  - [3.4 Kernel libraries, languages and compilers](#kernel-libraries)
  - [3.5 GEMM kernels and tile programming](#gemm-kernels)
- [4. Inference techniques](#inference-techniques)
  - [4.1 Quantization](#quantization)
  - [4.2 Speculative decoding](#speculative-decoding)
  - [4.3 KV-cache reuse, routing and offload](#kv-cache)
  - [4.4 Long-context handling](#long-context)
  - [4.5 Parallelism and communication](#parallelism-and-communication)
  - [4.6 Disaggregation](#disaggregation)
  - [4.7 Structured output, sampling and tool calling](#structured-output)
- [5. Multimodal inference](#beyond-llms)
  - [5.1 Omni-modal inference](#omni-modal-inference)
  - [5.2 Vision-language models](#vision-language-models)
  - [5.3 Speech](#speech)
  - [5.4 Computer vision and video analytics](#vision-and-video-analytics)
  - [5.5 Image and video generation](#image-and-video-generation)
- [6. Production inference](#production)
  - [6.1 Containers and GPU enablement](#containers-and-gpus)
  - [6.2 Kubernetes serving platforms](#kubernetes-serving)
  - [6.3 Gateways, routing and autoscaling](#routing-and-autoscaling)
  - [6.4 Cloud and GPU hosting platforms](#cloud-platforms)
  - [6.5 Observability, security and debugging](#observability-and-reliability)
  - [6.6 Cost and efficiency](#cost-and-efficiency)
  - [6.7 Offline batch inference, RL rollouts and reasoning budgets](#batch-and-rollout)
  - [6.8 Serving reliability and deployment safety](#serving-reliability)
- [7. Learning resources](#learning)
  - [7.1 Courses and books](#courses-and-books)
  - [7.2 Repositories and reading lists](#repositories-and-reading-lists)
  - [7.3 Talks](#talks)
- [Contributing](#contributing)
- [License](#license)

<a id="fundamentals"></a>

## 1. Inference fundamentals

<a id="inference-basics"></a>

### 1.1 What inference is

- [The Difference Between AI Training and Inference](https://nebius.com/blog/posts/difference-between-ai-training-and-inference) - How serving a trained model differs from training it.
- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works/) - Prompt-to-response walkthrough of tokenization, embedding and autoregressive generation.

<a id="transformers-and-attention"></a>

### 1.2 Transformers and attention

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Diagram-led walkthrough of self-attention and the encoder-decoder transformer.
- [FlashAttention](https://arxiv.org/html/2205.14135v2) - **Paper, 2022**. IO-aware exact attention that tiles computation to avoid materializing the attention matrix.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - **Paper, 2023**. Better work partitioning and parallelism across thread blocks and warps.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) - **Paper, 2024**. Hopper-specific attention using asynchronous tensor cores, TMA and FP8.
- [Flash-Decoding for Long-Context Inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) - Splitting the KV sequence across thread blocks to keep the GPU busy during decode.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - **Paper, 2019**. Multi-query attention (MQA), which shares keys and values across heads to shrink the KV cache.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) - **Paper, 2023**. Grouped-query attention, the middle ground between multi-head and multi-query attention used by most open models.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - **Paper, 2024**. Introduces multi-head latent attention (MLA), which compresses the KV cache into a low-rank latent.

<a id="kv-cache-basics"></a>

### 1.3 The KV cache

- [KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching) - How caching keys and values speeds up transformer decoding.
- [What Is a KV Cache, and Why Does It Make LLM Inference Faster?](https://sebastianraschka.com/faq/docs/kv-cache.html) - What the KV cache stores and why LLM inference depends on it.
- [Paged Attention from First Principles](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - PagedAttention explained from virtual-memory basics.

<a id="prefill-and-decode"></a>

### 1.4 Prefill, decode and serving modes

- [Prefill vs Decode](https://redis.io/blog/prefill-vs-decode/) - Why prefill often hits compute limits and small-batch decode often hits memory-bandwidth limits; the balance changes with batching and workload.
- [Using Asynchronous Inference in Production](https://www.baseten.co/blog/using-asynchronous-inference-in-production) - Queue-based inference for long-running or bursty workloads.
- [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) - The chat, completions and embeddings endpoints most engines now expose, including streaming responses.
- [OpenAI-Compatible API](https://handbook.modular.com/model-interaction/openai-compatible-api/) - LLM Inference Handbook chapter on why the OpenAI API shape became the common interface for self-hosted models.
- [Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating) - How chat messages are rendered into the token sequence a model expects; a mismatched template silently degrades output.

<a id="batching-and-scheduling"></a>

### 1.5 Batching and request scheduling

- [Continuous Batching from First Principles](https://huggingface.co/blog/continuous_batching) - Hugging Face walkthrough from static batching to iteration-level scheduling.
- [Continuous Batching: Optimizing LLM Inference Throughput](https://mbrenndoerfer.com/writing/continuous-batching) - Interactive explanation of how requests join and leave a running batch.
- [Efficient Request Queueing](https://huggingface.co/blog/tngtech/llm-performance-request-queueing) - How queueing in front of an LLM server affects latency and throughput under load.
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - **Paper, 2022**. OSDI paper that introduced iteration-level scheduling, the basis of continuous batching.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - **Paper, 2023**. SOSP 2023 vLLM paper on paging the KV cache to raise batch sizes.
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) - **Paper, 2024**. Chunked prefill and stall-free batching to balance throughput against decode latency.

<a id="latency-metrics"></a>

### 1.6 Latency and throughput metrics

- [LLM Inference Benchmarking: Fundamental Concepts](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) - Definitions of TTFT, inter-token latency, tokens per second and end-to-end latency.
- [NIM LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/metrics) - How NVIDIA defines and measures TTFT, inter-token latency, throughput and request latency.
- [Key Metrics for LLM Inference](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics/) - LLM Inference Handbook chapter on latency, throughput, goodput and tail latency, with an interactive TPOT calculator.
- [Reproducible Performance Metrics for LLM Inference](https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference) - Anyscale's case for measuring TTFT, inter-token latency and cost the same way across providers.
- [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/) - The Prometheus metrics vLLM exposes for queueing, latency, KV-cache usage and throughput.

<a id="bottleneck-analysis"></a>

### 1.7 Bottleneck analysis

- [Transformer Inference Arithmetic](https://kipp.ly/p/transformer-inference-arithmetic) - Back-of-the-envelope compute and memory math for transformer inference.
- [How Fast Can We Perform a Forward Pass?](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/) - Theoretical and practical limits on forward-pass latency.
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - Separates compute, memory-bandwidth and overhead bottlenecks.
- [A Guide to LLM Inference and Performance](https://www.baseten.co/blog/llm-transformer-inference-guide) - Ops:byte ratio, arithmetic intensity and memory bandwidth applied to choosing a GPU.
- [What Is the Roofline Model?](https://modal.com/gpu-glossary/perf/roofline-model) - Short definition of the roofline model with a worked diagram.
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) - NVIDIA's background guide to math-bound and memory-bound operations.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) - Roofline analysis and arithmetic intensity applied to model workloads.

<a id="optimization-overview"></a>

### 1.8 Inference optimization overview

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - Prefill and decode, compute and memory limits, batching and KV-cache trade-offs.
- [Top 5 AI Model Optimization Techniques for Faster, Smarter Inference](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/) - Post-training quantization, quantization-aware training, quantization-aware distillation, speculative decoding and pruning with distillation.
- [Inference Performance from First Principles](https://aleph-alpha.com/en/blog/inference-performance-from-first-principles/) - Aleph Alpha's analytical model of prefill and decode bottlenecks across compute, memory movement and communication, for dense and MoE models.

<a id="model-strategy"></a>

### 1.9 Model selection, evaluation and fine-tuning

#### Model selection

- [Choosing the Right Model](https://handbook.modular.com/getting-started/choosing-the-right-model/) - Criteria for picking an open model for a given task, size and budget.
- [LLM Benchmarking Framework with LLM-as-a-JURY](https://github.com/aws-samples/sample-bedrock-migration-and-modernization-tools/blob/main/360-eval/README.md) - AWS 360-Eval, which benchmarks candidate models on your own data using a jury of LLM judges.

#### Evaluation

- [LLM Evaluation: A Beginner's Guide](https://www.evidentlyai.com/llm-guide/llm-evaluation) - Evaluation methods for LLM systems, from benchmarks to LLM-as-a-judge.
- [Langfuse](https://github.com/langfuse/langfuse) - Open-source platform for tracing, evaluating and monitoring LLM applications.
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - Open-source LLM tracing and evaluation built on OpenTelemetry.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - EleutherAI's framework for running standard academic benchmarks against a model.
- [Lighteval](https://github.com/huggingface/lighteval) - Hugging Face toolkit for evaluating LLMs across inference backends such as vLLM and SGLang.
- [DeepEval](https://github.com/confident-ai/deepeval) - Pytest-style framework for unit-testing LLM outputs with metrics and LLM-as-a-judge.
- [promptfoo](https://github.com/promptfoo/promptfoo) - CLI for testing prompts, models and agents, including red-teaming.
- [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) - UK AI Security Institute's framework for LLM and agent evaluations.

#### Fine-tuning and adapters

- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning methods such as LoRA, which produce small adapters that engines can serve together on one base model.
- [TRL](https://github.com/huggingface/trl) - Hugging Face library for supervised fine-tuning, DPO and RL-based post-training.
- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient fine-tuning and RL for open models, with export to GGUF and vLLM.
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Config-driven fine-tuning for open models, from LoRA to multi-GPU full fine-tuning.
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory) - Unified fine-tuning for 100+ LLMs and VLMs with a CLI and web UI.

#### Pruning and distillation

- [LLM Model Pruning and Knowledge Distillation with NVIDIA NeMo Framework](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/) - Producing a smaller student model through pruning and distillation.
- [SparseGPT](https://arxiv.org/abs/2301.00774) - **Paper, 2023**. One-shot pruning of large GPT-family models to 50% or more sparsity without retraining.
- [Wanda](https://arxiv.org/abs/2306.11695) - **Paper, 2023**. Pruning by weight magnitude times input activation, with no retraining or weight updates.
- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679) - **Paper, 2024**. NVIDIA Minitron recipe for deriving smaller models from a larger one.

<a id="model-architectures"></a>

### 1.10 Mixture of Experts and state-space models

- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - How sparse expert layers work, and what they change for training and inference.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - **Paper, 2024**. Sparse MoE model that activates two of eight experts per token.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - **Paper, 2024**. Large MoE model with MLA, auxiliary-loss-free load balancing and multi-token prediction.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - **Paper, 2023**. State-space model with constant-size recurrent state instead of a growing KV cache.
- [Jamba](https://arxiv.org/abs/2403.19887) - **Paper, 2024**. Hybrid Transformer-Mamba MoE model that interleaves attention and state-space layers.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) - **Paper, 2024**. Linear-attention layer combining gating with the delta rule, used in recent hybrid models.
- [Nemotron-H](https://arxiv.org/abs/2504.03624) - **Paper, 2025**. NVIDIA's hybrid Mamba-Transformer models that replace most attention layers to cut inference cost.
- [Hybrid Models as First-Class Citizens in vLLM](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/) - How vLLM manages state-space and attention caches together for hybrid models.

<a id="hardware"></a>

## 2. Hardware

<a id="gpu-architecture"></a>

### 2.1 GPU architecture

- [Intro to GPUs for the Researcher](https://hackbot.dad/writing/intro-to-gpus/) - Practical GPU mental model aimed at raising model FLOPS utilization.
- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - How GPU chips and their networks work, and what that means for LLMs.
- [What Is GPU Memory and Why It Matters for LLM Inference](https://www.bentoml.com/blog/what-is-gpu-memory-and-why-it-matters-for-llm-inference) - How weights, KV cache and activations compete for VRAM.

<a id="accelerators-by-vendor"></a>

### 2.2 Accelerators by vendor

#### NVIDIA

- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) - Architecture guide to the NVIDIA H100 Tensor Core GPU.
- [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) - Blackwell Ultra architecture, NVFP4 and attention-layer acceleration.
- [Inside the NVIDIA Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) - Vendor overview of the Vera Rubin rack-scale platform and its chips, including the Vera CPU.

#### AMD

- [AMD CDNA 3 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) - Architecture of the MI300 series, including chiplets and memory.
- [AMD CDNA 4 Architecture White Paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) - Architecture of the MI350 series and its low-precision data types.
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/) - Documentation for AMD's ROCm software platform for Instinct accelerators.
- [AMD Instinct MI300 and MI350 Workload Optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) - AMD's guide to tuning vLLM and other inference workloads on Instinct GPUs.

#### Google TPU

- [Ironwood: The first Google TPU for the age of inference](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/) - Google's announcement of its seventh-generation, inference-focused TPU.
- [TPU Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) - Official reference for TensorCores, MXUs, slices, pods and multislice topologies.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) - TPU compute, memory and interconnect explained for model workloads.
- [vLLM JAX](https://github.com/vllm-project/vllm-jax) - vLLM's TPU backend with JAX and PyTorch model support (formerly tpu-inference).
- [JetStream](https://github.com/AI-Hypercomputer/JetStream) - Throughput- and memory-optimized LLM inference engine for TPUs and other XLA devices.

#### AWS Trainium and Inferentia

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) - SDK, compiler and vLLM integration for AWS Trainium and Inferentia accelerators.
- [NxD Inference](https://github.com/aws-neuron/neuronx-distributed-inference) - AWS NeuronX Distributed Inference library and model hub for serving LLMs on Trainium and Inferentia.

#### Intel Gaudi

- [Intel Gaudi Documentation](https://docs.habana.ai/en/latest/) - Software stack, PyTorch integration and vLLM support for Intel Gaudi accelerators.
- [vLLM Gaudi Plugin](https://github.com/vllm-project/vllm-gaudi) - Community-maintained hardware plugin for running vLLM on Intel Gaudi.

#### Other accelerators

- [Cerebras Inference Documentation](https://inference-docs.cerebras.ai/) - API and model documentation for inference on Cerebras wafer-scale systems.
- [Groq Documentation](https://console.groq.com/docs/overview) - API documentation for inference on Groq LPUs.
- [TT-Metalium](https://github.com/tenstorrent/tt-metal) - Tenstorrent's low-level programming model and model library for its accelerators.
- [Qualcomm Efficient Transformers](https://github.com/qualcomm/efficient-transformers) - Library for porting and running Hugging Face models on Qualcomm Cloud AI 100.

<a id="instances-and-interconnects"></a>

### 2.3 Instances, interconnects and sizing

- [NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/) - Vendor overview of NVIDIA's GPU-to-GPU interconnect and switch fabric.
- [RoCE networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/) - How Meta designs and runs RDMA over Converged Ethernet networks for its GPU clusters.
- [NVIDIA InfiniBand](https://www.nvidia.com/en-us/networking/products/infiniband/) - Vendor overview of the InfiniBand fabric used in most multi-node GPU clusters.
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) - How NICs read and write GPU memory directly, which multi-node KV transfer and expert parallelism rely on.
- [Amazon EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/) - AWS accelerated-computing families for GPU, Trainium and Inferentia.
- [Google Cloud GPU Machine Types](https://docs.cloud.google.com/compute/docs/gpus) - GPU models, machine families and regional availability on Compute Engine.
- [Azure VM Sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview) - Azure VM families, including the GPU-accelerated N-series.
- [Artificial Analysis Hardware Benchmarks](https://artificialanalysis.ai/hardware-inference-stack/) - Independent throughput, speed and cost benchmarks of inference hardware, from datacenter systems to laptops and phones.
- [Calculating GPU Memory for Serving LLMs](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms/) - LLM Inference Handbook chapter on estimating weight and KV-cache memory before choosing a GPU.
- [Accelerate Model Memory Estimator](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator) - CLI that estimates a Hub model's memory footprint across precisions without downloading weights.
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) - Runtime controls for memory utilization, preemption, chunked prefill and batch sizes.

<a id="software"></a>

## 3. Software

<a id="frameworks-and-formats"></a>

### 3.1 Frameworks, runtimes and model formats

- [PyTorch](https://github.com/pytorch/pytorch) - Core framework, including torch.compile and the PyTorch profiler.
- [JAX](https://github.com/jax-ml/jax) - Composable transformations and XLA compilation, widely used on TPUs.
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face model definitions for text, vision, audio and multimodal models, used for both inference and training.
- [Tokenizers](https://github.com/huggingface/tokenizers) - Fast tokenizer implementations used by most open language models.
- [Common AI Model Formats](https://huggingface.co/blog/ngxson/common-ai-model-formats) - Comparison of GGUF, safetensors, PyTorch and ONNX formats.
- [safetensors](https://github.com/safetensors/safetensors) - Safe, zero-copy tensor serialization format used by most open checkpoints.
- [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) - Specification of the single-file quantized model format used by llama.cpp.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Cross-platform inference runtime for ONNX models.
- [TensorRT](https://github.com/NVIDIA/TensorRT) - NVIDIA SDK that compiles and optimizes models for inference on NVIDIA GPUs.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) - Intel toolkit for optimizing and running models on Intel CPUs, GPUs and NPUs.

<a id="inference-engines"></a>

### 3.2 Inference engines

#### Datacenter serving

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput, memory-efficient LLM serving engine with continuous batching and PagedAttention.
- [Inside vLLM](https://www.aleksagordic.com/blog/vllm) - Walkthrough of vLLM internals, from paged attention, continuous batching, prefix caching and speculative decoding to multi-GPU and multi-node serving.
- [SGLang](https://github.com/sgl-project/sglang) - High-performance serving framework for language and multimodal models.
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA library and runtime for optimized LLM inference on NVIDIA GPUs.
- [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA's multi-framework model server, with TensorRT-LLM, vLLM, PyTorch and ONNX backends.
- [LMDeploy](https://github.com/InternLM/lmdeploy) - Toolkit for compressing, deploying and serving LLMs, built on its own TurboMind and PyTorch inference engines.
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - Fast LLM inference engine written in Rust.
- [Modular MAX](https://github.com/modular/modular) - Modular's inference framework and serving stack, with kernels written in Mojo.
- [KTransformers](https://github.com/kvcache-ai/ktransformers) - CPU/GPU hybrid inference that keeps MoE experts in CPU memory to run large MoE models on limited VRAM.
- [DeepSpeed-MII](https://github.com/deepspeedai/DeepSpeed-MII) - Microsoft DeepSpeed's inference library, built on DeepSpeed-FastGen.
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - Hugging Face's LLM server. Historical: in maintenance mode and archived.

#### Local and desktop

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++ for CPUs and GPUs, using the GGUF model format.
- [Ollama](https://github.com/ollama/ollama) - Downloads and runs open models locally behind a simple API.
- [llamafile](https://github.com/mozilla-ai/llamafile) - Packages a model and llama.cpp into a single executable that runs on most operating systems.
- [LocalAI](https://github.com/mudler/LocalAI) - Self-hosted, OpenAI-compatible server for running LLM, image and audio models locally.
- [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) - Consumer-GPU engine exploiting activation sparsity across GPU and CPU.
- [AirLLM](https://github.com/lyogavin/airllm) - Layer-by-layer loading that runs large models on small GPUs at the cost of latency.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) - Inference library for quantized models on consumer GPUs, using the EXL3 format.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Optimized C++ and Python runtime for transformer models on CPU and GPU.

#### Apple Silicon

- [MLX](https://github.com/ml-explore/mlx) - Array framework for machine learning on Apple Silicon.
- [vllm-metal](https://github.com/vllm-project/vllm-metal) - Community plugin for running vLLM on Apple Silicon GPUs.
- [On-Device Llama 3.1 with Core ML](https://machinelearning.apple.com/research/core-ml-on-device-llama) - Apple's walkthrough of optimizing an LLM for on-device Core ML inference.

#### Mobile and edge

- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch runtime for on-device inference on mobile and embedded hardware.
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) - Compiler-based deployment of LLMs to phones, browsers and GPUs.
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) - Google AI Edge's open-source inference framework for running LLMs on edge devices.
- [MediaPipe LLM Inference](https://developers.google.com/edge/mediapipe/solutions/genai/llm_inference) - Google AI Edge API for running LLMs on Android, iOS and the web.

#### Browser

- [WebLLM](https://github.com/mlc-ai/web-llm) - In-browser LLM inference with WebGPU acceleration.
- [Transformers.js](https://github.com/huggingface/transformers.js) - Runs Hugging Face models in the browser and Node.js on ONNX Runtime, with WebGPU support.

#### Embeddings and rerankers

- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) - Hugging Face server for embedding, reranker and sequence-classification models.
- [Infinity](https://github.com/michaelfeil/infinity) - OpenAI-compatible server for embedding, reranking and CLIP models.
- [vLLM Pooling Models](https://docs.vllm.ai/en/latest/models/pooling_models/) - Serving embedding, classification and reward models in vLLM.
- [Sentence Transformers: Speeding up Inference](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) - ONNX, OpenVINO and quantized backends for embedding models.

<a id="benchmarking-and-profiling"></a>

### 3.3 Benchmarking and profiling

- [GuideLLM](https://github.com/vllm-project/guidellm) - vLLM project's tool for benchmarking deployments against realistic traffic.
- [genai-bench](https://github.com/sgl-project/genai-bench) - SGLang's token-level benchmark for LLM serving endpoints.
- [AIPerf](https://github.com/ai-dynamo/aiperf) - Generative AI load testing, trace replay and SLO analysis.
- [MLPerf Inference](https://github.com/mlcommons/inference) - Reference implementations of the MLPerf Inference benchmarks.
- [Vidur](https://github.com/microsoft/vidur) - Simulator that estimates LLM serving performance across configurations without running them.
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) - SemiAnalysis's open benchmark that continuously runs inference engines across NVIDIA and AMD hardware (formerly InferenceMAX).
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) - Recording operator- and kernel-level timings and memory from PyTorch code.
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) - Meta's library for analyzing PyTorch profiler traces across ranks, including idle time and communication overlap.
- [Perfetto](https://perfetto.dev/) - Trace viewer used to inspect PyTorch and other Chrome-format traces.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) - NVIDIA's system-wide timeline profiler for CPU, GPU and communication activity.
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) - AMD's system-wide timeline profiler for CPU and GPU activity.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) - Profiling CUDA kernels and interpreting hardware metrics.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) - Kernel-level performance analysis for AMD Instinct GPUs.

<a id="kernel-libraries"></a>

### 3.4 Kernel libraries, languages and compilers

- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA C++ templates and Python DSLs for high-performance GEMM and related kernels.
- [cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend) - C++ and Python API for cuDNN graph operations, including fused attention.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Official FlashAttention kernels for NVIDIA and AMD GPUs.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Attention, sampling and MoE kernels for LLM serving, used by vLLM and SGLang.
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) - DeepSeek's multi-head latent attention decoding kernels.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - DeepSeek's FP8 GEMM kernels with fine-grained scaling, including grouped GEMMs for MoE.
- [Marlin](https://github.com/IST-DASLab/marlin) - FP16xINT4 mixed-precision matmul kernel that underpins INT4 serving in vLLM and other engines. Reference code: production kernels ship in vLLM.
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) - Small embedded DSL of tile primitives for writing fast GPU kernels.
- [AITER](https://github.com/ROCm/aiter) - AI Tensor Engine for ROCm: optimized inference operators for AMD GPUs.
- [Composable Kernel](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) - AMD's templated library for fused GEMM, attention and other operators.
- [Triton](https://github.com/triton-lang/triton) - Language and compiler for writing GPU kernels in Python.
- [TileLang](https://github.com/tile-ai/tilelang) - Tile-level DSL for writing GEMM, attention and other high-performance kernels.
- [Mojo Manual](https://mojolang.org/docs/manual/) - Modular's systems language for CPU and GPU kernels.
- [Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) - PyTorch's JIT compiler, used by vLLM and SGLang to fuse operators.
- [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) - Capturing and replaying GPU work to cut kernel launch overhead, which matters most in small-batch decode.
- [FlexAttention](https://pytorch.org/blog/flexattention/) - PyTorch API for writing custom attention variants that compile to fused kernels.
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) - NVIDIA's reference for the CUDA programming model, memory hierarchy and execution.
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/) - AMD's C++ GPU programming model and runtime, with a porting path from CUDA.
- [Gluon Tutorials](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon) - Triton's lower-level language for explicit control over layouts and memory.
- [Apache TVM](https://github.com/apache/tvm) - Open machine learning compiler framework.
- [XLA](https://github.com/openxla/xla) - Machine learning compiler for GPUs, CPUs and TPUs, used by JAX.
- [IREE](https://github.com/iree-org/iree) - MLIR-based compiler and runtime for deploying models from datacenter to edge.

<a id="gemm-kernels"></a>

### 3.5 GEMM kernels and tile programming

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Step-by-step path from a naive kernel to near-cuBLAS performance.
- [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - Worklog of Hopper-specific matmul optimizations.
- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - From PTX/SASS to warp tiling and asynchronous tensor-core pipelines.
- [Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) - The mathematics behind CuTe's layout algebra.
- [Demystifying Numeric Conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) - Type-conversion primitives in CuTeDSL.
- [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) - Overlapping work with CuTeDSL asynchronous pipelines on Blackwell.

<a id="inference-techniques"></a>

## 4. Inference techniques

<a id="quantization"></a>

### 4.1 Quantization

#### Guides

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Illustrated introduction to LLM quantization.
- [Post Training Quantization](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html) - Walkthrough of post-training weight quantization.
- [Quantization-Aware Training for Large Language Models with PyTorch](https://pytorch.org/blog/quantization-aware-training/) - Recovering accuracy lost to quantization by simulating it during training.

#### Low-precision formats

- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) - **Paper, 2023**. Block-scaled MX formats (MXFP8, MXFP6, MXFP4, MXINT8) with a shared scale per block of elements.
- [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf/) - The Open Compute Project specification for MX data formats.
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) - NVIDIA's 4-bit floating-point format for Blackwell and how its two-level scaling differs from MXFP4.
- [gpt-oss](https://github.com/openai/gpt-oss) - OpenAI's open-weight models, released with MXFP4-quantized MoE weights and reference inference code.
- [llama.cpp Quantization Tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) - The GGUF quantization types (k-quants, i-quants) and how to produce them.

#### Papers

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - **Paper, 2022**. The E4M3 and E5M2 FP8 formats now supported by Hopper, Blackwell and MI300 hardware.
- [GPTQ](https://arxiv.org/abs/2210.17323) - **Paper, 2022**. One-shot weight quantization to 3-4 bits using approximate second-order information.
- [AWQ](https://arxiv.org/abs/2306.00978) - **Paper, 2023**. Activation-aware weight quantization that protects the most salient weight channels.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) - **Paper, 2022**. Moving activation outliers into weights to enable W8A8 quantization.
- [QuaRot](https://arxiv.org/abs/2404.00456) - **Paper, 2024**. Rotating hidden states to remove outliers for end-to-end 4-bit weights, activations and KV cache.
- [SpinQuant](https://arxiv.org/abs/2405.16406) - **Paper, 2024**. Learned rotations for low-bit weight, activation and KV-cache quantization.
- [KIVI](https://arxiv.org/abs/2402.02750) - **Paper, 2024**. Tuning-free 2-bit KV-cache quantization, per-channel for keys and per-token for values.

#### Tools

- [LLM Compressor](https://github.com/vllm-project/llm-compressor) - vLLM project's library for producing GPTQ, AWQ, FP8 and other quantized checkpoints.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) - Quantization, pruning, distillation and sparsity for deployment with TensorRT-LLM, vLLM and SGLang.
- [torchao](https://github.com/pytorch/ao) - PyTorch-native quantization and sparsity for training and inference.
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - 8-bit and 4-bit quantization for PyTorch, widely used for QLoRA.
- [GPTQModel](https://github.com/ModelCloud/GPTQModel) - GPTQ quantization toolkit with CUDA, ROCm, XPU and CPU backends.
- [AutoRound](https://github.com/intel/auto-round) - Intel's sign-gradient post-training quantization for low-bit LLMs and VLMs.

<a id="speculative-decoding"></a>

### 4.2 Speculative decoding

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/html/2211.17192v2) - **Paper, 2022**. Foundational draft-target speculative decoding algorithm with parallel verification.
- [EAGLE](https://github.com/SafeAILab/EAGLE) - Official implementation of the EAGLE-1, EAGLE-2 and EAGLE-3 speculative decoding methods.
- [Medusa](https://github.com/FasterDecoding/Medusa) - Multiple decoding heads for candidate token generation. Reference code: production implementations are available in TensorRT-LLM.
- [SpecInfer](https://arxiv.org/abs/2305.09781) - **Paper, 2023**. Verifying a tree of speculated tokens in one pass instead of a single sequence.
- [Better and Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) - **Paper, 2024**. Training models to predict several tokens at once, used for self-speculative decoding.
- [Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding) - Draft-free parallel decoding using Jacobi iteration and an n-gram pool. Reference code: production implementations are available in TensorRT-LLM.
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) - Using n-gram matches from the prompt as drafts, effective for summarization and code editing. Reference code: production equivalent is n-gram speculation in vLLM.
- [MagicDec](https://arxiv.org/abs/2408.11049) - **Paper, 2024**. When speculative decoding still helps at large batch sizes for long-context requests.
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) - Configuring draft-model, n-gram, MTP and EAGLE speculation in vLLM.
- [Speculators: Production-Ready Speculative Decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding) - Why the speculators library standardizes draft-model formats.

<a id="kv-cache"></a>

### 4.3 KV-cache reuse, routing and offload

- [Prompt Caching](https://ngrok.com/blog/prompt-caching) - How cached prompt tokens skip recomputation and why that makes them cheaper.
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) - Design of vLLM's hash-based prefix cache.
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) - Radix-tree KV-cache reuse across requests in SGLang.
- [Ray Serve: Reduce LLM Inference Latency by 60% with Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) - Prefix-aware routing to cut TTFT in multi-turn chat.
- [How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) - KV-cache offload and reuse in Dynamo.
- [Inside vLLM's New KV Offloading Connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) - Offloading KV-cache to CPU DRAM and tuning host-to-device throughput.
- [LMCache](https://github.com/LMCache/LMCache) - KV-cache layer that stores and reuses KV cache across GPU, CPU, disk and remote tiers.
- [PrisKV](https://aibrix.github.io/posts/2025-11-26-priskv-intro/) - Colocated, tiered KV-cache store from the AIBrix project.
- [kvcached](https://github.com/ovg-project/kvcached) - Elastic, virtualized KV-cache that lets multiple models share a GPU.
- [Mooncake](https://github.com/kvcache-ai/Mooncake) - KV-cache-centric serving platform behind Moonshot AI's Kimi, whose Mooncake Store pools, reuses and offloads KV cache across serving instances.

<a id="long-context"></a>

### 4.4 Long-context handling

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - Explains RoPE, the position encoding most context-extension methods modify.
- [RoFormer](https://arxiv.org/abs/2104.09864) - **Paper, 2021**. The paper that introduced rotary position embeddings (RoPE).
- [YaRN](https://arxiv.org/abs/2309.00071) - **Paper, 2023**. Efficient RoPE scaling for extending a model's context window.
- [LongRoPE](https://arxiv.org/abs/2402.13753) - **Paper, 2024**. Extending RoPE-based models beyond 2 million tokens.
- [DroPE](https://pub.sakana.ai/DroPE/) - **Paper, 2025**. Extends context by dropping positional embeddings instead of long-context fine-tuning.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) - **Paper, 2023**. Splitting long sequences across devices with blockwise attention passed around a ring.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) - **Paper, 2023**. StreamingLLM: keeping initial sink tokens plus a sliding window for unbounded streaming generation.
- [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) - **Paper, 2023**. Evicting KV-cache entries that receive little attention.
- [SnapKV](https://github.com/FasterDecoding/SnapKV) - Compressing the KV cache of long prompts by selecting important positions before generation. Reference code: also implemented in kvpress.
- [kvpress](https://github.com/NVIDIA/kvpress) - NVIDIA library implementing and benchmarking many KV-cache compression methods.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) - **Paper, 2025**. DeepSeek's hardware-aligned, trainable sparse attention for long-context models.
- [Evaluating Long Context (Reasoning) Ability](https://nrehiew.github.io/blog/long_context/) - Why models degrade well before their advertised context length, measured on LongCodeEdit.

<a id="parallelism-and-communication"></a>

### 4.5 Parallelism and communication

- [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) - Configuring tensor, pipeline, data and expert parallelism for single-node and multi-node vLLM deployments.
- [How Multi-Node Inference Works for Massive LLMs like DeepSeek-R1](https://www.baseten.co/blog/how-multi-node-inference-works-llms-deepseek-r1) - A DeepSeek-R1 deployment using 16 H100s across two nodes, explaining the memory and communication choices in that configuration.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) - **Paper, 2019**. The paper that defined intra-layer tensor parallelism for transformers.
- [Shift Parallelism](https://arxiv.org/html/2509.16495v2) - **Paper, 2025**. Switching between tensor and data parallelism to balance latency and throughput.
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving) - Wide expert parallelism, dual-batch overlap and disaggregated serving for DeepSeek on H200 clusters.
- [NCCL](https://github.com/NVIDIA/nccl) - NVIDIA collective communication library for multi-GPU and multi-node transfers.
- [RCCL](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl) - AMD's collective communication library for ROCm GPUs.
- [NVSHMEM](https://developer.nvidia.com/nvshmem) - NVIDIA's GPU-initiated communication library, used by DeepEP and other MoE kernels.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) - Expert-parallel communication library for MoE dispatch and combine.
- [UCCL](https://github.com/uccl-project/uccl) - GPU communication library for collectives, P2P KV transfer and expert parallelism.
- [llm-d Wide Expert Parallelism](https://llm-d.ai/docs/architecture/disaggregation/wide-expert-parallelism) - How llm-d deploys large MoE models with data and expert parallelism across nodes.

<a id="disaggregation"></a>

### 4.6 Disaggregation

- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - Running separate prefill and decode pools with vLLM.
- [DistServe](https://arxiv.org/abs/2401.09670) - **Paper, 2024**. OSDI 2024 paper on placing prefill and decode on separate GPUs to meet TTFT and TPOT targets.
- [Splitwise](https://arxiv.org/abs/2311.18677) - **Paper, 2023**. Splitting prompt and token phases onto different machines, including different GPU types.
- [Disaggregated Serving in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog05_Disaggregated_Serving_in_TensorRT-LLM.md) - TensorRT-LLM's disaggregated serving design.
- [SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) - Configuring prefill/decode disaggregation in SGLang.
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) - Datacenter-scale distributed inference serving framework with disaggregated prefill and decode.
- [NIXL](https://github.com/ai-dynamo/nixl) - NVIDIA Inference Xfer Library for moving KV cache between GPUs, nodes and storage, used by Dynamo, vLLM and llm-d.
- [llm-d Architecture](https://llm-d.ai/docs/architecture) - Component design and topology of llm-d.
- [llm-d](https://github.com/llm-d/llm-d) - Kubernetes-native distributed LLM inference with KV-cache-aware scheduling.
- [llm-d P/D Disaggregation](https://llm-d.ai/docs/architecture/disaggregation/pd-disaggregation) - When to split prefill and decode in llm-d and how KV-cache is transferred between them.

<a id="structured-output"></a>

### 4.7 Structured output, sampling and tool calling

- [How to Generate Text](https://huggingface.co/blog/how-to-generate) - Greedy search, beam search, top-k and top-p sampling compared.
- [Inference Parameters](https://handbook.modular.com/model-interaction/inference-parameters/) - LLM Inference Handbook chapter on temperature, top-p, penalties and other generation controls.
- [Min-p Sampling](https://arxiv.org/abs/2407.01082) - **Paper, 2024**. Truncating the token distribution relative to the top token's probability.
- [XGrammar](https://github.com/mlc-ai/xgrammar) - Fast grammar-constrained decoding engine, integrated into serving engines including vLLM and SGLang; backend defaults depend on engine version.
- [llguidance](https://github.com/guidance-ai/llguidance) - Low-latency constrained decoding for JSON schemas, regular expressions and context-free grammars.
- [Outlines](https://github.com/dottxt-ai/outlines) - Library for generating structured output from LLMs using regex, JSON schemas and grammars.
- [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer) - Enforcing JSON schema or regex output by filtering allowed tokens.
- [vLLM Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) - Guided JSON, regex, choice and grammar decoding in vLLM.
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/) - How vLLM parses model-specific tool-call formats into OpenAI-compatible responses.
- [vLLM Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) - Separating reasoning content from the final answer for reasoning models.

<a id="beyond-llms"></a>

## 5. Multimodal inference

<a id="omni-modal-inference"></a>

### 5.1 Omni-modal inference

- [What Is an Omni-Model?](https://www.nvidia.com/en-us/glossary/omni-model/) - NVIDIA definition of models that take and produce several modalities.
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) - **Paper, 2025**. Thinker-talker architecture that reads text, images, audio and video and streams speech back.
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) - End-to-end omni-modal model that understands text, audio, images and video and generates speech in real time.
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - vLLM framework for serving omni-modal models.
- [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204) - **Paper, 2026**. Paper on serving multimodal pipelines as independently scheduled stages.

<a id="vision-language-models"></a>

### 5.2 Vision-language models

- [Vision Language Models Explained](https://huggingface.co/blog/vlms) - Hugging Face introduction to VLM architectures, training and how to choose one.
- [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025) - Hugging Face update on newer VLM trends, including any-to-any models, reasoning, small models and MoE decoders.
- [Vision Language Models](https://www.baseten.co/inference-engineering/book/06-modalities/vision-language-models) - Inference Engineering book chapter on serving models that take images and video, and the cost of the vision encoder.
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) - Passing images, video and audio to multimodal models in vLLM.
- [Vision Language Model Prompt Engineering Guide for Image and Video Understanding](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/) - NVIDIA guide to feeding single images, multiple images and video frames to VLMs.

<a id="speech"></a>

### 5.3 Speech

- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) - **Paper, 2022**. The Whisper paper.
- [Silero VAD](https://github.com/snakers4/silero-vad) - Lightweight voice activity detector for finding speech before transcription.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Whisper transcription reimplemented on CTranslate2 for faster inference.
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) - Port of Whisper to C/C++ for CPU, GPU and on-device transcription.
- [WhisperX](https://github.com/m-bain/whisperX) - Batched Whisper transcription with word-level timestamps and speaker diarization.
- [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) - NVIDIA's speech toolkit, home of the Parakeet and Canary ASR models.
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - Offline streaming ASR, TTS, VAD and diarization on ONNX Runtime for mobile, embedded and server targets.
- [Piper](https://github.com/OHF-Voice/piper1-gpl) - Fast, local neural text-to-speech engine.
- [Kokoro](https://github.com/hexgrad/kokoro) - Small 82M-parameter open-weight text-to-speech model.
- [Pipecat](https://github.com/pipecat-ai/pipecat) - Open-source framework for real-time voice and multimodal agents that chains streaming ASR, LLM and TTS services.
- [LiveKit Agents](https://github.com/livekit/agents) - Framework for real-time voice AI agents over WebRTC, with turn detection and interruption handling.
- [What Is Speech-to-Speech?](https://deepgram.com/blog/what-is-speech-to-speech) - Cascaded versus end-to-end speech-to-speech pipelines and where each fits.
- [Moshi](https://github.com/kyutai-labs/moshi) - Speech-text foundation model and full-duplex dialogue framework with a streaming audio codec.

<a id="vision-and-video-analytics"></a>

### 5.4 Computer vision and video analytics

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Training, export and inference for YOLO detection, segmentation, pose and tracking models.
- [Ultralytics Export Formats](https://docs.ultralytics.com/modes/export) - Exporting YOLO models to ONNX, TensorRT, OpenVINO, Core ML and other deployment formats.
- [Triton Inference Server with Ultralytics YOLO](https://docs.ultralytics.com/guides/triton-inference-server) - Exporting a YOLO model to ONNX and serving it from a Triton model repository.
- [NVIDIA DeepStream SDK](https://github.com/NVIDIA/DeepStream) - GStreamer-based toolkit for multi-stream video analytics pipelines on NVIDIA GPUs and Jetson.
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) - Community implementation for running YOLO models inside DeepStream pipelines.
- [Savant](https://github.com/insight-platform/Savant) - Python framework for video analytics pipelines, built on DeepStream for NVIDIA GPUs and Jetson.
- [DL Streamer](https://github.com/open-edge-platform/dlstreamer) - Intel's GStreamer-based pipeline framework for video analytics on Intel CPUs, GPUs and NPUs.
- [Roboflow Inference](https://github.com/roboflow/inference) - Self-hosted server for running vision models and video-stream workflows on cloud or edge devices.

<a id="image-and-video-generation"></a>

### 5.5 Image and video generation

- [Diffusion Models](https://huggingface.co/blog/Esmail-AGumaan/diffusion-models) - Introductory explanation of how diffusion models are trained and sampled.
- [Diffusers](https://github.com/huggingface/diffusers) - Hugging Face library of diffusion models and pipelines for image, video and audio generation.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) - Node-graph interface, API and backend for running diffusion pipelines.
- [SGLang Diffusion](https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/) - SGLang's serving stack applied to image and video generation.
- [xDiT](https://github.com/xdit-project/xDiT) - Parallel inference engine for diffusion transformers, splitting image and video generation across GPUs.
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) - Inference and post-training framework for accelerated video generation.
- [vLLM-Omni Diffusion Guide](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion) - Attention backends and acceleration options for diffusion models in vLLM-Omni.
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - Diffusion model inference in C/C++ on ggml, for CPUs and consumer GPUs.
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378) - **Paper, 2023**. Distilling diffusion models to generate images in a few sampling steps.
- [DeepCache](https://github.com/horseee/DeepCache) - Training-free acceleration by reusing high-level U-Net features across denoising steps.
- [TeaCache](https://github.com/ali-vilab/TeaCache) - Training-free caching that skips redundant timesteps in video and image diffusion models.
- [Nunchaku](https://github.com/nunchux-ai/nunchaku) - 4-bit inference engine for diffusion models based on SVDQuant.

<a id="production"></a>

## 6. Production inference

<a id="containers-and-gpus"></a>

### 6.1 Containers and GPU enablement

- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) - Runtime hooks that expose NVIDIA GPUs to containers.
- [AMD Container Toolkit](https://github.com/ROCm/container-toolkit) - Runtime and tools that expose AMD GPUs to containers.
- [NVIDIA NIM](https://docs.nvidia.com/nim/) - Prebuilt, optimized inference microservice containers for NVIDIA GPUs.
- [BentoML](https://github.com/bentoml/BentoML) - Framework for packaging models as inference APIs, job queues and multi-model pipelines.
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) - Exposes NVIDIA GPUs as schedulable Kubernetes resources.
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) - Installs and manages NVIDIA drivers, device plugin and monitoring on Kubernetes.
- [AMD GPU Device Plugin](https://github.com/ROCm/k8s-device-plugin) - Exposes AMD GPUs as schedulable Kubernetes resources.
- [AMD GPU Operator](https://github.com/ROCm/gpu-operator) - Installs and manages AMD GPU drivers, device plugin and metrics on Kubernetes.

#### Model loading and cold starts

- [Model Streamer](https://github.com/dsx-ai-factory/model-streamer) - Concurrent streaming of safetensors weights from object storage or disk into GPU memory (formerly Run:ai Model Streamer).
- [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) - Faster safetensors loading using GPUDirect Storage and parallel reads.
- [Tensorizer](https://github.com/coreweave/tensorizer) - CoreWeave's serialization format and loader for streaming model weights from HTTP, S3 or disk.

#### Model artifact packaging and delivery

- [KServe OCI Storage (Modelcars)](https://kserve.github.io/website/docs/model-serving/storage/providers/oci) - Packaging models as OCI images so nodes cache and share them like container layers.
- [Kubernetes Image Volumes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/) - Mounting an OCI image as a read-only volume, which lets model weights ship separately from the serving image.
- [ModelPack Model Spec](https://github.com/modelpack/model-spec) - Open specification for packaging and distributing models as OCI artifacts.
- [KitOps](https://github.com/kitops-ml/kitops) - CNCF tool for packaging and versioning models, datasets and configuration as OCI artifacts.

<a id="kubernetes-serving"></a>

### 6.2 Kubernetes serving platforms

- [KServe](https://github.com/kserve/kserve) - Kubernetes platform for deploying generative and predictive models across frameworks.
- [Ray](https://github.com/ray-project/ray) - Distributed compute engine whose Ray Serve library serves models at scale.
- [vLLM Production Stack](https://github.com/vllm-project/production-stack) - vLLM's reference system for Kubernetes-native, cluster-wide deployment.
- [AIBrix](https://github.com/vllm-project/aibrix) - vLLM project's Kubernetes control plane for LLM serving, with LoRA management, routing and autoscaling.
- [KAITO](https://github.com/kaito-project/kaito) - CNCF Kubernetes AI Toolchain Operator that provisions GPU nodes and deploys open models.
- [HAMi](https://github.com/Project-HAMi/HAMi) - CNCF project for sharing and isolating heterogeneous GPUs on Kubernetes.
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) - Kubernetes API for deploying a group of pods as one unit, used for multi-node inference.
- [Kueue](https://github.com/kubernetes-sigs/kueue) - Kubernetes job queueing and quota management for sharing GPU capacity across teams.

#### GPU sharing and allocation

- [NVIDIA Multi-Instance GPU User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/) - Partitioning a GPU into isolated instances with dedicated memory and compute.
- [GPU Operator: GPU Sharing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) - Configuring time-slicing so several pods share one GPU on Kubernetes.
- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/) - The Kubernetes API for requesting and sharing devices such as GPUs with structured parameters.
- [DRA Driver for NVIDIA GPUs](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu) - Kubernetes DRA driver that allocates NVIDIA GPUs, MIG devices and shared GPUs.

<a id="routing-and-autoscaling"></a>

### 6.3 Gateways, routing and autoscaling

- [The Baseten Inference Stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack) - Runtime and infrastructure layers of a production stack, including geo-aware load balancing.
- [LiteLLM](https://github.com/BerriAI/litellm) - AI gateway and SDK that calls 100+ LLM APIs in OpenAI format, with load balancing and cost tracking.
- [Agent Router (formerly Envoy AI Gateway)](https://github.com/theagentrouter/agent-router) - Unified access to generative AI services, built on Envoy Gateway.
- [Portkey Gateway](https://github.com/Portkey-AI/gateway) - Open-source AI gateway with routing, retries, fallbacks and guardrails across LLM providers.
- [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) - Kubernetes SIG project that adds model-aware routing to Gateway API through InferencePool.
- [llm-d Router](https://github.com/llm-d/llm-d-router) - llm-d's scheduler that routes requests by prefix-cache hits, load and predicted latency.
- [KEDA](https://github.com/kedacore/keda) - Event-driven Kubernetes autoscaler, often used to scale inference pods on queue depth.
- [Karpenter](https://github.com/kubernetes-sigs/karpenter) - Kubernetes node autoscaler that provisions GPU nodes to fit pending pods.
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/) - Serving many LoRA adapters on one base model and routing requests to them.
- [S-LoRA](https://arxiv.org/abs/2311.03285) - **Paper, 2023**. Serving thousands of concurrent LoRA adapters with unified paging of adapter weights and KV cache.
- [Punica](https://arxiv.org/abs/2310.18547) - **Paper, 2023**. Batching requests for different LoRA adapters with a custom segmented-gather kernel.
- [vLLM Semantic Router](https://github.com/vllm-project/semantic-router) - Programmable layer that classifies requests and routes them to the right model.
- [Autoscaling Endpoints for LLM Inference](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) - Choosing scaling metrics, tuning scaling windows and budgeting for cold starts.
- [vLLM: Why Requests Take Hours Under Load](https://blog.dotieuthien.com/posts/vllm) - Diagnosing KV block exhaustion and queue starvation.

<a id="cloud-platforms"></a>

### 6.4 Cloud and GPU hosting platforms

#### Choosing a hosting model

- [Serverless vs. Self-hosted LLM Inference](https://handbook.modular.com/getting-started/serverless-vs-self-hosted-llm-inference/) - Trade-offs between managed APIs and self-hosted deployments in control, cost, scaling and compliance.
- [Bring Your Own Cloud](https://handbook.modular.com/getting-started/bring-your-own-cloud/) - The middle ground of managed serving software running in your own cloud account.
- [On-Prem LLM Deployments](https://handbook.modular.com/getting-started/on-prem-llms/) - When on-premises inference makes sense and what it requires to operate.

#### AWS (EKS)

- [Load and Serve Models on Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/ml-inference-load-serve-model.html) - Deploying vLLM on GPUs and streaming weights from S3 with the Run:ai Model Streamer to cut cold starts.
- [EKS Best Practices: Application Scaling and Performance](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-performance.html) - Model artifact management, serving frameworks and container startup optimization.
- [GPU Inference on EKS Auto Mode](https://aws.amazon.com/blogs/containers/how-to-run-ai-model-inference-with-gpus-on-amazon-eks-auto-mode/) - GPU node pools with AWS-managed node lifecycle and patching.
- [Introducing Disaggregated Inference on AWS Powered by llm-d](https://aws.amazon.com/blogs/machine-learning/introducing-disaggregated-inference-on-aws-powered-by-llm-d/) - Running llm-d prefill/decode disaggregation and wide expert parallelism on AWS.

#### Google Cloud (GKE)

- [Deploy GKE Inference Gateway Powered by llm-d](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway) - Managed gateway using llm-d scheduling.
- [Best Practices for Autoscaling LLM Inference Workloads on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/autoscaling) - Choosing HPA signals such as queue size and batch size over GPU utilization.
- [Tuning the GKE HPA to Run Inference on GPUs](https://cloud.google.com/blog/products/containers-kubernetes/tuning-the-gke-hpa-to-run-inference-on-gpus) - Experiments comparing autoscaling metrics for LLM servers.
- [Serve an LLM Using TPUs on GKE with KubeRay](https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray) - Deploying a RayCluster on a TPU node pool and serving an LLM with Ray Serve.

#### Azure (AKS)

- [Deploy an AI Model on AKS with the AI Toolchain Operator](https://learn.microsoft.com/en-us/azure/aks/ai-toolchain-operator) - Managed KAITO add-on that deploys open models with vLLM and provisions GPU nodes.
- [Use GPUs on AKS](https://learn.microsoft.com/en-us/azure/aks/use-nvidia-gpu) - Creating GPU node pools and installing the NVIDIA device plugin.
- [GPU Node Partitioning Strategies in AKS](https://learn.microsoft.com/en-us/azure/aks/concepts-gpu-partitioning) - Comparison of MIG, time-slicing and MPS for sharing GPUs.
- [Create a Multi-Instance GPU Node Pool in AKS](https://learn.microsoft.com/en-us/azure/aks/gpu-multi-instance) - Creating MIG-partitioned node pools on AKS.

#### GPU clouds

- [Serving Llama 3.1 with vLLM on Lambda](https://docs.lambda.ai/education/large-language-models/serving-llama-3-1-docker/) - Lambda's guide to running vLLM in Docker on on-demand A100 and H100 instances.
- [About CoreWeave Inference](https://docs.coreweave.com/products/inference) - CoreWeave's options for serving models: serverless, dedicated, or self-managed on its Kubernetes service.
- [vLLM on Vast.ai](https://docs.vast.ai/vllm-llm-inference-and-serving) - Vast.ai's template for serving LLMs with vLLM on marketplace GPUs.
- [Serve LLM Inference on E2E GPU Nodes](https://docs.e2enetworks.com/docs/myaccount/gpu/guides/serve-llm-inference/) - E2E Networks' guide to serving an LLM on its GPU virtual machines.

#### Managed inference endpoints

- [RunPod Serverless vLLM](https://docs.runpod.io/serverless/vllm/overview) - Deploying an OpenAI-compatible vLLM endpoint on RunPod Serverless that scales with traffic.
- [Replicate: Deploy a Custom Model](https://replicate.com/docs/get-started/deploy-a-custom-model) - Packaging a model with Cog and running it as a dedicated, autoscaling deployment on Replicate.
- [Cog](https://github.com/replicate/cog) - Replicate's open-source tool for packaging models as production-ready containers.
- [E2E Networks TIR Model Endpoints](https://docs.e2enetworks.com/docs/tir/Inference/ModelEndpoints/QuickStartGuide/) - Creating managed endpoints with vLLM, SGLang or Triton runtimes on E2E's TIR platform.
- [Modal: LLM Inference with vLLM](https://modal.com/docs/examples/llm_inference) - Modal's example of serving an OpenAI-compatible vLLM server on serverless GPUs.
- [Baseten Model Deployment](https://docs.baseten.co/deployment/concepts) - Deploying models on Baseten with Truss, from a config file to an autoscaling API.
- [Truss](https://github.com/basetenlabs/truss) - Baseten's open-source framework for packaging models into deployable containers.
- [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) - Deploying Hub models on managed, autoscaling infrastructure across cloud providers.
- [Together AI Dedicated Inference](https://docs.together.ai/docs/dedicated-endpoints/overview) - Running models on dedicated, single-tenant GPU endpoints on Together AI.

<a id="observability-and-reliability"></a>

### 6.5 Observability, security and debugging

- [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) - Prometheus exporter for NVIDIA GPU metrics, built on DCGM.
- [AMD Device Metrics Exporter](https://github.com/ROCm/device-metrics-exporter) - Exports AMD GPU metrics to Prometheus and other collectors.
- [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - Evolving conventions for spans and metrics covering model calls, tokens and latency; pin the convention version when instrumenting.
- [OpenLLMetry](https://github.com/traceloop/openllmetry) - OpenTelemetry instrumentation for LLM providers, frameworks and vector databases.
- [Helicone](https://github.com/Helicone/helicone) - Open-source LLM observability platform and gateway for request logging, cost and latency.
- Langfuse and Arize Phoenix, listed under [Evaluation](#model-strategy), also trace production LLM traffic.
- [Confidential Compute for AI Inference](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments) - How Chutes uses trusted execution environments for verifiable private inference on third-party GPUs.
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/) - Common security risks for LLM applications, from prompt injection to unbounded consumption.
- [ModelScan](https://github.com/protectai/modelscan) - Scans model files for unsafe serialized code, such as malicious pickle payloads.
- [Pickle Scanning](https://huggingface.co/docs/hub/security-pickle) - Why pickle-based checkpoints can execute code on load, and how the Hugging Face Hub scans for it.
- [Model Transparency](https://github.com/sigstore/model-transparency) - Sigstore and OpenSSF tooling for signing models and verifying their provenance.
- [Llama Guard](https://arxiv.org/abs/2312.06674) - **Paper, 2023**. LLM-based safeguard model for classifying prompts and responses.
- [NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) - Programmable input, output and dialog guardrails for LLM applications.
- [Token-Level Truth](https://vllm.ai/blog/2025-12-14-halugate) - HaluGate: token-level hallucination detection in vLLM Semantic Router, checked against tool outputs and context.
- [One Token to Corrupt Them All](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) - How AI21 traced a state-corruption bug in vLLM's scheduler and memory management for Mamba models.

<a id="cost-and-efficiency"></a>

### 6.6 Cost and efficiency

- [LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/) - NVIDIA method for turning benchmark results into cost per million tokens and total cost of ownership.
- [The Hidden Economics of LLM Inference](https://www.hebbia.com/blog/the-hidden-economics-of-llm-inference) - Queueing math and GPU utilization, and why pooled multi-tenant serving costs less than single-tenant serving.
- [Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling](https://arxiv.org/abs/2509.20241) - **Paper, 2025**. Per-query energy estimates and efficiency pathways at scale.
- [GPU Inefficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Why production GPUs sit underutilized.
- [Zeus](https://github.com/ml-energy/zeus) - Library for measuring and optimizing GPU energy consumption of AI workloads.
- [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/) - Energy-per-request measurements for open models across serving configurations.

<a id="batch-and-rollout"></a>

### 6.7 Offline batch inference, RL rollouts and reasoning budgets

#### Offline batch inference

- [vLLM Offline Inference](https://docs.vllm.ai/en/latest/serving/offline_inference/) - Running vLLM as a library for batch generation without an HTTP server.
- [Ray Data LLM](https://docs.ray.io/en/latest/data/working-with-llms.html) - Batch LLM inference over large datasets with vLLM on Ray Data.

#### RL rollout generation

- [verl](https://github.com/verl-project/verl) - RL post-training framework that uses vLLM and SGLang as rollout engines.
- [slime](https://github.com/THUDM/slime) - RL scaling framework that connects Megatron training with SGLang rollouts.

#### Reasoning budgets

- [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419) - **Paper, 2025**. Survey of methods for shortening reasoning traces and lowering their inference cost.
- [s1: Simple Test-Time Scaling](https://arxiv.org/abs/2501.19393) - **Paper, 2025**. Budget forcing: capping or extending a model's reasoning length at decode time.
- Separating reasoning tokens from the final answer is covered in [vLLM Reasoning Outputs](#structured-output).

<a id="serving-reliability"></a>

### 6.8 Serving reliability and deployment safety

- [Ray Serve Production Best Practices](https://docs.ray.io/en/latest/serve/production-guide/best-practices.html) - Request handling, timeouts and operational configuration for Ray Serve deployments.
- [Ray Serve Fault Tolerance](https://docs.ray.io/en/latest/serve/production-guide/fault-tolerance.html) - How Ray Serve recovers from replica, node and head-node failures.
- [Kubernetes Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) - Probes, termination grace periods and shutdown semantics for serving pods.
- [Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) - **Paper, 2023**. Virtual Token Counter (VTC), a fair scheduler that accounts for both input and output tokens per client.
- [Llumnix](https://arxiv.org/abs/2406.03243) - **Paper, 2024**. Rescheduling requests across serving instances with live KV-cache migration for load balancing and priorities.
- KV-cache exhaustion under load is covered in [vLLM: Why Requests Take Hours Under Load](#routing-and-autoscaling), and engine-side signals for SLOs in [vLLM Production Metrics](#latency-metrics).

<a id="learning"></a>

## 7. Learning resources

<a id="courses-and-books"></a>

### 7.1 Courses and books

- [Inference Engineering](https://www.baseten.co/inference-engineering/book) - Philip Kiely's book on building faster, cheaper and more reliable generative AI inference systems.
- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - TPU/GPU architecture, communication and parallelism for training and inference.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Hugging Face's interactive guide to parallelism and communication for training LLMs on GPU clusters.
- [Hands-On LLM Serving and Optimization](https://www.oreilly.com/library/view/hands-on-llm-serving/9798341621480/) - **Paid book**. Chi Wang and Peiheng Hu's O'Reilly book on hosting and optimizing LLMs at scale.
- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) - Song Han's course on pruning, quantization, distillation and efficient LLM inference.
- [CMU 15-442/15-642: Machine Learning Systems](https://mlsyscourse.org/) - Course on the systems behind ML, from GPU programming to LLM serving.
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/) - Course that builds a language model end to end, including kernels, parallelism and inference.
- [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) - **Paid book**. Hwu, Kirk and El Hajj's textbook on CUDA and parallel programming patterns.

<a id="repositories-and-reading-lists"></a>

### 7.2 Repositories and reading lists

- [100 Days of LLM Inference](https://github.com/elizabetht/100-days-of-inference) - Day-by-day runnable experiments that follow the Inference Engineering book, from KV cache and kernels to quantization, speculative decoding and disaggregated serving.
- [LLM Inference Engineering](https://github.com/amitshekhariitbhu/llm-inference-engineering) - Step-by-step blog series from autoregressive generation, prefill and decode to KV cache, PagedAttention, continuous batching, vLLM, SGLang and GPUs.
- [AI Inference Resources](https://github.com/aerlabsAI/ai-inference-resources) - AER Labs' tiered reading guide covering LLM serving, GPU kernels, attention, quantization, distributed inference and production deployment.
- [Inference Engineering Library](https://inference-engineering-library.danialh.chatgpt.site/) - Community-shared books, courses, code and research for inference engineering, filterable by format and level.
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures) - Slides and code from the GPU MODE lecture series on CUDA, Triton and inference kernels.
- [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294) - **Paper, 2024**. Taxonomy of data-, model- and system-level inference optimizations.
- [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) - Companion list of LLM and VLM inference papers with code.

<a id="talks"></a>

### 7.3 Talks

- [The Rise of vLLM](https://www.youtube.com/watch?v=WLl8D1nyaW8) - How vLLM grew from a research project into a widely used engine.
- [SGLang: An Efficient Open-Source Framework for Large-Scale LLM Serving](https://www.youtube.com/watch?v=w9-AYqIhHRo) - Ray Summit 2025 talk on SGLang's architecture and production use.
- [How to Become an Inference Engineer](https://www.youtube.com/watch?v=LUGxx1XNqcM) - Baseten talk on the skills and path to becoming an inference engineer.
- [Inference OSS Ecosystem featuring vLLM](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday06/) - Disaggregation and wide-EP for MoE on GB200 racks.
- [Inside the MAX Framework](https://www.youtube.com/watch?v=WK5dVQ8vhbU) - Modular meetup walkthrough of the MAX inference stack.
- [Introduction to LLM Serving with SGLang](https://www.youtube.com/watch?v=Ahtaha9fEM0) - AI Engineer workshop by Philip Kiely and Yineng Zhang on serving LLMs with SGLang.
- [GPU MODE](https://www.youtube.com/@GPUMODE) - Recorded lectures on GPU programming, kernels and inference systems.

<a id="contributing"></a>

## Contributing

Read [the contribution guide](CONTRIBUTING.md) before submitting. Use the issue templates to propose a topic.

<a id="license"></a>

## License

This repository's original content is licensed under the [MIT License](LICENSE). Linked projects, documentation, papers, images and code retain their respective licenses.
