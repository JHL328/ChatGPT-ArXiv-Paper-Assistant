# Personalized Daily ArXiv Papers 2026-10-31

| *[gpt-5]*   | Prompt   | Completion   | Total   |
|:-----------:|:--------:|:------------:|:-------:|
| **Token**   | 0        | 0            | 0       |
| **Cost**    | $0.0     | $0.0         | $0.0    |

Total arXiv papers: 605

Total scanned papers: 392

Total relevant papers: 2

**Table of contents with paper titles:**

1. [Proving and Improving the Platonic Representation Hypothesis with JEPAs](#user-content-link1)
**Authors:** Randall Balestriero, Yann LeCun

2. [Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live IOS Apps](#user-content-link2)
**Authors:** Peitong Duan, Khashayar Gatmiry, Jeffrey Nichols, Samy Bengio, Emmanuel Abbe

---

## 1. [Proving and Improving the Platonic Representation Hypothesis with JEPAs](https://arxiv.org/abs/2610.39302) <a id="link1"></a>

**ArXiv ID:** 2610.39302

**Authors:** Randall Balestriero, Yann LeCun

**Abstract:** Whether models trained on different modalities converge to the same representation of the world is central to AI. The Platonic Representation Hypothesis (PRH) conjectures that they do as they scale, but without a proof it offers neither a guarantee nor a recipe. We turn the hypothesis into a theorem. First, we prove a \emph{Platonic Representation Theorem}: Joint-Embedding Predictive Architectures (JEPAs) trained separately on any number of modalities learn the same representation up to one rotation whenever their positive pairs are two observations of one scene--\emph{good views}--and their misalignment is bounded otherwise. Second, we show that next-token prediction is a JEPA with \emph{bad views}, parts of a single document, which explains why language models align with vision only through scale. Third, we validate the theory from 145k captions to the texts of 940M web images, and uncover a scaling law of alignment that predicts unseen runs within seed noise. Good views need 42 times fewer training samples than next-token prediction on the same corpus to reach the same alignment, and a 109M-parameter text JEPA aligns with DINOv2 and CLIP's image encoders better than every language model we evaluated, up to 72B parameters, and than CLIP's text encoder: next-token prediction is a remarkably inefficient route to platonic representations. As a showcase, one rotation turns separately trained encoders into an image captioner (CIDEr .838 on COCO). Even that rotation needs no pair: on the latest pair-free benchmark, our text JEPA beats every zero-shot accuracy reported with language models up to 8B parameters, e.g., 64\% against at most 56\% top-5 on ImageNet-100. Positive views, and not scale alone, are thus the lever to build platonic representations.

**Comment:** Author match



---

## 2. [Online Dense Reinforcement Learning with Planning-Grounding Agent for UI Navigation on Live IOS Apps](https://arxiv.org/abs/2610.39264) <a id="link2"></a>

**ArXiv ID:** 2610.39264

**Authors:** Peitong Duan, Khashayar Gatmiry, Jeffrey Nichols, Samy Bengio, Emmanuel Abbe

**Abstract:** While frontier vision-language models demonstrate strong capabilities in user interface (UI) navigation, their high inference costs prohibit everyday use. Conversely, smaller models trained via supervised fine-tuning (SFT) or offline reinforcement learning often memorize static trajectories, leading to brittle execution and an inability to recover from errors. To address these limitations, we propose a novel online reinforcement learning framework that trains UI navigation agents directly on live iOS applications. Our framework decouples the agent into two specialized components: a Planning agent and a UI Grounding model. We rely on an LLM-as-a-Judge to provide comprehensive reward signals, combining dense, step-level feedback to evaluate intermediate progress with a terminal reward for overall task completion. To effectively leverage this dual reward structure, we optimize the agent using Proximal Policy Optimization (PPO) with step-level advantage estimation. Furthermore, we introduce GiGPPO, a novel neighbor-conditioned value estimation method that improves critic stability by leveraging reward signals from visually similar states across concurrent trajectories. To support this online training paradigm at scale, we introduce an efficient infrastructure that orchestrates hundreds of concurrent virtual iOS devices. Evaluations on rigorous benchmarks, including UINavBench, iOSWorld, and AndroidWorld, demonstrate that our online RL trained agents substantially outperform both SFT and terminal-only RL baselines while accelerating training convergence. Notably, our 8B RL agents achieve nearly a sevenfold improvement over the SFT baseline in out-of-distribution generalization to Android environments, while outperforming Gemini-3.5-Flash and tying with Gemini-3.1-Pro on real iOS applications.

**Comment:** Author match



---

# Paper Selection Prompt

## System Prompt

> You are a helpful paper reading assistant whose job is to read daily posts from ArXiv and identify a few papers that your friend will enjoy reading.
> Your job is to carefully read the paper titles and abstracts below and find the ones that match the criteria below.

## User Prompt

> ## Instructions
> 
> Write the response in JSONL format with {ARXIVID, COMMENT, RELEVANCE, NOVELTY} on each line, one for each paper.
> 
> - ARXIVID: should be the ArXiv ID.
> - COMMENT: should identify whether there is a criteria that match the paper very closely. These matches should not be based on general terms like "language modeling" or "advancements" and should specifically refer to a criterion. No need to mention the non-matching criteria.
> - RELEVANCE: should be a score from 1-10.
> - NOVELTY: should be a score from 1-10.
> 
> ## Scoring Criteria
> 
> > The "Relevance" score measures how closely the paper aligns with the core topics of the prompt.
> > The "Novelty" score assesses the originality and impact of the paper.
> > They are two **ORTHONORMAL** axes and **SHOULD NOT** be confused with each other.
> 
> ### Relevance Scoring
> 
> - Relevance 9-10 (Completely Relevant)
>   - Focus: Fully aligned with core topics with no deviation, score the highest if contains relevant keywords in it.
>   - Examples: Papers focused on foundational methods or theoretical research, whose titles contain topic keywords like "MoE".
> 
> - Relevance 7-8 (Relevant)
>   - Focus: Retain a solid link to the main research area, though may touch on peripheral elements.
>   - Examples: Papers research on the fundamental part of MoE through a less critical aspect like its behavior in GNN.
> 
> - Relevance 5-6 (Borderline)
>   - Focus: Maintains a link to the core topic but also extends into at least one other domain/area beyond the primary focus.
>   - Examples: Work referencing MoE centered on reinforcement learning.
> 
> - Relevance 3-4 (Irrelevant)
>   - Focus: Largely outside our interests with no association to our topics.
>   - Examples: Application-focused papers like using MoE to solve a problem in the real world.
> 
> - Relevance 1-2 (Ignore)
>   - Focus: Purely unrelated to our topics. Completely a different domain.
>   - **Exception**: If the paper hints at a cutting-edge, radically new direction that could eventually transform the primary domain, consider a score of 9–10 despite initial appearances. (Usually a very rare concept that belongs to the fundamental research)
> 
> ### Novelty Scoring
> 
> - Novelty 9-10 (Breakthrough)
>   - Definition: Groundbreaking methods/theory introducing new directions or solving major challenges.
>   - Examples: Entirely new paradigm for foundational models; a novel theory transforming representation learning.
> 
> - Novelty 7-8 (Improvements)
>   - Definition: Substantial insights/enhancements, though not a full paradigm shift.
>   - Examples: Modifications on existing methods yielding significantly better results.
> 
> - Novelty 5-6 (Borderline)
>   - Definition: Incremental contributions with possible long-term benefits, not immediately transformative.
>   - Examples: Moderately novel extension to an existing architecture; refining current methods without fundamentally altering them.
> 
> - Novelty 3-4 (Tangential)
>   - Definition: Minor or domain-specific improvements with limited broader impact.
>   - Examples: Slight modifications to known methods with strange motivation; purely engineering jobs like a new benchmark/dataset.
> 
> - Novelty 1-2 (Low)
>   - Definition: Minimal originality, applying standard approaches without real innovation.
>   - Examples: Using an off-the-shelf model without adding new insights; purely application-driven studies like finetuning a pretrained model using existing methods.
> 
> ## Papers
> 
> [PAPER LIST HERE]
> 
> ## Relevant Topics
> 
> Use the following relevance criteria to focus on foundational research. Keep **relevant** papers and filter out **irrelevant** ones. Avoid purely **application-driven** work.
> 
> 1. Model Architecture
>    - Relevant: Mixture-of-Experts (MoE), Transformers, Conditional/Dynamic Networks, Autoencoders, analysis/innovations on existing architectures.
>    - Irrelevant: Merely using existing architectures for a certain task without insights into the structure themselves.
> 
> 2. Model Compression and Efficiency
>    - Relevant: Sparsity, pruning, quantization, low-rank approaches, cache, or other algorithmic/theoretical efficiency breakthroughs.
>    - Irrelevant: Straightforward applications of existing compression methods to new tasks.
> 
> 3. High Performance Computing
>    - Relevant: Algorithmic or systems-level innovations enabling training of large-scale models, distributed training techniques, memory optimization.
>    - Irrelevant: Incremental engineering improvements without novel algorithmic contributions.
> 
> 4. Representation Learning
>    - Relevant: Insights into how deep networks encode information, feature/dictionary learning, sparse/contrastive methods, training dynamics in neural networks.
>    - Irrelevant: Standard applications of known techniques lacking new theoretical or methodological contributions.
> 
> 5. ML Systems
>    - Goal: Keep ML-Systems work that provides fundamental, generalizable systems/algorithmic insights for training, inference, or deployment — not one-off application engineering.
>    - Relevant: 
>       - Distributed training algorithms and optimizations with theoretical/empirical scalability analysis (e.g., new sync/async protocols, communication compression with provable/empirical benefits).
>       - Memory / storage / I/O management improvements for very large models (hierarchical memory, recompute/checkpoint strategies, rematerialization optimizations).
>       - Communication & networking innovations (efficient AllReduce variants, topology-aware scheduling, bandwidth/latency–aware strategies).
>       - Compiler & automatic code-generation advances that enable operator fusion, memory scheduling, quantization-friendly IR passes.
>       - Heterogeneous acceleration & hardware–software co-design (CPU–GPU–NPU scheduling, kernel-level innovations with measurable gains).
>       - Inference-serving systems with strong evidence of low-latency / high-throughput tradeoffs, model-parallel + pipeline concurrency strategies, SLA-aware resource elasticity.
>       - Reproducible benchmarks & measurement methodologies that reveal system behavior and provide open tools/protocols.
>       - Algorithm–system co-design (e.g., systems built specifically for sparse/low-rank models, joint approximations that trade accuracy for system efficiency).
>       - Work with convincing quantitative/theoretical analysis, ablations, and results that generalize across topologies / hardware / model scales.
> 
>    -Irrelevant (Filter out):
>       - Papers that simply apply an existing framework/library to a dataset and report speedups without new system/algorithmic design.
>       - Purely application-focused engineering for a single domain (medical imaging, autonomous driving, etc.) without extracting generalizable system principles.
>       - Deployment notes or single-node config checklists without system-level analysis or broader lessons.
> 
>    - Practical filters / judging criteria:
>       - Does the paper include publicly reproducible code or benchmarks?
>       - Does it extract general principles or design patterns (not only case-specific optimizations)?
>       - Is there theoretical / complexity / communication-cost analysis or large-scale, multi-setting empirical validation?
>       - Does it address low-level kernels / communication / compilation / memory or propose a new system paradigm (e.g., new parallelism model, hierarchical storage design, combined algorithm/system optimization)?
> 
> **Keywords:**
> 
> - Relevant: Mixture of Experts (MoE), Representation Learning, Compression/Efficiency, Sparse/Sparsity, Pruning, Quantization, Low-rank, Foundation Model, etc.
> - Irrelevant: Reinforcement Learning, Transfer Learning, Federated Learning, Online Learning, Diffusion Models, etc.
> - Application: Image Segmentation, Medical Imaging, 3D Vision, Video Understanding, Information Retrieval, Summarization, Recommendation Systems, Machine Translation, Speech Recognition, Signal Processing, Spatial/Temporal Modeling, Time Series, Knowledge Graph, etc.