# Parallel Inference Guide

This document provides a detailed introduction to TeleFuser's distributed parallel inference architecture, including principles, configuration methods, and usage examples.
For tensor dataflow, synchronization, transport ownership, and performance invariants, see the
[Communication Architecture](communication.md).

## Overview

TeleFuser provides multi-dimensional parallel inference capabilities, supporting the following parallel strategies:

| Parallel Type | Description | Use Case |
|--------------|-------------|----------|
| **Data Parallel (DP)** | Replicate model across GPUs, process different data in parallel | Throughput optimization |
| **CFG Parallel** | Parallel computation of positive/negative prompts | CFG acceleration |
| **Sequence Parallel (SP)** | Split long sequences across multiple GPUs | Long video generation |
| **Pipeline Parallel (PP)** | Split model layers across multiple GPUs | Large model inference |
| **Tensor Parallel (TP)** | Split tensor dimensions across multiple GPUs | Large model inference |

## Architecture Design

### Device Mesh Layout

TeleFuser uses PyTorch DeviceMesh to manage distributed parallelism with dimension order:

```
DP -> CFG -> SP (ring, ulysses) -> PP -> TP
```

```python
from telefuser.distributed import create_device_mesh_from_config
from telefuser.core.config import ParallelConfig

config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    dp_degree=1,
    cfg_degree=2,
    sp_ulysses_degree=2,
    sp_ring_degree=1,
    pp_degree=1,
    tp_degree=1,
)

device_mesh = create_device_mesh_from_config(config)
```

### Core Modules

```
telefuser/distributed/
├── device_mesh.py      # DeviceMesh creation and process group management
├── collectives.py      # Shared contiguous gather and reduction primitives
├── pp_comm.py          # Pipeline parallel P2P communication
├── ulysses_comm.py     # Ulysses All-to-All communication primitives
├── ring.py             # Ring Attention P2P communication
├── parallel_shard.py   # Sequence parallel tensor shard/unshard
├── vae_spatial.py      # Height-sharded VAE halo exchange
├── fsdp.py             # FSDP data parallel
└── tp_parallelize.py   # Tensor parallel utilities
```

Model code owns tensor layout and reconstruction semantics, while reusable collective buffer allocation and reduction
submission stay in `collectives.py`. Strategy-specific protocols remain in their corresponding modules.

## Sequence Parallelism

Sequence parallelism is used to process very long sequences (e.g., long videos) by splitting the sequence dimension across multiple GPUs.

### Ulysses Attention

Sequence parallelism based on All-to-All communication:

**Principle**:
1. Each GPU holds a portion of the sequence
2. Redistribute heads via All-to-All
3. Each GPU has complete sequence but partial heads
4. After local attention computation, restore via All-to-All

**Data Flow**:
```
Input: (B, S_LOCAL, H_GLOBAL, D)
  -> All-to-All QKV -> (B, S_GLOBAL, H_LOCAL, D)
  -> Local attention computation
  -> All-to-All O -> (B, S_LOCAL, H_GLOBAL, D)
```

**Characteristics**:
- Communication overhead: 2 All-to-All operations (QKV + Output)
- Suitable for medium-length sequences
- Requires number of heads to be divisible by GPU count

When the installed `tf-kernel` wheel contains the Ulysses CUDA IPC operators and every rank in the Ulysses process
group is on the same host, TeleFuser uses the source-built Copy Engine backend for grouped Q/K/V scatter. It writes
directly into each peer final-layout target buffer, keeps up to 12 target allocations in a tag/shape/dtype LRU cache,
and fans out over one high-priority copy stream. Eviction synchronizes participating devices before closing peer
mappings. Q, K, and V stay as separate submissions so projection compute can overlap
with communication, while the three transfers share one CUDA stream-memory handshake that does not occupy an SM.
Single collectives and output gather stay on the faster PyTorch/NCCL path. Multi-host groups, missing kernels,
and unsupported CUDA IPC configurations also use the PyTorch/NCCL fallback.

### Ring Attention

Sequence parallelism based on P2P communication:

**Principle**:
1. Each GPU holds a portion of Q and a portion of K/V
2. K/V rotates in a ring of GPUs
3. Each GPU sees all K/V blocks sequentially
4. Uses online softmax to merge attention outputs

**Algorithm Flow**:
```python
for step in range(world_size):
    # 1. Compute attention with current KV chunk
    out, lse = attention(q, k, v)
    
    # 2. Send current KV to next GPU
    # 3. Receive new KV from previous GPU
    next_k, next_v = send_recv_kv(k, v)
    
    # 4. Merge results using online softmax
    out, lse = merge_attn_states(prev_out, prev_lse, out, lse)
    
    # 5. Update KV
    k, v = next_k, next_v
```

**Characteristics**:
- Supports arbitrary length sequences
- Communication can overlap with computation
- Requires attention implementation with log-sum-exp support

### USP (Unified Sequence Parallelism)

Combined Ulysses + Ring strategy for larger scale parallelism:

**Principle**:
1. Ring dimension: sequence splitting
2. Ulysses dimension: head splitting
3. Two strategies complement each other, supporting more GPUs

**Configuration Example**:
```python
# 4 GPU: ring=2, ulysses=2
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    sp_ring_degree=2,
    sp_ulysses_degree=2,
)
```

### Async Ulysses (async_usp_forward)

Asynchronous All-to-All implementation, overlapping computation and communication:

```python
# Initiate async All-to-All
q_wait = ulysses_scatter_heads(q, group, tag="q", barrier=False)
k_wait = ulysses_scatter_heads(k, group, tag="k", barrier=False)
v_wait = ulysses_scatter_heads(v, group, tag="v")

# Wait for completion
q = q_wait()
k = k_wait()
v = v_wait()

# Compute attention
x = attention(q, k, v)

# Async All-to-All output
out_wait = ulysses_gather_heads(x, group, num_heads=num_heads)
out = out_wait()
```

## Pipeline Parallelism (PP)

Split model layers across multiple GPUs for large model inference.

### Cross-worker tensor channels

Independent `ParallelWorker` groups can connect adjacent stages with a `WorkerTensorChannel`. CPU tensors use
multiprocessing shared memory. CUDA tensors use two bounded producer-owned IPC slots per stable tensor profile; each
IPC allocation is opened once and then reused. Pool handles and rank-local offsets travel as private
`WorkerTensorRef` metadata through the existing control path. Reusable interprocess CUDA events order producer
staging before consumer copies and order the next slot write after every consumer copy. Consumers record completion
events before publishing generation acknowledgements; the producer waits on those events in its staging stream
without device-wide synchronization.

```python
from telefuser.worker import ParallelWorker, WorkerTensorChannel

latent_channel = WorkerTensorChannel(consumer_world_size=vae_parallel_config.world_size)
denoise_worker = ParallelWorker(
    denoise_stage,
    tensor_output_channel=latent_channel,
    tensor_output_methods=("denoise",),
)
vae_worker = ParallelWorker(vae_stage, tensor_input_channels=(latent_channel,))
```

Channel setup is not automatic. The pipeline that owns the stage topology must create and retain each channel, pass it
to both `ParallelWorker` endpoints, and close it after the consumer and producer workers stop. Actor or scheduler
edges carry only `WorkerTensorRef` metadata; cancellable actor flows must release terminal references through the
consumer worker. See [Communication Architecture](communication.md#cross-stage-worker-and-actor-dataflow) for the
complete cross-stage wiring and lifecycle contract.

Set `shard_dim` when consumer ranks operate on disjoint tensor slices. The producer stages the tensor once and each
rank copies only its slice. LingBot's spatial VAE uses `shard_dim=-2`, so aggregate peer-copy traffic stays at one
logical latent instead of growing with the VAE world size. Including producer-local staging, the device-copy budget is
two logical latents. At most eight stable CUDA tensor profiles are pooled per channel; additional dynamic profiles
fall back to PyTorch CUDA IPC instead of growing retained HBM without bound.

This is a point-to-point, single-producer/single-consumer-group path. Enable it only for outputs whose complete
consumer set is the connected worker group. Calls that need to inspect a tensor in the parent may pass
`_tensor_transport=False`. Start both workers before submitting work, stop consumers before producers, then close the channel,
preserve producer order at the consumer, and treat transported tensors as immutable until the consumer finishes.
Schedulers that cancel a terminal artifact must call the consumer worker's
`discard_tensor_refs(ref, sync=True)` in producer order. This releases CPU shared memory or CUDA IPC storage without
materializing the tensor in the parent. A normal receive also discards older cancelled FIFO entries as a fallback.

Regular worker dispatch also sends shared-memory or CUDA IPC handles to each rank and lets the receiving rank perform
the final device placement. The parent does not allocate a temporary copy on every target GPU.

Use the local SGLang checkout to run the end-to-end latency gate on the same GPUs and tensor shape:

```bash
python tools/validation/benchmark_tensor_channel_vs_sglang.py
```

The comparison includes producer staging, metadata transport, target copy, target synchronization, and slot
acknowledgement for both implementations. Its default 200-sample gate rejects p50 regressions above 5% and p95
regressions above the larger of 10% or 0.05 ms, accounting for sub-millisecond multiprocessing scheduling jitter.

### Principle

```
Stage 0 (GPU 0):  Embedding + Layers [0:N/4]
       ↓ send hidden states
Stage 1 (GPU 1):  Layers [N/4:N/2]
       ↓ send hidden states
Stage 2 (GPU 2):  Layers [N/2:3N/4]
       ↓ send hidden states
Stage 3 (GPU 3):  Layers [3N/4:N] + Head
       ↓ output
```

### P2P Communication

```python
from telefuser.distributed import PipelineP2PComm, get_pp_group

pp_group = get_pp_group(device_mesh)
comm = PipelineP2PComm(pp_group)

# Send hidden states to next stage
if not comm.is_last_stage:
    comm.send_latent(hidden_states)

# Receive hidden states from previous stage
if not comm.is_first_stage:
    hidden_states = comm.recv_latent(shape=latent_shape)
```

### Layer Assignment

```python
# Automatically assign layers to stages
start_idx, end_idx = comm.get_stage_indices(num_layers)

# Example: 40 layers, 4 stages
# Stage 0: [0:10]
# Stage 1: [10:20]
# Stage 2: [20:30]
# Stage 3: [30:40]
```

### PP Forward Implementation

```python
def pp_forward(self, x, timestep, context, latent_shape, **kwargs):
    # First stage: Embedding + first blocks
    if self.is_pp_first_stage:
        x = self.patch_embedding(x)
        x, grid_size = self.patchify(x)
        x = self.forward_blocks_pp(x, timestep, context, **kwargs)
        self.pp_comm.send_latent(x)
        return None
    
    # Middle stages: Receive + Process + Send
    elif not self.is_pp_last_stage:
        x = self.pp_comm.recv_latent(shape=latent_shape)
        x = self.forward_blocks_pp(x, timestep, context, **kwargs)
        self.pp_comm.send_latent(x)
        return None
    
    # Last stage: Receive + Process + Output
    else:
        x = self.pp_comm.recv_latent(shape=latent_shape)
        x = self.forward_blocks_pp(x, timestep, context, **kwargs)
        x = self.head(x)
        return x
```

## CFG Parallelism

Parallel computation of Classifier-Free Guidance positive/negative prompts:

### Principle

Standard CFG computation:
```python
noise_pred = noise_neg + cfg_scale * (noise_pos - noise_neg)
```

CFG parallelism assigns positive and negative to different GPUs:

```
GPU 0: Compute noise_pos (positive prompt)
GPU 1: Compute noise_neg (negative prompt)
Then: All-Gather to merge results
```

### Usage

```python
# 2 GPU CFG parallel
config = ParallelConfig(
    device_ids=[0, 1],
    cfg_degree=2,
)

# Enable in model
dit.enable_cfgp()
```

### Implementation

```python
# Shard
cfg_parallel_shard(device_mesh, [x, timestep, context, ...])

# Select positive/negative based on CFG rank
cond_flag = False if get_cfg_rank(device_mesh) else True

# Compute
output = model(x, context, cond_flag=cond_flag)

# Merge
output = cfg_parallel_unshard(device_mesh, [output])[0]
```

## Data Parallelism (DP)

Use FSDP for data parallel training/inference:

### FSDP1

```python
from telefuser.distributed.fsdp import shard_model

model = shard_model(
    model,
    device_id=device_id,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    wrap_module_names=["blocks"],
    param_dtype=torch.bfloat16,
)
```

### FSDP2

```python
from telefuser.distributed.fsdp import shard_model_fsdp2

model = shard_model_fsdp2(
    model,
    wrap_module_names=["blocks"],
    param_dtype=torch.bfloat16,
)
```

## Tensor Parallelism (TP)

Split tensor dimensions across multiple GPUs:

### Usage

```python
from telefuser.distributed.tp_parallelize import parallelize_module
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

tp_plan = {
    "self_attn.q": ColwiseParallel(),
    "self_attn.k": ColwiseParallel(),
    "self_attn.v": ColwiseParallel(),
    "self_attn.o": RowwiseParallel(),
    "ffn.0": ColwiseParallel(),
    "ffn.2": RowwiseParallel(),
}

model = parallelize_module(model, device_mesh, tp_plan)
```

### Notes

- SP and TP may be enabled together as independent device-mesh dimensions when the selected pipeline supports the combination
- Need to ensure number of heads is divisible by TP degree

## Worker Implementations

### ParallelWorker

Multi-process parallel worker using `multiprocessing.spawn`:

```python
from telefuser.worker import ParallelWorker

worker = ParallelWorker(stage)

# Call method
result = worker.process(latents, ...)

# Cleanup
del worker
```

**Features**:
- One process per GPU
- Automatic process group initialization
- Supports sync/async calls

### RayWorker

Ray cluster distributed worker:

```python
from telefuser.worker import create_ray_worker

worker = create_ray_worker(stage, enable_parallel=True)
result = worker.process.remote(latents, ...)
```

## Configuration Reference

### ParallelConfig

```python
@dataclass
class ParallelConfig:
    device_ids: list | None = None        # GPU ID list
    dp_degree: int = 1                    # Data parallel degree
    cfg_degree: int = 1                   # CFG parallel degree
    sp_ulysses_degree: int = 1            # Ulysses sequence parallel degree
    sp_ring_degree: int = 1               # Ring sequence parallel degree
    pp_degree: int = 1                    # Pipeline parallel degree
    tp_degree: int = 1                    # Tensor parallel degree
    enable_fsdp: bool = False             # Enable FSDP
    timeout: int = 600                    # Timeout in seconds
    queue_with_cpu: bool = False          # Use CPU queue
    worker_intra_op_threads: int = 1      # CPU intra-op threads per worker
```

The default of one thread matches `torchrun` and prevents CPU oversubscription
across GPU worker processes. Increase it only for distributed stages with
substantial CPU-side computation; it does not change the parent process thread
pool.

### Validation Rules

```python
# Device count must equal product of parallel degrees
world_size = dp * cfg * sp_ring * sp_ulysses * pp * tp

# SP and TP may be combined as independent mesh dimensions.
# The selected pipeline must implement and validate the requested combination.
```

## Usage Examples

### Single GPU Inference

```python
from telefuser.core.config import ParallelConfig

config = ParallelConfig(device_ids=[0])
```

### 2 GPU Ulysses Sequence Parallel

```python
config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)
pipe_config.dit_config.parallel_config = config
pipe_config.enable_denoising_parallel = True
```

### 4 GPU CFG + Ulysses

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

### 4 GPU USP

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    sp_ring_degree=2,
    sp_ulysses_degree=2,
)
```

### 4 GPU Pipeline Parallel

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    pp_degree=4,
)
```

### 8 GPU Hybrid Parallel

```python
# DP=2, CFG=2, Ulysses=2
config = ParallelConfig(
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    dp_degree=2,
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

## Wan Video Example

```python
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.core.config import AttentionConfig, AttnImplType

# Create Pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
pipe_config = Wan21VideoPipelineConfig()

# Configure parallelism
if gpu_num > 1:
    pipe_config.dit_config.parallel_config.device_ids = list(range(gpu_num))
    pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
    pipe_config.enable_denoising_parallel = True

# Configure attention
pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)

# Initialize
pipe.init(module_manager, pipe_config)

# Inference
video = pipe(
    prompt="A stylish girl playing with her dog",
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
)
```

## Device Mesh Utility Functions

```python
from telefuser.distributed import (
    # Process groups
    get_dp_group, get_dp_rank, get_dp_world_size,
    get_cfg_group, get_cfg_rank, get_cfg_world_size,
    get_ulysses_group, get_ulysses_rank, get_ulysses_world_size,
    get_ring_group, get_ring_rank, get_ring_world_size,
    get_pp_group, get_pp_rank, get_pp_world_size,
    get_tp_group, get_tp_rank, get_tp_world_size,
    
    # PP helpers
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    
    # Strategy detection
    get_attention_strategy,  # Returns "local", "ulysses", "ring", "usp"

    # Communication
    ulysses_scatter_heads,
    ulysses_gather_heads,
    RingP2PComm,
    PipelineP2PComm,
    merge_attn_states,
    ring_attention_forward,
)

# Get current attention strategy
strategy = get_attention_strategy(device_mesh)
# "local": No sequence parallelism
# "ulysses": Ulysses only
# "ring": Ring only
# "usp": Ulysses + Ring combination
```

## Performance Optimization Tips

### Choosing Parallel Strategy

| Scenario | Recommended Strategy | Notes |
|----------|---------------------|-------|
| Short video (81 frames) | Single GPU or CFG=2 | Low communication overhead |
| Medium video (161 frames) | Ulysses=2 | High All-to-All efficiency |
| Long video (241+ frames) | Ring or USP | Supports arbitrary length |
| Large model (14B) | PP or FSDP | Model splitting |

### FSDP vs TP Selection

FSDP and TP shard model weights differently, but their inference communication has different scaling. FSDP
all-gathers parameter units; TP reduces activation tensors after tensor-parallel operators. Do not infer the faster
strategy from model size or sequence length alone.

| Condition | Candidate strategy | Reason |
|-----------|--------------------|--------|
| The largest FSDP wrapping unit plus activations fits per GPU | **FSDP is feasible** | FSDP must materialize a complete wrapped unit during its all-gather. |
| A wrapped unit cannot fit per GPU | **TP or PP is required** | Splitting only the stored FSDP shards does not remove the full-unit all-gather peak. |
| Parameter all-gathers dominate exposed time | **TP** | TP keeps parameter shards resident and communicates activations instead. |
| Activation reductions dominate exposed time, especially at large `B * S * H` | **FSDP or TP + SP** | FSDP parameter traffic is mostly independent of sequence length; SP reduces the TP-local sequence but adds its own communication. |
| TP group is inside a fast NVLink/NVSwitch domain | **TP is a strong candidate** | Its many latency-sensitive activation collectives benefit from low-latency, high-bandwidth links. |

For a first-pass estimate, measure bytes per rank per denoising step. With FSDP degree `f`, full parameter bytes
`P_u` for each wrapped unit, and `r_u` all-gathers per request:

```text
V_fsdp ~= sum_u(r_u * P_u * (f - 1) / f)
```

With TP degree `t`, SP degree `s`, activation element size `e_a`, and one TP collective `j` over an activation of
shape approximately `[B, S / s, H]`:

```text
V_tp ~= sum_j(2 * (t - 1) / t * B * (S / s) * H * e_a)
```

The sum normally includes the row-parallel reductions in every transformer block. Add Ulysses or Ring traffic
separately; SP reduces the TP-local activation size but is not free. Convert bytes into a decision with measurements
on the target topology:

```text
T_comm ~= N_collectives * latency + V_wire / effective_bandwidth
T_step ~= T_compute + exposed(T_comm after overlap)
```

Profile p50/p95 step time, raw and exposed communication time, collective count, and per-rank peak HBM for the
actual model, sequence length, dtype, denoising-step count, and mesh degrees. FSDP can be combined with PP and SP;
TP and SP may be combined when the selected pipeline implements and validates that mesh.

### Communication Optimization

1. **Use async communication**: `async_usp_forward` overlaps computation and communication
2. **Batch communication**: `batch_isend_irecv` reduces communication count
3. **FP8 quantization**: Reduces communication data volume

### Memory Optimization

1. **Sequence parallelism**: Reduces sequence length per GPU
2. **Pipeline parallelism**: Reduces number of layers per GPU
3. **CPU Offload**: Offloads weights to CPU

## Troubleshooting

### Device Count Mismatch

```
RuntimeError: device num 4 and world size 2 not match
```

**Solution**: Ensure `len(device_ids) == dp * cfg * sp_ring * sp_ulysses * pp * tp`


### Ring Attention Requires LSE

```
RuntimeError: Ring attention requires log-sum-exp from attention implementation
```

**Solution**: Use attention implementation with LSE support (Flash Attention 2/3/4).

### Communication Timeout

```
RuntimeError: ParallelWorker timeout
```

**Solution**: Increase `timeout` parameter value, or check network connection.

## References

- [Ulysses: Sequence Parallelism](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [GPipe: Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
