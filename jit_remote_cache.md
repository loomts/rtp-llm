# JIT Remote Cache 设计文档 - Direct Remote/FUSE 版本

分支：`chore/direct-remote-jit`

对比分支：`features/jit-remote-cache`

## 目标

这个分支用于验证 FUSE/remote 目录直接承载 JIT cache 的行为：启动时不再从远端 snapshot restore 到本地，也不再通过 watchdog 把本地增量同步回远端，而是直接把传入的 `REMOTE_JIT_DIR` 解析成一个可写目录，并把六个 JIT 组件的 env 直接指向这个目录下的组件子目录。

覆盖范围：

| 组件 | 产物类型 | env |
| --- | --- | --- |
| flashinfer | JIT workspace 中的动态库 | `FLASHINFER_WORKSPACE_BASE` |
| deep_gemm | JIT kernel 源文件和 cubin | `DG_JIT_CACHE_DIR` |
| tensorrt_llm_deep_gemm | FlashInfer 内部 TensorRT-LLM DeepGEMM cubin | `TRTLLM_DG_CACHE_DIR` |
| torch_extensions | PyTorch extension 编译产物 | `TORCH_EXTENSIONS_DIR` |
| triton | Triton 编译缓存 | `TRITON_CACHE_DIR` |
| triton_autotune | Triton autotune JSON 配置 | `TRITON_AUTOTUNE_CONFIG_DIR` |

明确不缓存：模型权重、KV cache、CUDA graph 运行态。

## 核心结论

当前 direct-remote 分支的设计很简单：

1. `REMOTE_JIT_DIR` 非空时，先解析 remote URI 或本地路径。
2. 解析后的目录存在且是目录时，直接作为 JIT cache 根目录。
3. 六个组件的 env 直接指向该根目录下的组件子目录。
4. `REMOTE_JIT_DIR` 缺失或不可用时，**直接禁用 direct remote JIT cache**：`remote_root=None`、`components=()`，`bootstrap()` 提前返回，不设置任何组件 env。此时各 JIT 库回退到自身默认路径（例如 `$HOME/.triton`、`$HOME/.cache` 等）。**本分支不回退到 `LOCAL_JIT_DIR`。**
5. `prepare()` 只做 direct remote 目录可见性检查和日志记录，不做 remote 到 local 的拷贝；`start_background_sync()`、`stop()` 都是 no-op。

这个版本刻意砍掉了 `features/jit-remote-cache` 中的 snapshot、delta、watchdog、compact、后台同步和 zstd/tar archive 逻辑，目的是做 FUSE 测试和行为对照。

> ⚠️ **失败模式差异（对照实验的关键变量之一）**：这是 direct 相对 snapshot 分支一个结构性更差的行为。snapshot 分支在 remote 不可用时会优雅降级到 `LOCAL_JIT_DIR`，服务仍然有本地 JIT cache；而 direct 分支会让本次启动**完全没有 JIT cache 隔离目录**，每个组件从零编译到各自默认路径。fuse 生产环境如果挂载在启动瞬间未就绪 / 抖动 / URI 解析失败，direct 会付出全量重编译的 TTFT 长尾。这一点必须在 fuse 测试里主动注入验证（见文末「FUSE 测试关注点」），而不能假设 remote 永远可用。

## 与 features/jit-remote-cache 的差异

| 项目 | `features/jit-remote-cache` | `chore/direct-remote-jit` |
| --- | --- | --- |
| 远端使用方式 | 远端只存 snapshot/delta archive | 远端/FUSE 目录直接作为运行时 cache 根目录 |
| 启动 restore | 从 snapshot + delta 解压到本地 | 无 restore |
| 运行期新增文件 | watchdog 捕获后 stage 到本地 delta | 组件直接写入 remote/FUSE 子目录 |
| 后台线程 | 定期上传 delta，并尝试 compact snapshot | 无后台线程 |
| 远端目录 | `.jit_snapshot.tar.zst`、`.delta/`、compact lock | 普通 JIT 散文件目录 |
| 依赖 | `watchdog`、`zstandard`、tar archive 逻辑 | 无这些依赖 |
| remote 不可用时 | 降级到 `LOCAL_JIT_DIR`，仍有本地 cache | 禁用 JIT cache，组件回退各自默认路径 |
| 热路径 remote I/O | 无（restore 一次，运行期只写本地） | 有（每次 JIT 读写直接命中 remote/FUSE） |
| 测试目的 | 避免远端 I/O 进入推理/JIT 热路径 | 真实观察 FUSE 直接读写的性能和稳定性 |

direct-remote 版本会把 remote I/O 放回 JIT/推理相关路径，所以它不是原 PR 的优化方案，而是一个对照实验分支。

## 代码入口

| 模块 | 位置 |
| --- | --- |
| 核心实现 | `rtp_llm/utils/jit_cache_manager.py` |
| 配置对象 | `rtp_llm/config/py_config_modules.py` 的 `JITConfig` |
| CLI 参数 | `rtp_llm/server/server_args/jit_group_args.py` |
| 后端启动入口 | `rtp_llm/start_backend_server.py` |
| 单测 | `rtp_llm/utils/test/jit_cache_manager_test.py` |
| GPU 集成测试 | `rtp_llm/utils/test/jit_cache_remote_integration_test.py` |

`jit_cache_manager.py` 只负责三件事：

1. 解析 `remote_jit_dir`（`resolve_remote_root`）。
2. remote 可用时，把六个组件目录创建在 remote 根目录下并注入六个 env；remote 不可用时，禁用整个功能。
3. 提供 `apply_jit_cache_env()` 供非 rank0 或外部调用直接注入 env。

远端 URI 的挂载复用 `rtp_llm.utils.fuser.fetch_remote_file_to_local(uri, MountRwMode.RWMODE_RW)`。

## 启动时序

`local_rank == 0`：

1. 创建 `JitCacheManager(py_env_configs.jit_config)`。
2. `bootstrap()`：remote 可用时创建 remote 根目录和六个组件目录，并设置组件 env；remote 不可用时直接返回，不设 env。
3. `prepare()`：remote 可用时检查六个组件目录是否已有条目，记录 direct remote cache 可见性；remote 不可用时直接返回。
4. `start_background_sync()`：no-op。
5. `finally` 中设置 `jit_ready_event`。

`local_rank != 0`：

1. 等待 rank0 的 `jit_ready_event`，最长 600s。
2. 创建 `JitCacheManager(py_env_configs.jit_config)` 并调用 `bootstrap()`，让非 rank0 也把组件 env 指向同一个 remote/FUSE 根目录（remote 不可用时同样禁用，不设 env）。

> 注意：与 snapshot 分支不同，direct 分支的**每个 rank 都会各自调用 `bootstrap()`**，即每个 rank 都会各自解析 URI / 挂载 FUSE / `mkdir` 组件目录。TP 内多个进程会并发在同一个 remote 根目录下 `mkdir` 同名组件目录。这是 direct 分支需要在 FUSE 上重点验证的并发点之一。

JIT cache 初始化失败会记录异常并继续启动服务；该功能只影响 cache 命中率，不影响推理正确性。

## 架构

```text
                              backend startup
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
        +---------------------+               +----------------------+
        | local_rank 0        |               | non-local-rank0      |
        | JitCacheManager     |               | waits jit_ready_event|
        +----------+----------+               +----------+-----------+
                   |                                     |
                   | bootstrap()                         | bootstrap()
                   v                                     v
        +-------------------------------------------------------------+
        | remote_root = resolve_remote_root(remote_jit_dir)           |
        |                                                             |
        | remote 可用 (存在且是目录): 直接作为 JIT root               |
        | remote 不可用 (空/解析失败): remote_root=None, 禁用, 不设 env|
        +-----------------------------+-------------------------------+
                                      | remote 可用
                                      v
        +-------------------------------------------------------------+
        | component envs (根目录 = remote_root)                       |
        | FLASHINFER_WORKSPACE_BASE    -> flashinfer/...              |
        | DG_JIT_CACHE_DIR             -> deep_gemm/...               |
        | TRTLLM_DG_CACHE_DIR          -> tensorrt_llm_deep_gemm/...  |
        | TORCH_EXTENSIONS_DIR         -> torch_extensions/...        |
        | TRITON_CACHE_DIR            -> triton/                      |
        | TRITON_AUTOTUNE_CONFIG_DIR   -> triton_autotune/...         |
        +-------------------------------------------------------------+
```

没有 remote snapshot store、没有 delta staging、没有 watchdog observer、没有 sync thread、没有 compact lock。

## 配置

| CLI 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--local_jit_dir` | `LOCAL_JIT_DIR` | `./.jit_cache` | 仅由 `JITConfig` 保留；**direct 分支不使用它做 fallback**。保留只为与 snapshot 分支共用同一份配置对象和 CLI |
| `--remote_jit_dir` | `REMOTE_JIT_DIR` | `""` | 远端目录或远端 URI；可用时直接作为 JIT cache 根目录 |

`remote_jit_dir` 解析逻辑（`resolve_remote_root`）：

1. 空字符串：返回 `None`，禁用 direct remote JIT cache（不设任何组件 env）。
2. 带 URI scheme：通过 `fetch_remote_file_to_local(uri, MountRwMode.RWMODE_RW)` 挂载或映射成可写本地路径。
3. 其他字符串：`expanduser().absolute()` 后作为本地路径使用。
4. 解析后的路径必须已经存在且是目录，否则返回 `None`（禁用）并打 warning log。

> `local_jit_dir` 在 direct 分支里是**惰性字段**：`JitCacheManager` 不读它，也不会在它下面创建任何目录。它只是为了让 snapshot / direct 两个分支共用同一个 `JITConfig`、同一套 CLI 参数、同一套测试 harness。不要误以为它是 remote 失败后的兜底。

## 硬编码常量

| 常量 | 当前值 | 说明 |
| --- | --- | --- |
| `BUILTIN_CONFIG_SENTINEL` | `__builtin__` | 保留内置 autotune 配置的 sentinel |
| `DETAILED_STATS_ENV` | `JIT_CACHE_DETAILED_STATS` | 是否输出组件详细文件统计 |

旧实现（snapshot 分支）中的这些常量在本分支已移除：

| 已移除常量/概念 | 原用途 |
| --- | --- |
| `SNAPSHOT_NAME` | 远端 full snapshot 文件名 |
| `REMOTE_DELTA_DIR_NAME` | 远端 delta archive 目录 |
| `REMOTE_SNAPSHOT_COMPACT_LOCK_DIR_NAME` | 远端 compact 互斥锁目录 |
| `LOCAL_COMPACT_WORK_DIR_NAME` | 本地 compact 临时工作目录 |
| `SNAPSHOT_LOCK_STALE_S` | compact lock 过期强制接管阈值 |
| `SYNC_POLL_S` | 后台同步轮询间隔 |
| `STOP_JOIN_TIMEOUT_S` | stop 时 observer/thread join 上限 |

## Remote Root 目录布局

`REMOTE_JIT_DIR` 可用时，下列目录直接创建在 remote/FUSE 根目录下：

```text
{remote_jit_root}/
  flashinfer/cuda-{torch.version.cuda or unknown}/
  deep_gemm/deep_gemm-{deep_gemm distribution version or unknown}/
  tensorrt_llm_deep_gemm/cuda-{torch.version.cuda or unknown}/
  torch_extensions/torch-{torch version}-{pyXY[_abi]_{cuXXX|cpu}}-cxxabi-{0|1|unknown}/
  triton/
  triton_autotune/{gpu_name}/
```

所有 scope 里的动态字符串都会经过 `safe_part()` 归一化：非 `[0-9A-Za-z]` 字符会被替换成 `_`，首尾 `_` 会被去掉。

`triton_autotune/{gpu_name}` 使用 `get_gpu_info()`，优先级是：

1. `TRITON_AUTOTUNE_GPU_NAME`
2. `torch.cuda.get_device_name(0)`
3. 无 GPU 或探测失败时使用 `unknown`

## 组件规则

| 组件 | env | scope |
| --- | --- | --- |
| `flashinfer` | `FLASHINFER_WORKSPACE_BASE` | `cuda-{torch.version.cuda}` |
| `deep_gemm` | `DG_JIT_CACHE_DIR` | `deep_gemm-{dist version}` |
| `tensorrt_llm_deep_gemm` | `TRTLLM_DG_CACHE_DIR` | `cuda-{torch.version.cuda}` |
| `torch_extensions` | `TORCH_EXTENSIONS_DIR` | `torch-{torch version}-{python/cuda}-cxxabi-{abi}` |
| `triton` | `TRITON_CACHE_DIR` | 无额外 scope |
| `triton_autotune` | `TRITON_AUTOTUNE_CONFIG_DIR` | `{gpu_name}` |

本分支没有 watchdog event 和同步后缀过滤。所有文件是否生成、如何复用、何时改写，都由各组件自身的 JIT/cache 实现决定。

`apply_jit_cache_env()` 会覆盖已有组件 env。唯一例外是 env 当前值等于 `__builtin__` 时保留不覆盖；这主要用于 `TRITON_AUTOTUNE_CONFIG_DIR=__builtin__`，让 Kimi KDA 等 smoke 固定使用代码仓内置 autotune 配置。

env 必须在 import `deep_gemm`、`flashinfer`、`triton` 或相关 autotune 模块之前设置，否则底层库可能已经读取了旧路径。

`TRTLLM_DG_CACHE_DIR` 来自 FlashInfer 包内部的 TensorRT-LLM DeepGEMM 代码。RTP-LLM 的 `CudaFp8FlashinferLinear` 在 small-M FP8 路径会调用 `flashinfer.gemm.fp8_blockscale_gemm_sm90`，该路径读取 `TRTLLM_DG_CACHE_DIR` 保存 `cache/.../nvcc_kernel.cubin`。如果不设置该 env，FlashInfer 会 fallback 到 `$HOME/.tensorrt_llm`，这会绕过 remote JIT cache 方案，并可能让生产实例共享或污染用户 home 下的 cache。

## 远端目录布局

direct-remote 分支不维护 archive 布局，远端目录就是普通 JIT 散文件树：

```text
{remote_jit_dir}/
  flashinfer/...
  deep_gemm/...
  tensorrt_llm_deep_gemm/...
  torch_extensions/...
  triton/...
  triton_autotune/...
```

旧实现（snapshot 分支）中的这些路径在本分支不会创建：

```text
.jit_snapshot.tar.zst
.jit_remote_snapshot_compact.lock.dir/
.delta/
.jit_delta/
.jit_compacting/
```

## 流程说明

### bootstrap()

1. remote 不可用 (`remote_root is None`)：打日志 `JIT cache bootstrap skipped: direct remote mode requires a valid remote_jit_dir`，直接返回，**不设任何 env、不创建任何目录**。
2. remote 可用：
   - `remote_root.mkdir(parents=True, exist_ok=True)`。
   - `apply_jit_cache_env(remote_root, create_dirs=True)`：设置六个组件 env，并创建各组件 scoped 目录。

### prepare()

direct-remote 分支里的 `prepare()` 不做 restore，也不会拷贝任何 remote/FUSE 文件。

1. remote 不可用：直接返回。
2. remote 可用：只打一条日志说明 direct remote 模式直接使用组件 cache 路径。

也就是说，第二次启动复用 cache 的方式不是 `prepare()` 拉取，而是 `bootstrap()` 后六个组件 env 已经直接指向同一个 remote/FUSE 目录，底层 JIT 组件会按自己的 cache 逻辑读取已有文件。

不读取 snapshot，不解压 delta，不访问远端 archive。

### start_background_sync()

no-op。不启动 watchdog，不创建 staging 目录，不上传 delta，不 compact snapshot。

### stop()

remote 可用时只输出一次组件可见性统计（受 `JIT_CACHE_DETAILED_STATS` 控制），不 flush 任何文件；remote 不可用时直接返回。

direct-remote 模式下文件由各 JIT 组件直接写入 env 指向的目录，进程退出时无需做任何同步。

## 写入和并发语义

本分支不额外提供跨进程 archive 原子性或 compact 互斥。并发语义完全取决于：

1. FUSE/remote 目录本身的文件系统语义。
2. 各 JIT 组件自身的 cache 写入协议。
3. 多 rank/多实例是否同时写入同一 cache key。

这也是该分支的测试重点：直接观察 remote/FUSE 散文件作为运行时 JIT cache 时，是否存在明显的性能抖动、锁竞争、部分写入、可见性延迟或同名文件冲突。特别注意 TP 内多 rank 并发 `bootstrap()` 时对同一 remote 根目录的并发 `mkdir`，以及多实例（多 pod）同时冷启同一模型时对同名 JIT 产物的并发写。

## 指标日志

默认日志会记录：

- remote URI/FUSE mount 耗时。
- JIT setup 总耗时（`start_backend_server.py` 记录）。
- bootstrap 耗时和六个组件 env。
- prepare 阶段的 skip 日志。
- stop 阶段六个组件目录最终是否仍可见。

如果需要递归统计文件数量和大小，设置：

```bash
export JIT_CACHE_DETAILED_STATS=1
```

开启后 `stop()` 会额外记录每个组件的：

- `file_count`
- `dir_count`
- `total_bytes`
- `max_file_bytes`
- `suffix_counts`
- `scan_ms`

> ⚠️ **对照公平性提醒**：详细统计会递归扫描 remote/FUSE 目录，可能放大 metadata 请求并影响 stop 时延；对 direct（散文件直挂 FUSE）的放大远大于 snapshot（本地目录）。压测做 direct vs snapshot 对比时，**务必分别记录开启和关闭该开关的两组结果**：关闭时测真实路径开销（作为对比结论依据），开启时才拿组件文件规模和小文件分布。不要用开启 `DETAILED_STATS` 那组数据去对比两方案的启动/退出开销，否则会系统性地高估 direct。

## 依赖升级和失效策略

当前实现只在部分组件路径中加入版本或硬件 scope，并不维护一个全局环境 fingerprint。升级依赖、CUDA、GPU 或 JIT 相关源码后，最稳妥的方式是使用新的 `REMOTE_JIT_DIR`。

| 组件 | 当前隔离维度 | 剩余风险 |
| --- | --- | --- |
| flashinfer | `cuda-{torch.version.cuda}`；flashinfer 自身 workspace 内部还会再按版本/arch 分目录 | 若 flashinfer 内部 cache key 未覆盖某类变更，可能命中过旧产物 |
| deep_gemm | `deep_gemm-{distribution version}` | 同版本但实现/header 变更时可能复用旧产物 |
| tensorrt_llm_deep_gemm | `cuda-{torch.version.cuda}` | 依赖 FlashInfer 内部 TensorRT-LLM DeepGEMM cache key；FlashInfer 或 TensorRT-LLM 内部实现变化时建议换新的 `REMOTE_JIT_DIR` |
| torch_extensions | torch 版本、Python ABI、CUDA/CPU、C++ ABI | 同名 extension 源码变更但缓存判断未触发时可能复用旧产物 |
| triton | JitCacheManager 不额外加 scope，依赖 Triton 自身 cache key | Triton 版本或 cache key 兼容性变化时建议隔离 |
| triton_autotune | GPU 名称 | 只缓存调参 JSON，kernel 仍由 Triton 编译；同名 kernel 的调参语义变化时需清理 |

清理建议：

- 清 direct remote/FUSE cache：删除 `${REMOTE_JIT_DIR}` 下对应组件目录，或直接换一个新的 `REMOTE_JIT_DIR`。
- 需要灰度不同依赖版本时，优先按版本/镜像/模型使用不同 `REMOTE_JIT_DIR`，避免新旧实例互相污染。

## FUSE 测试关注点

和 `features/jit-remote-cache` 对比时，重点看：

| 维度 | 关注内容 |
| --- | --- |
| 冷启动 | direct remote 是否能复用已有 JIT 文件，TTFT 是否下降 |
| 首请求 | JIT 编译期间 remote/FUSE 写入是否造成明显长尾 |
| 多 rank | TP 内多个 rank 各自 `bootstrap()`、并发 `mkdir` 同一 remote root 时是否稳定 |
| 多实例 | ≥2 实例（多 pod）并发冷启同一模型、写同一 `REMOTE_JIT_DIR` 时是否有同名文件冲突或部分写 |
| remote 不可用 | **主动注入** FUSE 未就绪 / URI 解析失败，确认 direct 会禁用 JIT cache 并付出全量重编译长尾（对比 snapshot 降级到本地） |
| 小文件 | Triton/deep_gemm/torch extension 小文件 metadata 操作是否成为瓶颈 |
| 可见性 | 一个进程写入后的文件是否能被另一个进程及时看到 |
| 清理 | 删除或更换 `REMOTE_JIT_DIR` 后是否能稳定重新生成 |

这个分支适合构造不同长度的推理请求，对同一个模型生成并复用全部六类 JIT cache，然后和 snapshot/delta 分支比较冷启动、TTFT、远端读写量和失败模式。

## 当前测试覆盖

单测覆盖：

- env 覆盖和 `__builtin__` 保留。
- bootstrap 目录创建（remote 可用时六个组件目录落在 remote root 下）。
- remote URI 挂载前置解析。
- `REMOTE_JIT_DIR` 可用时直接作为六个组件 env 根目录。
- `REMOTE_JIT_DIR` 缺失或不可用时**禁用 direct JIT cache**：`remote_root is None`、`components == ()`、不设任何组件 env、不创建 `local` 目录（`test_missing_remote_disables_direct_jit_cache`）。
- direct manager 不包含 `RemoteSnapshotStore`、`zstd_tar`、watchdog handler、delta 目录或 observer 状态。

GPU 集成测试覆盖：

- 单 GPU 生成 flashinfer、triton、triton_autotune、deep_gemm、tensorrt_llm_deep_gemm、torch extension JIT 产物，并确认这些产物直接落在同一个 remote root 下。
- 第二个 manager 复用同一个 remote root，确认 direct remote 目录里已有的组件产物可见。
- 两个 rank 并发指向同一个 remote root 运行 Triton JIT，并通过 marker 文件确认 remote 散文件树可被共同访问。
