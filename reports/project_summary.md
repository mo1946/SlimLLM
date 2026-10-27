# SlimLLM 项目总结与面试准备

整理日期：2026-10-27（Asia/Shanghai）。本轮仅做发布文档、忽略规则和复现检查，不新增核心功能、训练、下载、安装或性能重测，不提交或上传 GitHub。

## 一句话介绍

在 RTX 4060 Laptop 8GB 上，围绕 Qwen3-1.7B 完成合成工单治理、NF4 QLoRA、严格 JSON 抽取服务、有界请求调度和 WSL2 vLLM 推理对比的单机 MVP。

## 已实现与真实结果

| 环节 | 已完成的工程工作 | 结果及范围 |
| --- | --- | --- |
| 数据 | 七字段 Schema、原文实体校验、全量独立复标、近似模板分组划分、冻结哈希 | 499 条合成工单，399/50/50；无独立人工金标 |
| Baseline/训练 | 相同提示词与解码比较；NF4/double quant、r=8、答案专用损失、Gradient Checkpointing、只存最终 Adapter | 50 条测试字段宏准确率 63.71% → 85.14%，整条匹配 6% → 20% |
| 资源 | 复用现有模型与环境、不 merge、不保存中间 Checkpoint/优化器 | 训练/验证约 1018 秒；训练已分配峰值 3.10 GiB，加载峰值 4.43 GiB，Adapter 约 6.16 MiB |
| API/缓存 | lifespan、参数与输出校验、健康检查、Redis 指纹精确缓存及故障降级 | 真实 HTTP 与 Redis 命中/降级已有记录，默认回环地址单 worker |
| 请求调度 | 原忙时 503 改为 CPU 有界 FIFO、等待超时、取消/关闭保护、单模型线程 | 并发 8 成功率 12.5% → 100%；GPU 吞吐量没有因队列提升 |
| vLLM | WSL2 独立 Linux 环境、共用权重、单引擎最多 2 个序列、有限并发 HTTP 网关、保留 TF 回退 | 并发 8 RPS 0.367 → 0.747，P95 21.219 → 10.973 秒；四档 128/128 成功 |
| 约束解码 | vLLM 0.11.2 structured_outputs.json + guidance，保留完整 Schema，模式/哈希隔离缓存 | 独立 12 条回归枚举/Schema 11/12 → 12/12；冻结 50 条输出不变，语义问题仍存在 |
| 演示页 | Vue3 + Vite，输入/JSON/字段视图/复制/耗时/健康/错误，本地 API 代理 | 10 项前端测试、构建及两次真实 Transformers 代理调用通过；用户人工确认交互正常 |

服务测试每档相同 32 条合成工单，关闭 Redis 和 vLLM prefix cache，两次预热排除。vLLM 四档 JSON/Schema 100%、字段宏准确率 86.61%（194/224）、整条匹配 18.75%（6/32），输出全部与原 FIFO 参考一致。并发 1 P95 从 3.172 秒增至 3.815 秒；最高整卡显存采样为 7181 MiB。完整数字、统计单位及本机原始文件 SHA256 见 [project_metrics.json](project_metrics.json)。

## 可展示的技术亮点

1. **完整比较链路**：数据、划分、提示词、模型 revision、解码和 ID 顺序对齐，避免把提示词或样本变化当成微调收益。
2. **资源边界清楚**：训练只更新约 321 万 LoRA 参数；推理保持一份 GPU 权重，队列不持有 GPU 张量；环境/模型/缓存保存在 B 盘，未为推理合并完整模型。
3. **控制流经过验证**：参数 422、非法输出 502、队列满 503、等待超时 504；取消运行请求不释放仍在执行的模型槽位，Redis HIT 可绕过满队列。
4. **优化不只看成功率**：同时报告成功 RPS、端到端 P95、等待时间、输出正确性与显存代价。GPU 引擎日志实际观察到 Running=2，与客户端并发 8 区分。
5. **保留可用回退**：原 Transformers + LoRA/FIFO 未被移除；历史真实验证先停止 vLLM 并释放显存，再重新加载 TF 调用成功。

## 复现的真实边界

本机启动步骤见 [运行说明](../docs/RUNNING.md)，缺少制品时看 [模型与 Adapter 准备](../docs/ARTIFACTS.md)。默认推理需要 `.venv` 及其原共享依赖、现有模型、最终 Adapter 和 `reports/baseline/run.json`；新克隆可用明确标记的纯推理提示词模板初始化，但不能冒充实验清单。完整评测/压测还需原冻结 records/splits。Windows `.venv` 不能复制到 Linux；vLLM 环境在 B 盘 Ubuntu VHD 内。

发布内容只包含代码、手写样例、文档和指标摘要。原始合成数据、Adapter、模型、环境、运行清单和逐请求证据保持本地忽略，不提供未经确认的下载链接。因此仅凭 Git 仓库不能独立重算模型输出；没有干净机器的一键安装验证，不能宣称完全开箱即用。

拥有完整匹配制品后，可按以下入口重跑实验；这些命令会真实占用 GPU，本次整理没有执行。使用新的空报告目录，原记录保持不变；已有文件哈希不同或目录已存在时先检查原因，不删除或覆盖以强行通过。

```powershell
Set-Location 'B:\python\SlimLLM'
# 相同冻结测试集：先基座，再挂载既有 Adapter，无需重训
& '.\.venv\Scripts\python.exe' -B -m slimllm.training.baseline run --output reports/baseline_reproduce
& '.\.venv\Scripts\python.exe' -B -m slimllm.training.baseline run --adapter outputs/qlora/final_adapter --output reports/adapter_reproduce
& '.\.venv\Scripts\python.exe' -B -m slimllm.evaluation.evaluate --run reports/baseline_reproduce
& '.\.venv\Scripts\python.exe' -B -m slimllm.evaluation.evaluate --run reports/adapter_reproduce --compare-to reports/baseline_reproduce
# API 已就绪、缓存关闭后，重复当前实现的压测；不能冒充历史忙时拒绝版本
& '.\.venv\Scripts\python.exe' -B -m slimllm.evaluation.benchmark --requests 32 --concurrency 1 2 4 8 --output reports/api_reproduce
```

原始来源保留在本机 `reports/stage1.md` 至 `stage4.md`、`data_report.md`、`stage2_preparation.md`、`concurrency.md`、`vllm.md` 与相应 JSON；发布摘要仅提取原值及来源哈希，不是新实验。历史报告按阶段记载当时状态，README 和本总结反映当前完成状态。

## 局限与尚未验证

- 合成生成与复标共用 DeepSeek，存在标签共同偏差；少数类型 no_reason 仅 2 条、other 仅 3 条。近似模板去泄漏是启发式，不能证明不存在语义泄漏。
- 只有 50 条效果测试，reason 精确片段匹配受边界影响；微调后 reason 仅 42%、整条匹配仅 20%。Schema 合法不代表业务语义正确。
- 性能为单次、小样本、分组闭环负载与同日历史参考，非交错测量，未隔离温度/功耗/后台负载。vLLM 增加显存，并发 1 尾延迟未改善。
- 单 GPU、单 worker、无公网鉴权/TLS/租户隔离、无长期稳定性和开放到达流量容量验证；入口并发 8 不代表可持续高 QPS 或生产 SLA。
- 无流式接口，TTFT 未测；vLLM 远端处理耗时不是 CUDA 生成时间，未提供的进程显存字段不补造。
- 源码按所有者确认的 MIT 发布；未公开模型、Adapter 和完整实验数据，其许可不被源码 MIT 代替，详见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

## 约束解码后的准确率边界

冻结 50 条的四分类宏准确率均为 178/200=89%，旧七字段仍为 298/350=85.14%。独立 12 条回归 JSON 均 12/12，枚举/Schema 11/12 → 12/12，四分类 43/48 → 44/48；旧七字段 72/84 → 77/84 的差异含 Schema 失败整条计错政策，不代表五个原始字段改善。两组整条匹配各自不变（10/50、7/12）。优先级 high/normal/urgent 及 refund/return_refund 混淆仍存在。

真实 grammar 检查是 75/76，合法 Unicode 转义被拒绝的边界经用户接受，未删除用例。Transformers 回退不具备同等约束保证。上文 RPS/P95 是未开启约束解码的历史测量；本轮未测新模式吞吐或 TTFT。当前自动化测试为 97 项 Python + 10 项前端，fixture 不代表模型输出。

## 面试常见问题

**1. 为什么选 Qwen3-1.7B 和 QLoRA？**
目标是 8GB 单卡可运行的领域结构化抽取。训练用 NF4 降低冻结基座的存储/显存负担，LoRA 只训练少量低秩参数；这不是全参数训练，也不意味着线上推理仍为 4-bit，本项目服务实际用 FP16 基座挂 Adapter。

**2. 为什么 max_length 不是 768？**
原冻结提示词较长，449 条 train/validation 完整序列均超过 768，最长 927；采用 1024 并拒绝超长，不通过截掉答案省显存。mask 保证提示词/padding 不计损失，JSON 答案和 EOS 才参与。

**3. 怎么避免数据泄漏和指标漂移？**
近似模板分组后按 seed=42 划分，Baseline 前冻结数据/划分/提示词；比较程序校验模型 revision、SHA、ID 顺序和解码参数。测试不参与训练/调参，但启发式分组不提供绝对无泄漏保证。

**4. 为什么 JSON 100%，整条匹配才 20%？**
前者只检查格式，Schema 进一步检查类型/枚举，字段匹配检查标签相等，整条要求七字段全对。合理但边界不同的 reason 片段也会计错；原文引用校验只防止生成不存在的片段，不判断标签语义。

**5. 21.43% 是什么提升？**
是字段宏准确率 63.71% 到 85.14% 的 21.43 个百分点，分子 223/350 到 298/350。32 条性能子集的 86.61% 不能拿来与 50 条结果做新提升。

**6. 队列为什么不提高吞吐量？**
Transformers 路径一次只生成一个请求，CPU 等待区只把立即拒绝变为有界等待。成功率提高，等待延迟也增加；长期输入速率超过处理速率仍会排满或超时。

**7. 如何处理取消与线程安全？**
运行中的 CUDA 线程不会因 HTTP 取消立刻停止，不能提前让下一请求进入模型。TF 使用单模型执行线程；vLLM 分支只把最多两个 HTTP 请求交给单个引擎调度，并为每次请求返回独立计量信息。

**8. continuous batching 如何接入原队列？**
若网关仍只有一个消费者，会把远端引擎提交串行化。可选分支允许两个有限在途 HTTP 请求，GPU 引擎 max_num_seqs=2；日志实际观察 Running=2。其余入口请求等待，没有增加 GPU 模型副本。

**9. 为什么不直接说“吞吐提升两倍”？**
必须附并发 8、32 条固定工单、缓存关闭、单卡以及成功 RPS 的口径。是历史 FIFO 约 0.367 对 vLLM 约 0.747；并发 1 的 P95 变差，且系统/版本/温度不是严格控制变量。

**10. Redis 缓存如何避免返回旧模型结果？**
键包含原文 SHA256 与模型/Adapter/Schema/提示词/生成配置指纹，远端还区分后端版本；命中值重新校验，Redis 异常回到真实推理。只缓存精确文本，不宣称语义缓存或已测缓存吞吐提升。

**11. Windows 上遇到的工程问题是什么？**
历史 mmap 加载曾触发原生访问异常，改用 safetensors pread 逐张量读取，再以 meta/assign 严格装载，正确处理共享权重及 RoPE 缓冲区。vLLM 在独立 WSL2 环境使用 Linux 二进制，保留原 Windows 路径而未强装原生 vLLM。

**12. 为什么训练显存 3.10 GiB，服务却约 7 GiB？**
统计对象不同：前者是训练阶段 PyTorch 已分配峰值，不含加载/驱动；后者是 vLLM 预分配 KV 等资源后的整卡采样，还包含其他进程。不能用这两个数字计算量化节省比例。

**13. 要走向业务验证，下一步是什么？**
优先补独立人工金标、少数类别和真实脱敏工单评估，再做持续流量与失败边界验证。鉴权、隔离和部署治理也是缺项；这些只是局限分析，本轮没有继续实现。

## 本轮验收与发布边界

已执行现有 60 项 unittest，全部通过、无跳过；已有 Starlette/httpx 弃用提示未通过安装新依赖消除。`slimllm/training/train.py check` 无缺失依赖，API/Baseline/benchmark CLI 帮助可正常解析；未重启 GPU 服务，仅复用历史真实推理证据。原始实验报告与核心代码保持原样，展示文件另行生成。

`.gitignore` 用报告白名单阻止机器探测/安装日志、逐请求文本和本机过程记录意外提交。实际 Git 规则验证通过 46 个用例，43 个发布候选文件未发现高置信密钥/机器标识或已知本地密钥的匹配；13 个文档相对链接均指向可发布文件。完整本机结果保存在 `reports/presentation_validation.json`（该文件本身不发布）；这不是对未知历史或未来文件的安全保证。整理时根目录不是 Git 仓库，验证使用项目 `.cache/tmp` 内的隔离 Git 元数据；没有初始化根仓库、提交、上传或更改许可证。

[返回 README](../README.md) · [简历表述](../docs/resume.md)
