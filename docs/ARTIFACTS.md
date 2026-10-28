# 模型、Adapter 与复现制品

本机现有制品已准备齐全，**不要为启动重复下载、训练或安装**。公开仓库仅含代码/手写样例/指标，不包含模型、Adapter、完整合成数据、环境或原始实验记录，也没有提供伪造的制品下载链接。

## 推理必需文件

| 制品 | 固定路径 | 获取与限制 |
| --- | --- | --- |
| Qwen3-1.7B | `.cache/huggingface/qwen3-1.7b/` | 从 [官方模型](https://huggingface.co/Qwen/Qwen3-1.7B) 获取，或直接复用已有目录 |
| LoRA Adapter | `outputs/qlora/final_adapter/` | 项目所有者通过另行授权渠道提供，或自行训练；当前没有公开发布地址 |
| 推理提示词清单 | `reports/baseline/run.json` | 原始清单保持本地；新克隆仅推理时可用下述非实验模板初始化 |
| Windows 环境 | `.venv/` | 依赖清单及平台区别见 [运行说明](RUNNING.md)；不能复制本机共享环境到其他电脑 |

基座 revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，定义于 slimllm/training/baseline.py。原两片 safetensors 总计 4,063,515,592 字节，另有 tokenizer/config/LICENSE 等；保留一份，不 merge。官方模型 Apache-2.0 不等同本项目源码 MIT。

新环境需要下载时，先核对磁盘/许可并明确批准，再运行 `python -B -m slimllm.training.baseline download`。该入口使用固定 revision，写入项目缓存；此文档整理没有执行下载。不得以此命令重下本机已有权重。

Adapter 至少包含 `adapter_config.json`（peft_type=LORA）和 `adapter_model.safetensors`；训练使用 r=8、q/k/v/o 投影，约 6.16 MiB。原权重 SHA256：`2feb45b7574a51a9998ab721f3906c72950173537e6a34e67e21d0539c43f9d5`。自行训练得到的 Adapter 不能声称重现原表格的准确率；不随意修改配置以绕过模型不匹配。

## 新克隆的纯推理提示词初始化

[inference_run.example.json](artifacts/inference_run.example.json) 仅含当前冻结提示词及其哈希、模型标识和“非实验模板”标记，不包含私有机器路径、测试数据或预测结果。提示词与现有 slimllm.runtime.extraction_prompt()、本机 baseline 清单逐字核对；没有增加 Prompt 补丁。

仅当目标不存在时，在项目根目录执行以下命令；本机已有原清单，**跳过此步骤**：

```powershell
$target = Join-Path $PWD 'reports\baseline\run.json'
if (Test-Path -LiteralPath $target) { throw 'Existing manifest retained; do not overwrite it.' }
New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
$bytes = [IO.File]::ReadAllBytes((Join-Path $PWD 'docs\artifacts\inference_run.example.json'))
$stream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
```

这是为现有推理代码提供 prompt 的最小模板，不是 baseline 评测结果，不能冒充原 run.json 来评测、训练或计算原指标。没有 Adapter 时现有服务会明确失败；本次未新增“无 Adapter 偷跑”模式。

## 完整实验复现的额外前提

完整数据需要 `data/generated/stage1/records.jsonl`、`splits.json` 及原 baseline 清单，提示词/模型/Adapter/解码设置和 ID 均须匹配来源哈希。公开聚合指标不能恢复原预测，当前没有干净机器端到端复现验证。

可读流程为：slimllm/dataset.py generate → review → split；slimllm/training/baseline.py run → slimllm/training/train.py prepare/train → 挂载 Adapter 评测 → 启动 API → slimllm/evaluation/benchmark.py。数据生成需要付费 DeepSeek API，新模型/依赖下载和 GPU 训练另行批准。完全独立的新实验使用新的工作目录/报告目录和自己的合规数据；不要把纯推理模板与完整训练清单混用，也不要改旧冻结划分来获得更好分数。

未公开的模型/数据/Adapter 不属于 MIT 源码发布范围；许可与合成数据属性见 [第三方声明](../THIRD_PARTY_NOTICES.md)。
