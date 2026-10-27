# 第三方组件与数据来源

核对日期：2026-10-27。本项目原创源码/文档按 [MIT](LICENSE) 发布（经项目所有者确认）。该授权不替代下列组件、模型、数据或服务自身的条款。本仓库不打包模型、Adapter、Python/Node 依赖、Redis 二进制或 Linux 虚拟磁盘。

| 组件 | 核对范围与许可 | 来源 |
| --- | --- | --- |
| Qwen3-1.7B | 官方模型页及本地下载的 LICENSE 为 Apache-2.0；固定模型 revision 见 baseline.py | [官方模型及许可](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE) |
| vLLM 0.11.2 | Apache-2.0 | [固定版本 LICENSE](https://github.com/vllm-project/vllm/blob/v0.11.2/LICENSE) |
| Transformers / PEFT / TRL / Accelerate / Datasets / Hugging Face Hub / safetensors | 本机安装元数据为 Apache 系列许可 | [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)、[PEFT](https://github.com/huggingface/peft/blob/main/LICENSE)、[TRL](https://github.com/huggingface/trl/blob/main/LICENSE) |
| PyTorch | 本机 2.14.0 元数据含 BSD、Apache、MIT、BSL 及 LLVM exception 等组合条款；不简写成单一许可 | [上游许可证](https://github.com/pytorch/pytorch/blob/main/LICENSE)，以安装包内各子组件声明为准 |
| bitsandbytes / FastAPI / Pydantic / redis-py | 本机安装元数据为 MIT | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE)、[FastAPI](https://github.com/fastapi/fastapi/blob/master/LICENSE)、[redis-py](https://github.com/redis/redis-py/blob/master/LICENSE) |
| Uvicorn / HTTPX | 本机安装元数据为 BSD-3-Clause | [Uvicorn](https://github.com/encode/uvicorn/blob/main/LICENSE.md)、[HTTPX](https://github.com/encode/httpx/blob/master/LICENSE.md) |
| Vue 3.5.44 / Vite 7.3.7 / Vue 插件 6.0.10 | 本机 package.json 为 MIT，精确依赖/来源在 frontend/package-lock.json | [Vue](https://github.com/vuejs/core/blob/main/LICENSE)、[Vite](https://github.com/vitejs/vite/blob/main/LICENSE) |
| Redis 服务端 | 历史本地测试使用 5.0.14.1 Windows 端口；Redis 上游 5.0.14 为 BSD-3-Clause，不套用其他版本许可。未分发 Windows 端口，端口包内附加声明未作完整审计 | [Redis 5.0.14 COPYING](https://github.com/redis/redis/blob/5.0.14/COPYING) |

上表是关键直接依赖的来源核对，不是所有传递依赖/系统组件的完整许可审计。以后若分发打包环境、前端构建包、模型或 Adapter，需保留相应许可/版权声明，并另行核对被实际分发的制品；本次仅准备源码仓库。

## 合成数据与模型输出

训练/冻结测试集的 499 条电商售后工单由 DeepSeek API 合成和复标，不是真实客户数据，没有独立人工业务金标；完整数据不随仓库发布。仓库仅含手写功能样例和独立测试标签，模型响应示例明确标为模型生成，指标是本机有限样本实测。

[DeepSeek 服务条款](https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html)（页面更新日期 2026-03-27）第 4 节在适用法律和条款范围内讨论输入/输出权利及包括训练模型在内的用途，第 3 节要求公开输出时标明合成属性并核实内容。不能把 API 输出当成 DeepSeek 模型权重的开源许可，也不据此保证输出唯一或不涉及第三方权利。未来公开完整数据/Adapter 时须单独审核，而不是直接套用本仓库 MIT。

本仓库不使用第三方商标作背书，不提供用户数据或付费服务凭据。
