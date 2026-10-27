# 运行与复现说明

以下启动路径已在本机验证；新机器的完整安装/模型制品重建未验证。所有命令从项目根目录执行，除非明确进入 frontend。运行前先阅读 [制品说明](ARTIFACTS.md)，不把缺少权重当作安装新框架的理由。

## 环境与依赖

| 环境 | 本机验证版本 | 依赖入口 |
| --- | --- | --- |
| Windows | Python 3.13.15 / Torch 2.14.0+cu126 / Transformers 5.16.1 | requirements-windows.txt |
| WSL2 Ubuntu 24.04.5 | Python 3.12.3 / vLLM 0.11.2 / Torch 2.9.0+cu128 / Transformers 4.57.1 / llguidance 1.3.0 | requirements-vllm.txt；setup_vllm.sh |
| 前端 | Node 22.14.0 / Vue 3.5.44 / Vite 7.3.7 | frontend/package-lock.json |

Windows `.venv` 原来通过 `.pth` 只读共享其他本地环境的大型依赖，不能搬到别的电脑或 WSL。`requirements-windows.txt` 显式列出大包并引用两个局部 lock；`constraints-stage2.txt` 只约束版本，不单独安装包。这些是已验证环境清单，不是所有平台都可解析的万能锁文件。

**本机已有环境：跳过安装。** 新机器先选择兼容 Python/驱动、创建项目内 `.venv`，配置缓存后按实际 wheel 可用性审查清单；下列 pip 命令会下载依赖，需先确认空间和下载权限，本次没有执行：

```powershell
Set-Location 'B:\python\SlimLLM'
. .\setup_env.ps1
& '.\.venv\Scripts\python.exe' -B -m pip install --only-binary=:all: -r requirements-windows.txt
```

setup_env.ps1 仅设置进程内 B 盘缓存；若执行策略不允许 dot-source，不要为文档示例修改系统策略。Python CLI 的 runtime.configure() 同样配置项目缓存。若 `.venv` 没有 pip，先选择已有包管理器及安装方案，不复制别的整个环境。Windows CUDA 与 Linux wheel 不共享。

## 默认 Transformers

```powershell
Set-Location 'B:\python\SlimLLM'
$env:SLIMLLM_BACKEND = 'transformers'
$env:SLIMLLM_REDIS_URL = ''
$env:SLIMLLM_QUEUE_MAX_WAITING = '8'
$env:SLIMLLM_QUEUE_WAIT_SECONDS = '30'
& '.\.venv\Scripts\python.exe' -B api.py --port 8000
```

等待启动完成，访问 `/health` 和 `/docs`。模型缺失时失败，不隐式下载或退到 CPU。保持单 worker，无 reload。

## 前端

先启动一个模型后端，再开第二终端：

```powershell
Set-Location 'B:\python\SlimLLM\frontend'
$env:TEMP = 'B:\python\SlimLLM\.cache\tmp'
$env:TMP = $env:TEMP
& 'B:\node\npm.cmd' run dev
```

访问 http://127.0.0.1:5173，选择示例/输入文本、抽取、切换字段视图或复制。Ctrl+Enter 可提交。输入改变后会标明当前结果属于上一次请求。手写示例不是真实客户数据；结果可能有语义错误。

`B:\node\npm.cmd` 是本机现有 npm，其他机器换成自己的 npm.cmd。只有首次复现、node_modules 不存在且已批准下载时，才执行 `npm.cmd ci --ignore-scripts --no-audit --no-fund`；ci 会重建依赖目录，不是日常启动步骤。缓存由项目 `.npmrc` 指向 `.cache/npm`，Vite 缓存在 `.cache/vite`。

```powershell
Set-Location 'B:\python\SlimLLM\frontend'
& 'B:\node\npm.cmd' test
& 'B:\node\npm.cmd' run build
& 'B:\node\npm.cmd' run preview
```

preview 使用 http://127.0.0.1:4173，同样代理 API。开发页只请求同源 `/api/extract`、`/api/health`，[Vite proxy](https://vite.dev/config/server-options.html#server-proxy) 去掉 `/api` 后转到 127.0.0.1:8000，无需修改 FastAPI CORS。dist 本身没有代理，不直接双击 index.html；不是生产部署。build 不自动清空旧 dist，可能保留旧哈希文件，未获批准不清理。

## 可选 WSL2 vLLM

先停止 Transformers 并确认显存释放。已有本机 WSL 环境无需重装；安装 WSL/Ubuntu/依赖/改系统功能是独立审批操作，不是每次启动步骤。模型直接从 B 盘读取，不复制或 merge。

```powershell
wsl.exe -d SlimLLM-Ubuntu -u root --exec bash /mnt/b/python/SlimLLM/run_vllm.sh
```

8001/health 就绪后，另一个终端启动同一个 API：

```powershell
Set-Location 'B:\python\SlimLLM'
$env:SLIMLLM_BACKEND = 'vllm'
$env:SLIMLLM_VLLM_URL = 'http://127.0.0.1:8001'
$env:SLIMLLM_VLLM_MAX_INFLIGHT = '2'
$env:SLIMLLM_VLLM_STRUCTURED_OUTPUTS = 'json_schema'
$env:SLIMLLM_REDIS_URL = ''
& '.\.venv\Scripts\python.exe' -B api.py --port 8000
```

引擎固定 GPU 0.70、上下文 2560、2 序列、2048 batched tokens、eager/chunked prefill、prefix cache 关闭、CPU offload/引擎 swap=0。启动需至少 6144 MiB 空闲显存。WSL 的 B 盘 2 GiB swap 是另一项系统配置，影响所有 WSL2 实例，不自动改动。

vLLM 仅验收 0.11.2；guidance/disable_fallback 由 run_vllm.sh 指定。模式 json_schema/off 在初始化后固定，切换需重启网关。健康模式/Schema 哈希仅是配置声明，不是语义保证；off 仍为 vLLM。上游 400/422 返回 502 backend_request_rejected，不悄悄去掉约束。Transformers 健康模式 none/null，仅普通生成后校验；切回时先停网关/引擎、确认显存释放，再用默认命令，不自动加载第二个模型。

## 可选 Redis

不启用 Redis 也能启动。项目配置为回环地址、6380、16 MiB LRU、禁用持久化。本机使用既有 5.0.14.1 Windows 端口，仅用于本地验证，不分发或推荐它作生产服务。

```powershell
Set-Location 'B:\python\SlimLLM'
New-Item -ItemType Directory -Force '.\.cache\redis' | Out-Null
$redisExe = 'B:\redis\Redis-x64-5.0.14.1\redis-server.exe'
& $redisExe '.\redis.windows.local.conf'
```

在 API 所在终端、启动前设置 `SLIMLLM_REDIS_URL=redis://127.0.0.1:6380/0`。此变量不是自动从 .env 加载的；用 `$env:SLIMLLM_REDIS_URL` 设置。服务故障会降级真实推理，缓存指纹隔离模型/提示词/Schema/模式。不连接或停止其他 Redis 实例。

## 验证与排错

仅克隆源码且已有必要 Python 依赖时，先运行不依赖私有制品的 42 项测试：

```powershell
Set-Location 'B:\python\SlimLLM'
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_schema.py -v
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_stage1.py -v
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_api.py -v
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -p test_request_queue.py -v
```

完整 97 项还依赖原冻结数据、运行清单及 Adapter，本机这些文件已存在；裸源码全量会失败，详见 [测试范围](../tests/README.md)。具备完整制品时再执行：

```powershell
Set-Location 'B:\python\SlimLLM'
& '.\.venv\Scripts\python.exe' -B -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -B train.py check
```

这些测试为协议/控制流验证，不能作为模型准确率。完整模型对照/压测需要未公开的冻结制品，入口为 baseline.py / evaluate.py / evaluate_structured.py compare / benchmark.py；使用新的报告目录，不能覆盖原证据。本次整理未重跑 GPU 测量。

Windows PowerShell 5 可能因响应缺少 charset 显示中文乱码，可使用显式 UTF-8 解码：

```powershell
$body = @{ text = '订单001234的蓝牙耳机左耳没有声音，我想退货退款。' } | ConvertTo-Json -Compress
$response = Invoke-WebRequest 'http://127.0.0.1:8000/extract' -UseBasicParsing -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
[Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray()) | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

502 invalid_model_output 是模型输出被拒绝，不能把 Swagger 的 Example Value 当成结果。排队满/等待超时分别为 503/504；客户端等待超时不保证 GPU 工作已取消，前端不自动重试。请求耗时包含网络/排队，不是 CUDA 生成时间。/health 失败时先查模型终端与端口。

演示结束分别在前端、API、vLLM 终端 Ctrl+C，确认显存释放。只停止自己的 Redis 实例，不清理模型/缓存或关闭用户其他服务。
