# 测试范围与制品前提

这不是模型训练或真实推理测试套件，HTTP/模型 fixture 不产生模型准确率。

## 源码可独立运行的测试

在已安装所需 Python 依赖、没有模型/Adapter/冻结生成数据的源码副本上，以下四组共 **42 项通过、无跳过**：

- `test_schema.py`：公开 Schema、手写样例及边界。
- `test_stage1.py`：JSON/标签/数据处理与预算控制契约。
- `test_api.py`：FastAPI 请求/输出/缓存/错误控制流，使用明确的测试 fixture。
- `test_request_queue.py`：有界 FIFO、超时、取消和关闭行为。

命令见 [运行说明](../docs/RUNNING.md)。前端 `node --test frontend/tests/api.test.js` 可在 Node 22.14 上执行 **10 项** API 契约测试，不加载 Vue 页面或模型；构建演示页仍需安装 Vue/Vite。

## 完整本机测试

本机具备未公开制品时，`python -B -m unittest discover -s tests -v` 为 **97 项通过**。其余测试中的部分用例读取：

- `data/generated/stage1/records.jsonl`、`splits.json`：固定 ID、数据哈希与训练/测试隔离检查。
- `reports/baseline/run.json`：冻结提示词及来源信息；推理模板不能代替完整实验清单。
- `outputs/qlora/final_adapter/`：Adapter 配置及权重身份，仅做离线元数据/哈希验证，不加载 GPU 模型。

发布前对仅含公开候选文件的副本实际执行完整命令，97 项中出现 **4 failures、47 errors**，主要是文件缺失引起，少数测试预期另一种错误而断言不匹配。这个结果不是“全量源码测试通过”；不能拿本机 97 项替代它。没有复制私有制品去掩盖问题，没有跳过原断言或更改测试以凑通过。

完整数据、Adapter 未公开，因此当前仓库提供源码与部分可复现契约验证，不宣称干净机器开箱即用。若以后要让所有测试完全脱离制品，需要单独处理测试隔离；本次发布准备保持核心代码和原测试不变。
