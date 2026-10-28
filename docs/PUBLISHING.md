# GitHub 提交与发布步骤

仓库已经完成首次发布。日常修改先在本地分支完成、测试并检查，再用中文提交信息同步 GitHub；不在网页上逐个移动源码。不强制推送或重写历史。

## 已有仓库：目录重构或功能更新

以下命令从项目根目录执行；分支已存在时切换到该分支，不重复创建。模型、环境、缓存和原始数据保持原路径。

```powershell
git status --short
git switch -c refactor/project-layout
# 修改后，按 docs/RUNNING.md 运行测试，并更新本目录的发布文件清单。
git add --pathspec-from-file=docs/GITHUB_RELEASE_FILES.txt
# git mv 会暂存旧路径删除；若使用其他方式移动，还需显式暂存对应的旧路径删除。
git diff --cached --check
git diff --cached --stat
git commit -m '重构：按职责整理项目目录并更新启动方式'
git push -u origin refactor/project-layout
```

可在 GitHub 创建 Pull Request 审阅并合并。只有确认远程 main 未产生冲突且可快进时，才本地 `git switch main`、`git merge --ff-only refactor/project-layout` 后推送 main。不要使用 force push；远端出现他人更新时先拉取并审查，不覆盖它们。

以下保留从零创建仓库的说明，**当前仓库已存在，不重复执行 git init 或 remote add**。首次发布前“无历史”的旧安全结论不能代替当前历史/暂存区检查。

建议仓库名：**SlimLLM**。

Description：`Single-GPU QLoRA fine-tuning and structured ticket extraction with FastAPI, vLLM, Redis and Vue 3.`

Topics：`llm`、`qlora`、`lora`、`qwen3`、`fastapi`、`vllm`、`structured-output`、`redis`、`vue3`、`single-gpu`。

源码采用经所有者确认的 MIT，署名 SlimLLM contributors。模型/完整数据/Adapter/依赖二进制不上传。不要把 API Key 写进命令、remote URL 或截图。若凭据曾发到聊天/日志，应自行在供应商控制台撤销/轮换；不会因 .gitignore 忽略了它就自动失效。

## 1. 核对发布范围

[建议提交清单](GITHUB_RELEASE_FILES.txt) 是当前文件快照，不是日后自动维护的规则。新文件加入后需重新检查。根目录 `reports/github_release_check.md` 是本地最终检查报告，不随公开仓库上传。

公开：Python/WSL/前端源码及测试、依赖清单与 frontend lockfile、手写数据/Schema、提示词非实验模板、MIT/第三方声明、公开文档与聚合指标。

忽略：`.env`、真实凭据、AGENTS/内部计划、虚拟环境、node_modules、dist、模型/Adapter、完整生成数据、原始机器/请求报告、日志、WSL VHD。`.env.example` 仅空值模板，明确允许提交。没有执行文件删除；临时文件清单留在本地报告。

## 2. 初始化并审阅（仅适用于当前无 .git 的目录）

```powershell
Set-Location 'B:\python\SlimLLM'
git init -b main
git config --local user.name 'YOUR_DISPLAY_NAME'
git config --local user.email 'YOUR_GITHUB_NOREPLY_EMAIL'
git add --dry-run --pathspec-from-file=docs/GITHUB_RELEASE_FILES.txt
git status --short --untracked-files=all
git check-ignore .env .venv .cache .wsl frontend/node_modules frontend/dist outputs data/generated
```

替换提交身份占位符，建议用 GitHub noreply 邮箱。不要使用 `git add -f`，不修改/清理已有历史。确认输出没有任何凭据/权重/虚拟磁盘，且必要源码都在清单内，再暂存并检查：

```powershell
git add --pathspec-from-file=docs/GITHUB_RELEASE_FILES.txt
git diff --cached --name-only
git diff --cached --check
git diff --cached --stat
git status --short
```

`.gitignore` 只控制未跟踪文件，不能证明 Git 历史或已暂存文件安全。若出现敏感文件或不能确认的权利来源，停止，不提交/上传；不要自行重写历史。

## 3. 首次提交与远程发布

审阅后由所有者执行：

```powershell
git commit -m '初始化：发布 SlimLLM 微调与推理演示项目'
```

在 GitHub 网站手动创建空的 SlimLLM 仓库，不勾选自动生成 README、.gitignore 或 LICENSE，避免与本地已有文件产生额外初始提交。替换以下用户名，再连接远程并检查目标：

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/SlimLLM.git
git remote -v
git push -u origin main
```

通过 GitHub 支持的凭据管理器/SSH 登录，不把 PAT 写入 remote URL。若已有 origin、远程提交或报非快进，先停下来检查，不 force push。参考 [GitHub 官方已有代码导入说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

发布后检查 README 的 Mermaid/链接、许可证、源码清单及仓库体积，确认权重/数据/凭据没有上传。截图按所有者最新要求不展示；不要伪造模型实验或将未公开制品描述为开箱即用。
