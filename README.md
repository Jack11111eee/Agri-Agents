# Agri-Agents（农业病虫害诊断智能体系统）

基于知识图谱增强的农作物病虫害诊断系统。当前已完成 **M001 最小诊断闭环**：纯文本症状输入 → Kuzu 知识图谱检索 → 受约束的 LLM 生成 → 带溯源的诊断与防治建议，并配一个仅供演示的单页 Web 界面。

> 当前交付范围仅限 M001。记忆管理、图谱自进化、图片多模态、规模化部署等后续里程碑尚未开始，不包含在本 README 中。

## 已实现功能

以下功能均已落地、可运行并通过里程碑验证：

- **确定性检索 / 弃权内核**（`app/kg/`）——Kuzu 嵌入式图库承载水稻种子图谱（稻瘟病、纹枯病、白叶枯病、稻飞虱 4 种高频病害），每条关系边携带 `source` / `version` / `confidence` 溯源三要素。`retrieve()` 只走恒定参数化 Cypher：命中返回结构化子图，未命中返回代码级 `ABSTAIN`，全程不导入、不调用任何 LLM 或网络客户端。
- **受约束生成管线**（`app/pipeline/`）——检索命中后，LLM 仅允许返回 ID-only 实体选择（`ControlMethod` / `Pesticide`）；主诊断与全部展示文本由服务器从检索子图确定性装配，模型文本无法伪装成图谱事实。`INVALID_INPUT` / `ABSTAIN` 在触碰 LLM 之前短路。
- **三态 HTTP API + 演示 Web**（`app/api/main.py`、`web/index.html`）——`POST /diagnose` 返回「命中 / 未命中 / 输入无效」三态稳定 JSON；单页「稻作诊断台」按状态渲染三视图，命中视图分区展示「已验证知识 / 模型补充建议 / 证据链」。
- **可注入的 LLM 客户端**（`app/llm/`）——面向 DeepSeek OpenAI 兼容 JSON 模式；图连接与 LLM 客户端均通过 FastAPI 依赖注入提供，测试用 stub 完全避开真实网络。

## 技术栈

| 层 | 选型 |
|---|---|
| 知识图谱 | Kuzu（嵌入式图库，无需独立服务） |
| Web / API | FastAPI + StaticFiles 单页 |
| LLM | DeepSeek（OpenAI 兼容 JSON 模式） |
| 运行时 | Python ≥ 3.10 |

## 项目结构

```
app/
  api/main.py        FastAPI 应用、POST /diagnose 三态契约、静态页挂载
  kg/loader.py       创建并加载带溯源的水稻种子图谱（Kuzu schema + 数据插入）
  kg/retrieval.py    确定性、零 LLM 的症状检索 / 弃权内核（参数化 Cypher）
  pipeline/diagnose.py  受约束生成管线：ID-only 白名单 grounding、三态结果装配
  llm/client.py      DeepSeek 客户端（可注入、可 stub）
  llm/protocol.py    LLMClient 协议与消息类型
web/index.html       单页演示界面（稻作诊断台）
data/rice_seed_kg.json  水稻种子知识图谱（4 种病害，关系带溯源）
scripts/query_kg.py  命令行查询知识图谱，输出 JSON
tests/               单元 / 契约 / 敏感数据守卫测试
```

## 快速开始

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

开发测试（pytest、ruff）见 `pyproject.toml` 的 `[project.optional-dependencies] dev`。

### 配置 LLM（可选）

本地开发时在 `.env` 中配置 `DEEPSEEK_API_KEY`（`.env` 已 gitignore，密钥只从环境变量读取）：

```bash
echo "DEEPSEEK_API_KEY=你的_key" > .env
```

未配置 key 时，检索命中后的诊断生成会返回 `503`；检索 / 弃权 / 无效输入路径不受影响。已配置 key 后，`tests/test_llm_smoke.py` 会启用一次真实 DeepSeek 命中链路测试。

### 启动 Web 服务

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。演示端点无鉴权，仅限本地回环使用。

### 命令行查询图谱

```bash
python scripts/query_kg.py --symptoms "叶片出现褐色病斑"
python scripts/query_kg.py --symptoms "不存在的症状XYZ" --json
```

## API 契约

`POST /diagnose`，请求体：

```json
{ "symptoms": "叶片出现褐色病斑" }
```

`symptoms` 必须是非空文本，长度上限 2000 字符。响应为三态稳定 JSON：

| status | 含义 | 关键字段 |
|---|---|---|
| `DIAGNOSED` | 检索命中，诊断完成 | `diagnosis`、`verified_knowledge`、`model_suggestions`、`evidence_chain`、`grounding_rejections` |
| `ABSTAINED` | 检索未命中，系统弃权 | `reason`、`matched_symptoms` |
| `INVALID_INPUT` | 输入无效（非文本 / 空白） | `reason` |

主要响应字段：

- `verified_knowledge`：命中子图的实体列表，由代码从图谱确定性装配，未经 LLM 改写。
- `evidence_chain`：关系路径，每条携带 `source` / `version` / `confidence` 溯源三要素。
- `model_suggestions`：模型补充建议，`authoritative=false`、`label=模型补充建议`；仅能引用检索子图内的 `ControlMethod` / `Pesticide` 实体 ID，越界条目被代码过滤并计入 `grounding_rejections`。

## 验证

```bash
python -m pytest        # 当前：66 passed + 1 skipped（未配置 key 时在线 smoke 跳过）
ruff check .            # 当前：clean
```

## 设计要点与安全边界

- **Grounding 由代码控制流保证，而非 LLM 自觉**：未命中时代码级短路，根本不构造生成客户端；命中时生成输出只能过 ID-only allowlist（D006），任何额外字段或越界 ID 都被拒绝 / 过滤。
- **溯源三要素内建于图谱**：每条关系带 `source` / `version` / `confidence`，证据链直接取自检索子图，不经过 LLM。
- **图查询全部参数化**：恒定 Cypher 语句 + 参数字典，症状文本无法成为查询代码。
- **事实与推测显式分区**：系统区分「已验证知识」与「模型建议」，模型建议在界面标注「模型生成，非权威处方」。
- **敏感数据守卫**：不暴露 API Key / 个人联系信息 / 农户身份（`tests/test_sensitive_data.py` 专项覆盖）。

> **部署提示**：M001 定位为单进程 localhost 演示，未包含鉴权、速率限制与供应商成本限制。在任何非回环、共享主机或生产环境暴露前，需先补齐这些部署前置项。

## 许可证与文档

本项目源自国家级大学生创新创业训练计划；仓库内附有项目提案文档（DOCX / PDF）。规划与决策工件（需求、决策、验证记录）见 `.gsd/` 与 `AGENTS.md`。
