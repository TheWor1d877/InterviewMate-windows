# Dify 检索与项目知识

项目相关的题目先向 Dify 检索项目资料，把检索结果交给模型服务生成回答。Dify 只提供检索，最终回答由模型服务产出。

依赖：《3-文本输入与流式回答》。


## 交付物

```text
app/
├── retrieval/
│   ├── base.py                  检索接口
│   ├── dify_retriever.py        Dify 检索实现
│   ├── mock_retriever.py        假检索后端
│   ├── retrieval_models.py      检索结果结构
│   └── context_builder.py       片段整理与上下文拼装
├── core/
│   ├── retrieval_policy.py      是否检索的决策
│   └── question_classifier.py   题型与检索需求判定
└── ui/
    └── panels/
        └── rag_sources.py      检索来源浮层
```

## 检索的触发条件

只有项目相关问题才检索。需要检索的情况：

- 介绍项目架构。
- 解释项目里某个功能怎么实现。
- 解释项目中的接口。
- 说明项目用某个技术的原因。
- 回答项目里遇到的具体问题。
- 询问项目中的代码结构。
- 结合项目代码实现新功能。

这些情况默认不检索：C++ 基本语法、操作系统原理、计算机网络基本知识、数据结构与算法题、纯理论技术问题、不涉及用户项目的编程题。

题型与检索需求是两个独立字段。代码题可能同时是项目题，合成一个枚举会让其中一个信息丢失。

判定放在 `question_classifier.py` 里，用关键词和输入上下文做规则匹配，不调用模型。

## 检索流程

```text
用户提交项目问题
      |
QuestionManager
      |
判定项目相关
      |
Dify 检索接口
      |
整理相关片段
      |
问题与检索上下文一起交给模型服务
      |
流式生成
      |
两栏输出
```

## Dify 调用

端点、鉴权、请求体与响应体的形态见《接口契约》的「Dify 检索端点与鉴权」「Dify 检索请求体」「Dify 检索响应体」。

检索请求体：

```json
{
  "query": "项目中如何实现用户认证",
  "retrieval_model": {
    "search_method": "hybrid_search",
    "reranking_enable": false,
    "score_threshold_enabled": false,
    "score_threshold": 0.0,
    "top_k": 4
  }
}
```

内部知识库用 `retrieval_model`，外部知识库用 `external_retrieval_model`，字段名相同。

`retrieval_model` 的五个字段每次都要带完整。Dify 1.20.0 的检索端点不继承知识库里配好的检索设置，请求体里的 `retrieval_model` 会被逐字段校验，缺一个就以 HTTP 400 拒绝。服务端拒绝时映射到 `E_DIFY_REQUEST`，与响应结构不对的 `E_DIFY_RESPONSE` 区分开。

## 检索参数

```yaml
rag:
  enabled: true
  top_k: 4
  timeout: 8
  max_context_chars: 8000
  score_threshold_enabled: false
  reranking_enable: false
```

`top_k` 先用 4。是否开重排按已有部署的重排模型和实际延迟决定。

分数阈值默认关闭，不预设固定值。不同 Embedding、搜索方式和重排模型的分数分布不一样，写死 0.3 或 0.5 会误杀或放行无关片段。用真实检索样本跑一遍，看分数分布再定。

检索查询超过长度限制时截断到限制值，截断发生在客户端，发出去的请求一定在限制之内。

## 结果整理

从响应里取出片段正文、文档名、分数和片段位置，归一化成一条结果：

```python
class RetrievalResult:
    content: str
    document_name: str
    score: float | None
    metadata: dict
```

四个字段的来历：正文取自 `records[].segment.content`，文档名取自响应里的文档对象，分数取自 `records[].score`，附加信息放片段位置、片段 ID 这些从 `segment` 上带出来的字段。服务没给分数或分数类型不对时分数是 `None`，文档名取不到时是空串。

文档名在 `segment` 嵌套的 `document` 对象里，不在 segment 的直接字段上。取文档名时按 `segment.document.name`、`segment.document_name`、`record.document.name` 三处依次试，都取不到时这条片段仍然保留，只是文档名落空字符串。响应结构见《接口契约》的「Dify 检索响应体」一节。

只保留相关片段。整个知识库或大量无关文本不进 Prompt。

`context_builder.py` 负责拼装上下文：

- 按分数从高到低排列片段。
- 总量截断到 `max_context_chars`，默认 8000 字符。
- 每个片段带上文档名，让模型能引用来源。
- 片段之间用明确分隔，避免相邻片段的内容被模型当成一句。

## 故障降级

Dify 连接失败时：

- 不让整个问题处理过程崩掉。
- 状态栏显示 DIFY 为 degraded。
- 继续直接调用模型服务。
- 回答里不能声称已经检索到项目资料。
- 问题必须依赖项目细节时，说明当前没取到项目检索结果。

Dify 没有检索到相关内容时：

- 用通用知识回答。
- 不伪造项目内部实现。
- 必要时提示缺少项目上下文。

连接失败映射到 `E_DIFY_AUTH`、`E_DIFY_TIMEOUT`、`E_DIFY_RESPONSE`，查询超长映射到 `E_DIFY_QUERY_TOO_LONG`。

## 检索来源展示

按 F3 打开一个检索来源面板，列出当前问题用到的片段：文档名、分数、正文前若干行。

面板是临时的浮层，列出后立即返回两栏布局，不改变主界面的两栏结构。

来源信息同时存进当前会话，模型输出结束后仍可再查。

## 检索接口抽象

检索框架可以替换，所以接口先定下来：

```python
class RetrievalProvider:

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        raise NotImplementedError
```

两个入口的分工：`retrieve` 返回命中的片段列表，失败时抛异常，异常自带错误码。实现类另外提供 `retrieve_outcome`，返回一次检索的完整结果，带 `ok`、`degraded`、`error_code` 与 `error_detail`，失败时把异常收敛成降级结果，不往外抛。

当前实现是 `DifyRetriever`，将来要换成 LlamaIndex 或直连 Qdrant 时加一个新的实现类。

`providers.retrieval` 取 `mock` 时用 `MockRetriever`，不发网络请求，片段取自本地片段池。它服务本地开发与 CI：判定要不要检索、发起检索、整理片段、拼上下文、展示来源这一整条链路没有外部服务也能跑通并被验证。假后端不会失败，`retrieve_outcome` 永远返回 `ok=True`、`degraded=False`。

问题处理模块和模型客户端不关心底层是 Dify 还是别的框架，只看 `RetrievalProvider` 接口。

问题处理模块不直接调检索层，它那边有两个注入点：`retrieve` 回调发起检索，返回拼好的上下文文本与本次用到的来源条目；`on_sources` 回调把来源条目交给界面，界面拿它刷新检索来源面板。两个回调都是可选的，没有注入时只做检索决策，不发起真实检索。

现在不为了迁移而迁移，保留 Dify。

## 验收

- 普通知识题不发出 Dify 请求。
- 算法题不发出 Dify 请求。
- 项目相关题发出 Dify 请求，且请求打到 `/datasets/{id}/retrieve`。
- 代码题加上项目关键词时，两个请求都发，检索上下文进入模型请求。
- 检索查询超过长度限制时，发出的请求体在限制之内。
- Dify 返回空记录时仍然给出通用回答，回答里没有编造的项目细节。
- Dify 连接失败时状态栏显示 degraded，普通问题照常回答。
- Dify 返回体字段有变化时，检索结果解析不抛出未捕获异常。
- 按 F3 能看到当前检索用到的文档名与片段。
