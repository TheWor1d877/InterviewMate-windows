# 项目清单（本文件注入题型分类器的 system prompt，作为 project_scope 判定依据。给模型看的极简清单，按下面 4 行格式维护，不要展开写成项目文档。）

## InterviewMate（面试实时辅助工具，即本仓库自身的项目）
定位：面试场景的实时问答助手——本地流式语音识别 + 大模型流式回答 + 项目知识检索，键盘失焦时全局热键可触发语音提问与截图提问
技术栈：Python + Textual(TUI) + FunASR 流式语音识别(paraformer-zh-streaming + fsmn-vad) + OpenAI 兼容模型流式生成(OpenAICompatClient) + Dify 知识库检索(bge-m3 嵌入) + 多进程 socket 通信 + AutoHotkey 全局热键 + mss 截图 + sounddevice/WASAPI 音频采集
核心技术主题：多进程通信、socket 协议设计(token 认证/先回执后台执行)、流式 ASR 管线搭建(200ms 块/聚合阈值/按句落库)、注入检索上下文、前缀缓存、并发处理事件总线、TUI 流式渲染、双流语音的角色分工