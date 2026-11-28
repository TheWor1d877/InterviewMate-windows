"""项目模式的对话日志层。

对话日志按句落库，每个会话一个 md 文件，见
:mod:`app.dialogue.dialogue_log`；双流转写控制器在
:mod:`app.audio.project_mode`，本包只放日志本身。
"""

from app.dialogue.dialogue_log import (
    DEFAULT_DIALOGUE_LOG_DIR,
    QUESTION_MARK_PREFIX,
    ROLE_INTERVIEWER,
    ROLE_SELF,
    ROLES,
    DialogueLog,
)

__all__ = [
    "DEFAULT_DIALOGUE_LOG_DIR",
    "QUESTION_MARK_PREFIX",
    "ROLE_INTERVIEWER",
    "ROLE_SELF",
    "ROLES",
    "DialogueLog",
]