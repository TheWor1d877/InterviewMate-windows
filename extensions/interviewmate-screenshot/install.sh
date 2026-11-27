#!/usr/bin/env bash
# 安装 InterviewMate 静默截图扩展。
#
# 装到 ~/.local/share/gnome-shell/extensions 下并启用。
# Wayland 会话下 GNOME Shell 不支持热重载，装完要注销重新登录一次才生效。

set -euo pipefail

UUID="interviewmate-screenshot@local"
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${HOME}/.local/share/gnome-shell/extensions/${UUID}"

mkdir -p "${DEST}"
cp "${SRC}/metadata.json" "${SRC}/extension.js" "${DEST}/"

if command -v gnome-extensions >/dev/null 2>&1; then
    gnome-extensions enable "${UUID}" || true
    gnome-extensions info "${UUID}" || true
fi

echo
echo "扩展已装到 ${DEST}"
echo "Wayland 下需要注销重新登录一次才生效；登录后用下面这条确认它在线："
echo "  gdbus introspect --session --dest com.interviewmate.Screenshot \\"
echo "    --object-path /com/interviewmate/Screenshot"
echo "再跑 ./interviewmate test 看截图接口的探测结果。"
