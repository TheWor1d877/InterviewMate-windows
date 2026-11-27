// InterviewMate 静默截图扩展。
//
// 供 InterviewMate 通过 D-Bus 调用，抓一张整屏画面，屏幕上不弹任何提示。
// 抓图用 GNOME Shell 自己的 Shell.Screenshot 接口，原理见
// 《4-截图输入与多模态》「静默截图扩展」一节。

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Shell from 'gi://Shell';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

// D-Bus 名字与对象路径，InterviewMate 侧按这两个值调用
const BUS_NAME = 'com.interviewmate.Screenshot';
const OBJECT_PATH = '/com/interviewmate/Screenshot';

const IFACE_XML = `
<node>
  <interface name="com.interviewmate.Screenshot">
    <method name="Capture">
      <arg type="s" name="target" direction="in"/>
      <arg type="i" name="monitor" direction="in"/>
      <arg type="ay" name="png" direction="out"/>
    </method>
  </interface>
</node>`;

/**
 * 拿显示器信息。
 *
 * 扩展跑在 GNOME Shell 进程里，global.display 就是 Meta.Display。
 *
 * @returns {Meta.Display} 显示器对象。
 */
function getDisplay() {
    if (typeof global !== 'undefined' && global.display)
        return global.display;
    const shellGlobal = Shell.Global.get();
    if (shellGlobal && shellGlobal.display)
        return shellGlobal.display;
    throw new Error('拿不到显示器信息');
}

/**
 * 算出要抓的那块屏在逻辑坐标下的矩形。
 *
 * Mutter 的 screenshot_area 收的是逻辑坐标，它自己会转成设备像素，
 * 所以这里直接用 Meta.Display 给的几何，不用自己乘缩放系数。
 *
 * @param {string} target - 截图目标，primary 是主屏，monitor 是指定的那块。
 * @param {number} monitor - 显示器序号，0 起数。
 * @returns {object} 矩形，含 x、y、width、height。
 */
function monitorRect(target, monitor) {
    const display = getDisplay();
    const count = display.get_n_monitors();
    let index = display.get_primary_monitor();
    if (target === 'monitor' && Number.isInteger(monitor) && monitor >= 0 && monitor < count)
        index = monitor;
    const rect = display.get_monitor_geometry(index);
    return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
}

/**
 * 把要回给 D-Bus 的字节统一成 Uint8Array。
 *
 * GIO 的 steal_as_bytes 返回的是 GLib.Bytes，不是 Uint8Array，直接
 * new Uint8Array() 拿它会得到长度 0 的空数组，得先 get_data() 取里面的字节。
 *
 * @param {*} raw - 字节数据，可能是 GLib.Bytes、Uint8Array 或普通数组。
 * @returns {Uint8Array} 字节数组。
 */
function toBytes(raw) {
    if (raw instanceof Uint8Array)
        return raw;
    if (raw instanceof GLib.Bytes)
        return raw.get_data();
    if (Array.isArray(raw))
        return Uint8Array.from(raw);
    return new Uint8Array(raw);
}

/** D-Bus 接口实现，方法名与 IFACE_XML 里的方法名一一对应。 */
class ScreenshotImpl {
    /**
     * 抓一张图，返回 PNG 字节。
     *
     * @param {string} target - 截图目标，screen 是整屏，primary 是主屏，
     * monitor 是指定的那块屏。
     * @param {number} monitor - 显示器序号，0 起数。
     * @returns {Promise<GLib.Variant>} 装着 PNG 字节的返回值。
     */
    async Capture(target, monitor) {
        const bytes = await this.capturePngBytes(target, monitor);
        return new GLib.Variant('(ay)', [toBytes(bytes)]);
    }

    /**
     * 抓一张 PNG。
     *
     * 整屏走 Mutter 的 Shell.Screenshot.screenshot，它把画面写成 PNG 直接写进
     * 给定的流；指定某块屏时走 screenshot_area，矩形用逻辑坐标给。
     *
     * @param {string} target - 截图目标。
     * @param {number} monitor - 显示器序号，0 起数。
     * @returns {Promise<Uint8Array>} PNG 文件的字节。
     */
    async capturePngBytes(target, monitor) {
        const shooter = new Shell.Screenshot();
        if (target === 'primary' || target === 'monitor') {
            if (typeof shooter.screenshot_area !== 'function')
                throw new Error('这个 Mutter 版本没有可用的按区域截图接口');
            return this.captureViaArea(shooter, monitorRect(target, monitor));
        }
        if (typeof shooter.screenshot === 'function') {
            try {
                return await this.captureViaStream(shooter);
            } catch (e) {
                logError(e, 'Shell.Screenshot.screenshot 失败，改走画面内容那条路');
            }
        }
        return this.captureViaContent(shooter);
    }

    /**
     * 用 Shell.Screenshot.screenshot_area 抓某一块屏。
     *
     * @param {Shell.Screenshot} shooter - 截图对象。
     * @param {object} rect - 逻辑坐标下的矩形。
     * @returns {Promise<Uint8Array>} PNG 文件的字节。
     */
    async captureViaArea(shooter, rect) {
        const stream = Gio.MemoryOutputStream.new_resizable();
        try {
            await new Promise((resolve, reject) => {
                shooter.screenshot_area(
                    rect.x, rect.y, rect.width, rect.height, stream,
                    (obj, result) => {
                        try {
                            resolve(obj.screenshot_area_finish(result));
                        } catch (e) {
                            reject(e);
                        }
                    });
            });
        } finally {
            stream.close(null);
        }
        return toBytes(stream.steal_as_bytes());
    }

    /**
     * 用 Shell.Screenshot.screenshot 把整屏 PNG 写进内存流。
     *
     * @param {Shell.Screenshot} shooter - 截图对象。
     * @returns {Promise<Uint8Array>} PNG 文件的字节。
     */
    async captureViaStream(shooter) {
        const stream = Gio.MemoryOutputStream.new_resizable();
        try {
            // 用回调形式而不是 Promise 形式，回调形式各版本 Mutter 都在。
            // include_cursor 传 false，画面里不画鼠标指针。
            await new Promise((resolve, reject) => {
                shooter.screenshot(false, stream, (obj, result) => {
                    try {
                        resolve(obj.screenshot_finish(result));
                    } catch (e) {
                        reject(e);
                    }
                });
            });
        } finally {
            stream.close(null);
        }
        return toBytes(stream.steal_as_bytes());
    }

    /**
     * 老版本 Mutter 的取图路子：先拿画面内容，再合成到流里。
     *
     * @param {Shell.Screenshot} shooter - 截图对象。
     * @returns {Promise<Uint8Array>} PNG 文件的字节。
     */
    async captureViaContent(shooter) {
        const content = await new Promise((resolve, reject) => {
            shooter.screenshot_stage_to_content((obj, result) => {
                try {
                    resolve(obj.screenshot_stage_to_content_finish(result));
                } catch (e) {
                    reject(e);
                }
            });
        });
        // 新版本 Mutter 返回的是元组，画面内容是第一个元素
        const picture = Array.isArray(content) ? content[0] : content;
        if (typeof picture?.get_texture !== 'function')
            throw new Error('这个 Mutter 版本没有可用的取纹理接口');
        const texture = picture.get_texture();

        const stream = Gio.MemoryOutputStream.new_resizable();
        try {
            await new Promise((resolve, reject) => {
                Shell.Screenshot.composite_to_stream(
                    texture,
                    0, 0, -1, -1,   // 整块画面：起点与尺寸，负数表示不限
                    1,              // 缩放，1 是原始像素
                    null, 0, 0, 0,  // 不画鼠标指针
                    stream,
                    (obj, result) => {
                        try {
                            resolve(Shell.Screenshot.composite_to_stream_finish(result));
                        } catch (e) {
                            reject(e);
                        }
                    });
            });
        } finally {
            stream.close(null);
        }
        return toBytes(stream.steal_as_bytes());
    }
}

export default class InterviewMateScreenshotExtension extends Extension {
    enable() {
        this._impl = new ScreenshotImpl();
        this._dbus = Gio.DBusExportedObject.wrapJSObject(IFACE_XML, this._impl, true);
        this._dbus.export(Gio.DBus.session, OBJECT_PATH);
        this._nameId = Gio.DBus.session.own_name(
            BUS_NAME, Gio.BusNameOwnerFlags.REPLACE, null, null);
    }

    disable() {
        if (this._nameId)
            Gio.DBus.session.unown_name(this._nameId);
        this._nameId = null;
        if (this._dbus)
            this._dbus.unexport();
        this._dbus = null;
        this._impl = null;
    }
}
