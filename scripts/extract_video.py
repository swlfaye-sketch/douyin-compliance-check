#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成片解析：为抖音发布前审核提取可判断的素材。

产出（默认写进 <成片目录>/_review/）：
    transcript.txt        口播全文
    transcript.srt        带时间轴字幕
    frames/               均匀抽取的关键帧
    contact_sheet.jpg     关键帧拼图（在产出目录根下；用 Read 看这一张即可判断画面形态）
    ocr.txt               画面文字识别结果
    review.md             人读汇总
    review.json           机器可读汇总

环境要求（缺哪个就跳过哪一步，不中断整体流程）：
    ffmpeg / ffprobe        —— 动态探测，不硬编码路径
    faster-whisper          —— 语音转写（可选；缺失时报告注明并建议走纯文字审核）
    opencv (cv2)            —— 人脸检测（可选）
    Pillow                  —— 抽帧拼图、色彩与黑边分析（可选）
    tesseract + chi_sim     —— 画面取字（可选；直接以子进程调用，不依赖 pytesseract）

⚠️ 本脚本输出的一切数值均为**辅助信号，不是判据**。
   判低质的是「无真人 ＋ 无实景 ＋ 无实质信息」这一组合，须看图确认。

用法：
    python extract_video.py --video <成片路径>
    python extract_video.py --video <成片路径> --model small --frames 8
    python extract_video.py --video <成片路径> --skip-asr          # 只要画面分析
"""

import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def log(msg):
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- 工具探测

def find_exe(name, extra_patterns=()):
    """动态探测可执行文件。绝不硬编码路径——包管理器安装路径含版本哈希，升级即变。"""
    p = shutil.which(name)
    if p:
        return p
    patterns = list(extra_patterns)
    patterns += [
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*") + r"\**\bin\%s.exe" % name,
        r"C:\ffmpeg\bin\%s.exe" % name,
        r"C:\Program Files\ffmpeg\bin\%s.exe" % name,
        r"C:\Program Files (x86)\ffmpeg\bin\%s.exe" % name,
        r"C:\Program Files\Tesseract-OCR\%s.exe" % name,
        "/usr/bin/%s" % name,
        "/usr/local/bin/%s" % name,
        "/opt/homebrew/bin/%s" % name,
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def run(cmd, timeout=None):
    """执行外部命令，返回 (rc, stdout, stderr)。"""
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, creationflags=(0x08000000 if os.name == "nt" else 0),
        )
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except Exception as exc:
        return -1, "", str(exc)


# ---------------------------------------------------------------- 视频信息

def probe(ffprobe, video):
    rc, so, se = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", video,
    ], timeout=60)
    if rc != 0:
        return {}, se
    try:
        return json.loads(so), ""
    except Exception as exc:
        return {}, str(exc)


def video_info(meta):
    info = {"duration": None, "width": None, "height": None, "fps": None}
    fmt = meta.get("format") or {}
    try:
        info["duration"] = round(float(fmt.get("duration")), 2)
    except Exception:
        pass
    try:
        info["size_bytes"] = int(fmt.get("size"))
    except Exception:
        pass
    for st in meta.get("streams") or []:
        if st.get("codec_type") == "video":
            info["width"] = st.get("width")
            info["height"] = st.get("height")
            fr = st.get("r_frame_rate") or "0/0"
            try:
                a, b = fr.split("/")
                info["fps"] = round(float(a) / float(b), 2) if float(b) else None
            except Exception:
                pass
            info["vcodec"] = st.get("codec_name")
            break
    for st in meta.get("streams") or []:
        if st.get("codec_type") == "audio":
            info["has_audio"] = True
            info["acodec"] = st.get("codec_name")
            break
    else:
        info["has_audio"] = False
    return info


# ---------------------------------------------------------------- 音频与转写

def extract_audio(ffmpeg, video, outdir):
    wav = os.path.join(outdir, "audio_16k.wav")
    rc, so, se = run([
        ffmpeg, "-y", "-i", video, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav,
    ], timeout=600)
    if rc != 0 or not os.path.exists(wav):
        return None, se
    return wav, ""


def resolve_cached_model(name):
    """在 HuggingFace 缓存里找已下载的模型快照。

    faster-whisper 默认会先请求线上校验，网络不通时报 502 并直接失败，
    即使本地模型完整。故先定位本地快照目录，优先离线加载。
    """
    roots = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else "",
    ]
    pats = []
    for base in roots:
        if base:
            pats.append(os.path.join(base, "models--Systran--faster-whisper-%s" % name,
                                     "snapshots", "*"))
    for pat in pats:
        for snap in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(snap, "model.bin")):
                return snap
    return None


def transcribe(wav, model_size, lang, vad, allow_network=False):
    """返回 (segments, note)。segments 元素为 (start, end, text)。

    回退链：本地缓存目录 → 模型名（仅用本地文件）→ 模型名（允许联网下载，**须显式开启**）。

    ⚠️ 默认不联网。本脚本的定位是「纯本地解析、不上传素材」，
    联网下载模型与该定位相冲突，故最后一级回退只在 --allow-network 时启用。
    模型缺失且未开启联网时，如实说明并给出补齐模型的做法。
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None, (
            "未安装 faster-whisper，跳过语音转写。若只有文案，请走纯文字审核（模式 A）。"
        )

    attempts = []
    cached = resolve_cached_model(model_size)
    if cached:
        attempts.append((cached, True, "本地缓存目录"))
    attempts.append((model_size, True, "模型名（仅用本地文件）"))
    if allow_network:
        attempts.append((model_size, False, "模型名（允许联网下载）"))

    last_err = ""
    for target, local_only, desc in attempts:
        try:
            model = WhisperModel(target, device="cpu", compute_type="int8",
                                 local_files_only=local_only)
        except Exception as exc:
            last_err = "%s 加载失败：%s" % (desc, exc)
            continue
        try:
            segs, _info = model.transcribe(wav, language=lang, vad_filter=vad)
            out = []
            for s in segs:
                out.append((round(s.start, 2), round(s.end, 2), (s.text or "").strip()))
            return out, ""
        except Exception as exc:
            last_err = "%s 转写失败：%s" % (desc, exc)
            continue
    if not allow_network and not cached:
        last_err += (
            "\n  ⓘ 本地未找到「%s」模型，且默认不联网。三种处理方式："
            "\n     ① 用 --model base（体积更小，可能已随依赖装好）；"
            "\n     ② 预先下载模型放入 HuggingFace 缓存后重跑；"
            "\n     ③ 显式加 --allow-network 允许下载（会联网，与「纯本地」定位不符，请自行确认）。"
            % model_size
        )
    return None, last_err or "转写失败，原因未知。"


def write_srt(segs, path):
    lines = []
    for i, (st, en, tx) in enumerate(segs, 1):
        lines.append(str(i))
        lines.append("%s --> %s" % (
            "%02d:%02d:%02d,%03d" % (int(st // 3600), int(st % 3600 // 60), int(st % 60), int((st % 1) * 1000)),
            "%02d:%02d:%02d,%03d" % (int(en // 3600), int(en % 3600 // 60), int(en % 60), int((en % 1) * 1000)),
        ))
        lines.append(tx)
        lines.append("")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------- 抽帧与画面

def grab_frames(ffmpeg, video, outdir, n, duration):
    fdir = os.path.join(outdir, "frames")
    os.makedirs(fdir, exist_ok=True)
    frames = []
    if not duration or duration <= 0:
        times = [i * 3 for i in range(n)]
    else:
        lo, hi = duration * 0.05, duration * 0.95
        if hi <= lo:
            lo, hi = 0, max(duration, 1)
        step = (hi - lo) / max(n - 1, 1) if n > 1 else 0
        times = [lo + step * i for i in range(n)]
    for i, t in enumerate(times):
        fp = os.path.join(fdir, "frame_%02d_%05.1fs.jpg" % (i + 1, t))
        rc, so, se = run([
            ffmpeg, "-y", "-ss", "%.2f" % t, "-i", video,
            "-frames:v", "1", "-q:v", "3", fp,
        ], timeout=120)
        if rc == 0 and os.path.exists(fp) and os.path.getsize(fp) > 0:
            frames.append({"index": i + 1, "time": round(t, 2), "path": fp})
    return frames


def build_contact_sheet(frames, outdir, cols=4):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None, "未安装 Pillow，跳过拼图（关键帧仍逐张可用）。"
    if not frames:
        return None, "无可用关键帧。"

    thumb_w = 360
    thumbs = []
    for f in frames:
        try:
            im = Image.open(f["path"]).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        nh = int(h * thumb_w / w) if w else thumb_w
        im = im.resize((thumb_w, max(nh, 1)))
        thumbs.append((f, im))
    if not thumbs:
        return None, "关键帧读取失败。"

    cell_h = max(t[1].size[1] for t in thumbs) + 26
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (thumb_w * cols, cell_h * rows), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    font = None
    for cand in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 "/System/Library/Fonts/PingFang.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(cand):
            try:
                font = ImageFont.truetype(cand, 16)
                break
            except Exception:
                pass

    for i, (f, im) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x, y = c * thumb_w, r * cell_h
        sheet.paste(im, (x, y + 24))
        label = "#%d  %.1fs" % (f["index"], f["time"])
        draw.text((x + 6, y + 4), label, fill=(255, 214, 102), font=font)

    path = os.path.join(outdir, "contact_sheet.jpg")
    sheet.save(path, quality=88)
    return path, ""


def detect_faces(frames, min_ratio=0.10):
    """人脸检测。返回 (per_frame, note)。

    两级确认，目的是压低误报：haar 级联会把远景中的机械结构、建筑轮廓、
    栏杆、模糊色块误判成人脸，此类误报足以把结论带反。
      ① 尺寸门槛：人脸框短边须不小于画面短边的 min_ratio（默认 10%）
         —— 口播特写的脸通常远大于此，远景噪声通常远小于此；
      ② 眼部确认：人脸框内须检出至少一只眼睛。
    同时给出未过滤的原始检出数 faces_raw，便于判断过滤是否过严。
    """
    try:
        import cv2
    except Exception:
        return None, "未安装 opencv，跳过人脸检测（请用 Read 打开拼图人工判断）。"

    base = getattr(getattr(cv2, "data", None), "haarcascades", "")
    fp = os.path.join(base, "haarcascade_frontalface_default.xml")
    ep = os.path.join(base, "haarcascade_eye.xml")
    if not os.path.exists(fp):
        return None, "未找到 haar 人脸模型文件，跳过人脸检测。"
    cc = cv2.CascadeClassifier(fp)
    ec = cv2.CascadeClassifier(ep) if os.path.exists(ep) else None

    res = []
    for f in frames:
        try:
            img = cv2.imread(f["path"])
            if img is None:
                continue
            h, w = img.shape[:2]
            min_side = max(int(min(h, w) * min_ratio), 40)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            raw = cc.detectMultiScale(gray, 1.08, 6, minSize=(min_side, min_side))

            confirmed = 0
            for (x, y, fw, fh) in raw:
                ok = True
                if ec is not None:
                    roi = gray[y:y + fh, x:x + fw]
                    if roi.size:
                        eyes = ec.detectMultiScale(roi, 1.1, 8, minSize=(max(fw // 8, 8),) * 2)
                        ok = len(eyes) >= 1
                    else:
                        ok = False
                if ok:
                    confirmed += 1
            res.append({
                "index": f["index"], "time": f["time"],
                "faces_raw": int(len(raw)), "faces": confirmed,
            })
        except Exception:
            continue
    return res, ""


def frame_motion(frames):
    """相邻关键帧的灰度平均绝对差，用于判断画面是否高度静止。"""
    try:
        from PIL import Image, ImageChops, ImageStat
    except Exception:
        return None
    vals = []
    prev = None
    for f in frames:
        try:
            im = Image.open(f["path"]).convert("L").resize((160, 90))
        except Exception:
            continue
        if prev is not None:
            diff = ImageChops.difference(prev, im)
            vals.append(round(ImageStat.Stat(diff).mean[0], 2))
        prev = im
    if not vals:
        return None
    return {"mean_diff": round(sum(vals) / len(vals), 2), "per_gap": vals}


def _band_is_uniform(im, box, samples=40, tol=26):
    """判断一条采样带是否近似纯色（用于黑边检测）。

    做法：把带子缩成 1 像素高，再逐点取色比对。不用 getdata()——该接口已弃用。
    """
    try:
        band = im.crop(box)
        if band.width < 2 or band.height < 2:
            return False
        strip = band.resize((max(samples, 2), 1))
        px = [strip.getpixel((x, 0)) for x in range(strip.width)]
    except Exception:
        return False
    if not px:
        return False
    r0, g0, b0 = px[0][:3]
    same = 0
    for p in px:
        r, g, b = p[:3]
        if abs(r - r0) <= tol and abs(g - g0) <= tol and abs(b - b0) <= tol:
            same += 1
    return same / len(px) >= 0.92


def detect_letterbox(frames):
    """黑边检测：四边是否存在近纯色的条带。返回统计与逐帧明细。

    ⚠️ 信号，非判据。竖屏原生拍摄出现上下黑边、或刻意留白的设计画面，
       都可能被检出。此信号仅用于提示「可能有录屏／横屏转竖屏」，
       必须看图确认。
    """
    try:
        from PIL import Image
    except Exception:
        return None
    per = []
    for f in frames:
        try:
            im = Image.open(f["path"]).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        if w < 20 or h < 20:
            continue
        band_v = max(int(h * 0.06), 4)
        band_h = max(int(w * 0.06), 4)
        edges = {
            "top": _band_is_uniform(im, (0, 0, w, band_v)),
            "bottom": _band_is_uniform(im, (0, h - band_v, w, h)),
            "left": _band_is_uniform(im, (0, 0, band_h, h)),
            "right": _band_is_uniform(im, (w - band_h, 0, w, h)),
        }
        hit = [k for k, v in edges.items() if v]
        per.append({"index": f["index"], "time": f["time"], "edges": hit})
    if not per:
        return None
    frames_v = sum(1 for p in per if "top" in p["edges"] and "bottom" in p["edges"])
    frames_h = sum(1 for p in per if "left" in p["edges"] and "right" in p["edges"])
    return {
        "per_frame": per,
        "frames_with_vband": frames_v,
        "frames_with_hband": frames_h,
        "total": len(per),
        "note": "检出上下或左右近纯色条带。可能是录屏／横屏转竖屏，也可能是原生竖屏或设计留白，须看图确认。",
    }


def color_richness(frames, bits=4):
    """色彩丰富度：把每通道降到 bits 位后统计被占用色格数与最大单色占比。

    ⚠️ 信号，非判据。三个量一起看才有意义：
      · occupied_bins 低 ＋ top1_share 高  → 提示画面接近少量纯色块，可能是模板图文卡片；
      · 但纯色背景的简约实景口播同样会偏低，**必须看图确认**。
    不用 getdata()（已弃用），改用 getcolors()。
    """
    try:
        from PIL import Image
    except Exception:
        return None
    shift = 8 - max(1, min(bits, 8))
    bins, shares, per = [], [], []
    for f in frames:
        try:
            im = Image.open(f["path"]).convert("RGB").resize((160, 90))
        except Exception:
            continue
        if shift:
            im = im.point(lambda v: (v >> shift) << shift)
        cols = im.getcolors(maxcolors=8192)
        if not cols:
            continue
        total = sum(c for c, _ in cols)
        used = len(cols)
        top1 = max(c for c, _ in cols) / float(total)
        bins.append(used)
        shares.append(top1)
        per.append({"index": f["index"], "time": f["time"],
                    "bins": used, "top1_share": round(top1, 3)})
    if not bins:
        return None
    return {
        "per_frame": per,
        "mean_bins": round(sum(bins) / len(bins), 1),
        "mean_top1_share": round(sum(shares) / len(shares), 3),
        "note": "占用色格数偏低且最大单色占比偏高时，提示画面接近少量纯色块（可能是模板图文卡片）；"
                "简约实景同样会偏低，须看图确认。",
    }


def run_ocr(frames, outdir, tesseract, langs):
    """画面取字。返回 (path, text, per_frame_chars)。"""
    if not tesseract:
        return None, "", None
    lines = []
    per_chars = []
    for f in frames:
        rc, so, se = run([
            tesseract, f["path"], "stdout", "-l", langs, "--psm", "6",
        ], timeout=120)
        txt = (so or "").strip()
        n = len([ch for ch in txt if not ch.isspace()])
        per_chars.append({"index": f["index"], "time": f["time"], "chars": n})
        if txt:
            lines.append("--- #%d  %.1fs ---\n%s" % (f["index"], f["time"], txt))
    out = "\n\n".join(lines)
    path = os.path.join(outdir, "ocr.txt")
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out if out else "（未识别到画面文字）")
    return path, out, per_chars


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="成片解析（抖音发布前审核用）")
    ap.add_argument("--video", required=True, help="成片路径")
    ap.add_argument("--out", help="输出目录，默认 <成片目录>/_review")
    ap.add_argument("--model", default="small", help="whisper 模型：base/small/medium（默认 small）")
    ap.add_argument("--frames", type=int, default=8, help="抽帧数量（默认 8）")
    ap.add_argument("--lang", default="zh", help="转写语言（默认 zh）")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="只转写前 N 秒（0 表示全量）")
    ap.add_argument("--allow-network", action="store_true",
                    help="允许联网下载语音模型（默认关闭，保持纯本地解析）")
    ap.add_argument("--skip-asr", action="store_true", help="跳过语音转写")
    ap.add_argument("--skip-ocr", action="store_true", help="跳过画面取字")
    ap.add_argument("--skip-frames", action="store_true", help="跳过抽帧")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    video = os.path.abspath(args.video)
    if not os.path.exists(video):
        log("✗ 找不到文件：%s" % video)
        return 2

    outdir = os.path.abspath(args.out) if args.out else os.path.join(os.path.dirname(video), "_review")
    os.makedirs(outdir, exist_ok=True)

    notes = []
    result = {
        "video": video,
        "review_dir": outdir,
        "video_info": {},
        "transcript": {},
        "frames": {},
        "faces": {},
        "motion": {},
        "letterbox": {},
        "colors": {},
        "ocr": {},
        "notes": notes,
    }

    ffmpeg = find_exe("ffmpeg")
    ffprobe = find_exe("ffprobe")
    tesseract = find_exe("tesseract")
    log("解析成片：%s" % video)
    log("  ffmpeg   : %s" % (ffmpeg or "✗ 未找到"))
    log("  ffprobe  : %s" % (ffprobe or "✗ 未找到"))
    log("  tesseract: %s" % (tesseract or "✗ 未找到"))
    if not ffmpeg:
        notes.append("未找到 ffmpeg，无法提音频与抽帧。请确认 PATH 或安装位置。")

    duration = None
    if ffprobe:
        meta, err = probe(ffprobe, video)
        info = video_info(meta) if meta else {}
        result["video_info"] = info
        duration = info.get("duration")
        if err:
            notes.append("ffprobe 报错：%s" % err[:200])
        log("  时长 %.2fs  %sx%s  %s fps" % (
            duration or 0, info.get("width"), info.get("height"), info.get("fps")))
    else:
        notes.append("未找到 ffprobe，无法读取时长与分辨率。")

    # --- 语音转写 ---
    if not args.skip_asr:
        if ffmpeg and result["video_info"].get("has_audio", True):
            wav, err = extract_audio(ffmpeg, video, outdir)
            if not wav:
                notes.append("提取音频失败：%s" % err[:200])
                log("  ✗ 提取音频失败")
            else:
                log("  音频已提取，开始转写（模型 %s）…" % args.model)
                segs, tnote = transcribe(wav, args.model, args.lang, vad=False,
                                         allow_network=args.allow_network)
                if segs is None:
                    notes.append("语音转写跳过：%s" % tnote)
                    log("  ✗ 转写跳过：%s" % tnote)
                else:
                    full = "".join(s[2] for s in segs)
                    with io.open(os.path.join(outdir, "transcript.txt"), "w",
                                 encoding="utf-8", newline="\n") as fh:
                        fh.write(full)
                    write_srt(segs, os.path.join(outdir, "transcript.srt"))
                    result["transcript"] = {
                        "segments": len(segs),
                        "chars": len(full),
                        "preview": full[:200],
                        "file": os.path.join(outdir, "transcript.txt"),
                        "srt": os.path.join(outdir, "transcript.srt"),
                    }
                    log("  ✓ 转写完成：%d 段，%d 字" % (len(segs), len(full)))
        else:
            notes.append("视频无音轨，已跳过转写。")

    # --- 抽帧 ---
    frames = []
    if not args.skip_frames and ffmpeg:
        log("  抽取 %d 帧…" % args.frames)
        frames = grab_frames(ffmpeg, video, outdir, args.frames, duration)
        log("  ✓ 实得 %d 帧" % len(frames))
        result["frames"] = {
            "count": len(frames),
            "dir": os.path.join(outdir, "frames"),
            "items": [{"index": f["index"], "time": f["time"], "file": f["path"]} for f in frames],
        }
        if frames:
            sheet, snote = build_contact_sheet(frames, outdir)
            if sheet:
                result["frames"]["contact_sheet"] = sheet
                log("  ✓ 拼图：%s" % sheet)
            elif snote:
                notes.append(snote)

    # --- 画面分析 ---
    if frames:
        per, fnote = detect_faces(frames)
        if per is None:
            notes.append(fnote)
        else:
            hit = sum(1 for p in per if p["faces"] > 0)
            raw = sum(1 for p in per if p.get("faces_raw", 0) > 0)
            ratio = round(hit / len(per), 2) if per else 0
            result["faces"] = {"per_frame": per, "frames_with_face": hit,
                               "total_frames": len(per), "ratio": ratio,
                               "frames_with_raw_hit": raw,
                               "note": "haar 级联误报率较高，原始检出数仅供参考；最终以看图为准。"}
            log("  ✓ 确认人脸：%d/%d 帧（占比 %.0f%%）；原始检出 %d 帧（含误报）"
                % (hit, len(per), ratio * 100, raw))

        motion = frame_motion(frames)
        if motion:
            result["motion"] = motion
            log("  ✓ 帧间差异均值：%.2f" % motion["mean_diff"])

        lb = detect_letterbox(frames)
        if lb:
            result["letterbox"] = lb
            log("  ✓ 黑边：上下条带 %d/%d 帧，左右条带 %d/%d 帧"
                % (lb["frames_with_vband"], lb["total"],
                   lb["frames_with_hband"], lb["total"]))

        cd = color_richness(frames)
        if cd:
            result["colors"] = cd
            log("  ✓ 色彩丰富度：占用色格均值 %.1f，最大单色占比均值 %.3f"
                % (cd["mean_bins"], cd["mean_top1_share"]))

    # --- 画面取字 ---
    if frames and not args.skip_ocr:
        if not tesseract:
            notes.append("未找到 tesseract，跳过画面取字。")
        else:
            path, txt, per_chars = run_ocr(frames, outdir, tesseract, "chi_sim+eng")
            if path:
                result["ocr"] = {"file": path, "chars": len(txt or ""),
                                 "preview": (txt or "")[:200],
                                 "per_frame_chars": per_chars}
                log("  ✓ 画面取字：%d 字" % len(txt or ""))

    # --- 汇总 ---
    write_review(result, outdir)
    log("")
    log("产出目录：%s" % outdir)
    log("  人读汇总：review.md")
    log("  机器汇总：review.json")
    if result["frames"].get("contact_sheet"):
        log("  ⭐ 判断画面形态请用 Read 打开：contact_sheet.jpg（在产出目录根下，不在 frames/ 内）")
    for n in notes:
        log("  提醒：%s" % n)
    return 0


def write_review(r, outdir):
    vi = r.get("video_info") or {}
    tr = r.get("transcript") or {}
    fr = r.get("frames") or {}
    fa = r.get("faces") or {}
    mo = r.get("motion") or {}
    lb = r.get("letterbox") or {}
    cd = r.get("colors") or {}
    oc = r.get("ocr") or {}
    vi_note = r.get("video_info") or {}

    lines = []
    lines.append("# 成片解析汇总")
    lines.append("")
    lines.append("> ⚠️ **本文件的所有数值均为辅助信号，不是判据。**")
    lines.append("> 判低质的是「**无真人 ＋ 无实景 ＋ 无实质信息**」这一组合，**必须看图确认**。")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 文件 | `%s` |" % r["video"])
    lines.append("| 时长 | %s 秒 |" % (vi.get("duration") or "未知"))
    lines.append("| 分辨率 | %s × %s |" % (vi.get("width"), vi.get("height")))
    lines.append("| 帧率 | %s fps |" % vi.get("fps"))
    lines.append("| 音轨 | %s |" % ("有" if vi.get("has_audio") else "无"))
    lines.append("| 文件大小 | %s 字节 |" % (vi.get("size_bytes", "未知")))
    lines.append("")

    lines.append("## 一、语音转写")
    if tr:
        lines.append("")
        lines.append("- 段数：%d　字数：%d" % (tr.get("segments", 0), tr.get("chars", 0)))
        lines.append("- 全文：`transcript.txt`　时间轴：`transcript.srt`")
        lines.append("")
        lines.append("> 开头 200 字：%s" % (tr.get("preview") or "").replace("\n", " "))
        lines.append("")
        lines.append("📌 **转写结果要按表述过一遍词表**（口播里同样会出现引流词、极限词、教学结构）。")
        lines.append("   可直接把 `transcript.txt` 交给 `scripts/scan_text.py` 再扫一次。")
    else:
        lines.append("")
        lines.append("（未转写，原因见文末提醒）")
    lines.append("")

    lines.append("## 二、画面形态（发布前最要紧的一项）")
    lines.append("")
    if fa:
        ratio = fa.get("ratio", 0)
        per = fa.get("per_frame") or []
        raw_hit = sum(1 for p in per if p.get("faces_raw", 0) > 0)
        total = fa.get("total_frames", 0)
        lines.append("- **确认人脸**（经尺寸与眼部两级过滤）：**%d/%d 帧**，占比 **%.0f%%**"
                     % (fa.get("frames_with_face", 0), total, ratio * 100))
        lines.append("- 原始检出（未过滤）：%d/%d 帧。两者之差即被滤掉的疑似误报——"
                     "haar 级联会把远景中的机械结构、建筑轮廓、栏杆、模糊色块误判为人脸，"
                     "**原始数偏高属常态**。" % (raw_hit, total))
        if ratio == 0:
            lines.append("- ⚠ **判定提示：确认人脸为 0，按「无真人出镜」处理。**")
            lines.append("  若题材为推荐／营销类，一律判 🔴（见 SKILL.md 3.2 节）。")
        elif ratio < 0.34:
            lines.append("- ⚠ 真人元素偏少，推荐／营销类题材建议补真人镜头。")
        else:
            lines.append("- 检出的真人面孔较多，但**仍须看图确认不是误报**。")
        lines.append("- 📌 **本项检测仅供初筛，最终判断以看图为准**（haar 级联准确率有限，"
                     "两个方向都可能出错）。")
    else:
        lines.append("- （未做人脸检测）")

    if mo:
        lines.append("- 相邻关键帧灰度差均值：**%.2f**" % mo.get("mean_diff", 0))
        if mo.get("mean_diff", 99) < 3:
            lines.append("  ⚠ 差异极低，**画面高度静止**——典型形态是纯录屏、图文卡片（PPT 式模板画面）或静态图文。")
        elif mo.get("mean_diff", 99) < 8:
            lines.append("  提示：画面变化偏小——常见于录屏、图文卡片、静态图文三类，**须看图区分**。")
        lines.append("  📌 **帧间差异小本身不判低质**；判低质的是「无真人 ＋ 无实景 ＋ 无实质信息」这一组合。")

    if lb:
        lines.append("- 黑边：上下条带 **%d/%d 帧**，左右条带 **%d/%d 帧**"
                     % (lb.get("frames_with_vband", 0), lb.get("total", 0),
                        lb.get("frames_with_hband", 0), lb.get("total", 0)))
        tot = max(lb.get("total", 1), 1)
        half = max(tot // 2, 1)
        v_band = lb.get("frames_with_vband", 0) >= half
        h_band = lb.get("frames_with_hband", 0) >= half
        if v_band and h_band:
            lines.append("  提示：上下与左右均检出纯色边带，**多为设计留白或卡片式版式**；"
                         "若画面内容为界面操作，则须考虑录屏，**看图确认**。")
        elif h_band:
            lines.append("  提示：多数帧检左右纯色边带。竖屏作品常见于**设计留白或卡片式版式**；"
                         "若是横屏素材缩放留边或录屏，**看图确认**。")
        elif v_band:
            lines.append("  提示：多数帧检上下纯色边带。竖屏作品常见于**上下压条版式**；"
                         "若是横屏素材缩放留边或录屏，**看图确认**。")
        lines.append("  📌 **黑边本身不判低质**，它只是一个形态线索；真正要看的是画面里有没有真人、有没有实景。")

    if cd:
        lines.append("- 色彩丰富度：占用色格均值 **%.1f**，最大单色占比均值 **%.3f**"
                     % (cd.get("mean_bins", 0), cd.get("mean_top1_share", 0)))
        low_bins = cd.get("mean_bins", 999) < 150
        high_top1 = cd.get("mean_top1_share", 0) > 0.35
        if low_bins and high_top1:
            lines.append("  ⚠ 色格少且单色占比高，**提示画面接近少量纯色块（可能是模板图文卡片）**；"
                         "但纯色背景的简约实景口播同样如此，**必须看图区分**。")
        elif low_bins or high_top1:
            lines.append("  提示：画面配色较为单一，可能是卡片式版式，也可能是简约实景，**须看图确认**。")

    lines.append("")
    lines.append("**判断画面形态必须看图**：用 Read 打开产出目录根下的 `contact_sheet.jpg`"
                 "（注意：拼图**不在** `frames/` 内），")
    lines.append("重点确认四件事——① 有无真人面部；② 有无录屏痕迹（状态栏、鼠标、竖向黑边）；")
    lines.append("③ 是**实景拍摄**，还是**模板化图文卡片**（PPT 式版式、纯文字卡）；④ 画面是否手持晃动或对焦模糊。")
    lines.append("")

    lines.append("## 三、画面文字")
    if oc:
        lines.append("")
        lines.append("- 识别字数：%d　文件：`ocr.txt`" % oc.get("chars", 0))
        pfc = oc.get("per_frame_chars") or []
        if pfc:
            dense = [p for p in pfc if p["chars"] >= 30]
            lines.append("- 文字密集帧（单帧 ≥ 30 字）：%d/%d 帧" % (len(dense), len(pfc)))
            if len(dense) >= max(len(pfc) * 2 // 3, 1):
                lines.append("  ⚠ 多数帧文字密集，**提示可能是文字卡或图文卡片**——须看图确认。")
        if oc.get("preview"):
            lines.append("")
            lines.append("> 预览：%s" % oc["preview"].replace("\n", " / "))
        lines.append("")
        lines.append("📌 **画面文字以看图为准。** 视频字幕多为描边白字，通用 OCR 对其识别率有限，")
        lines.append("   识别结果常有错字与乱码。`contact_sheet.jpg` 上通常可直接读出全部文字。")
        lines.append("")
        lines.append("📌 **画面文字同样要过词表**：字幕与画面贴纸里的联系方式、极限词、")
        lines.append("   行动引导词，与口播同等判定。")
    else:
        lines.append("")
        lines.append("（未取字）")
    lines.append("")

    if r.get("notes"):
        lines.append("## 四、提醒与跳过项")
        lines.append("")
        for n in r["notes"]:
            lines.append("- %s" % n)
        lines.append("")

    with io.open(os.path.join(outdir, "review.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    with io.open(os.path.join(outdir, "review.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
