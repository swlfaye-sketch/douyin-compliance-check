#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抖音合规分级词表扫描（通用版 v2.0.0）。

用法：
    python scan_text.py <文案文件>
    python scan_text.py <目录> --category ai            # 批量扫目录
    python scan_text.py --text "文案内容"
    python scan_text.py --json-input draft.json          # 多字段输入
    python scan_text.py <文案文件> --category ai,education --json --out report.json

输入格式（--json-input）：
    {
      "category": "ai",
      "fields": {
        "标题": "……",
        "口播": "……",
        "标签": "#a #b",
        "简介": "……"
      }
    }

设计要点（对应 references/wordlist.json）：
  A 档 出现即判 🔴
  B 档 须与行动词或产品名共现才判 🔴，否则降为 🟢
  C 档 须与指定语境共现才升级，否则 🟢
  D 档 统一 🟡（含「最X」构式与白名单）
  品类附加条目：--category 指定后叠加，比通用条目更严
  误报抑制：否定语境、引用规则本身，**按小句邻近**下调；「规避审核」属加重情节，不降级
  共现强度：同小句／同句为强共现；跨句但相距 100 字内降一级；更远不判风险
  AI 声明：检出生成类信号而未检出声明时提示

纯标准库实现，Python 3.8+ 均可运行，不依赖第三方包。
"""

import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST = os.path.join(os.path.dirname(HERE), "references", "wordlist.json")

LEVEL_ORDER = {"red": 0, "yellow": 1, "green": 2}
LEVEL_MARK = {"red": "🔴 必改", "yellow": "🟡 建议改", "green": "🟢 提醒"}
DOWNGRADE = {"red": "yellow", "yellow": "green", "green": "green"}

TEXT_EXT = (".txt", ".md", ".json", ".srt")

# 共现强度参数默认值（可被词表 cooccur 段覆盖）
DEFAULT_COOCCUR = {
    "cross_sentence_max_chars": 100,
    "cross_sentence_level": "weak_medium",
}

COVERAGE_NOTE = (
    "本词表为分级清单而非穷举清单：未收录的表述同样可能违规。"
    "扫描结果只覆盖表述层面，不构成合规结论，也不替代平台审核。"
)


def out(*a):
    print(*a)


def load_wordlist(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 切分

def _split(text, sep_class):
    """按分隔符类切分并保留位置：返回 [(start, end, seg)]。"""
    parts = []
    pos = 0
    for seg in re.split(r"(?<=[%s])" % sep_class, text):
        if seg:
            parts.append((pos, pos + len(seg), seg))
            pos += len(seg)
    if not parts:
        parts = [(0, len(text), text)]
    return parts


def split_sentences(text):
    """整句：。！？与换行。用于共现强度判定。"""
    return _split(text, "。！？!?\\n")


def split_clauses(text):
    """小句：整句分隔符再加逗号、分号、冒号。用于抑制判定。

    ⚠️ 顿号「、」**不作为分隔符**——它用于并列，不改变小句语义完整性。
    若把它当分隔符，「禁止刷单、赌博」会被切成两段，「赌博」因所在段无否定词
    而不再受抑制，造成误报。实测「广告法禁止使用国家级、最高级这类绝对化用语」
    即为此类，故此处排除顿号。
    """
    return _split(text, "。！？!?\\n，,；;：:")


def span_at(spans, pos):
    for s, e, t in spans:
        if s <= pos < e:
            return (s, e, t)
    return spans[-1] if spans else None


def sentence_of(sents, start):
    sp = span_at(sents, start)
    return sp[2].strip() if sp else ""


# ---------------------------------------------------------------- 匹配

def iter_matches(pattern, text):
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        out("  ⚠ 正则错误 %s: %s" % (pattern, exc))
        return
    for m in rx.finditer(text):
        yield m


def resolve_categories(wl, cats):
    """把用户输入的品类串解析成 [(key, label, items)]。未知名给出提示。"""
    out_list = []
    known = wl.get("category_extra", {})
    for c in cats:
        c = (c or "").strip()
        if not c or c == "none":
            continue
        if c in known:
            entry = known[c]
            out_list.append((c, entry.get("label", c), entry.get("items", [])))
        else:
            out("  ⚠ 未知品类「%s」，已忽略。可选：%s"
                % (c, "、".join(k for k in known if k != "desc")))
    return out_list


# ---------------------------------------------------------------- 共现强度

def cooccur_strength(trigger_pos, require_pat, text, sents, clauses, cfg):
    """判定 trigger 与 require 的共现强度。

    返回 (strength, require_pos, distance)：
      strong       同一小句或同一句内 —— 按条目原级执行
      weak_medium  跨句但相距在字数上限内 —— 降一级并标注
      weak         命中但距离过远 —— 不计风险
      none         全文未命中 require —— 不计风险
    """
    try:
        rx = re.compile(require_pat)
    except re.error:
        return ("none", None, None)
    reqs = [m.start() for m in rx.finditer(text)]
    if not reqs:
        return ("none", None, None)

    cl = span_at(clauses, trigger_pos)
    if cl:
        for rp in reqs:
            if cl[0] <= rp < cl[1]:
                return ("strong", rp, rp - trigger_pos)

    se = span_at(sents, trigger_pos)
    if se:
        for rp in reqs:
            if se[0] <= rp < se[1]:
                return ("strong", rp, rp - trigger_pos)

    limit = cfg.get("cross_sentence_max_chars", 100)
    near = [rp for rp in reqs if abs(rp - trigger_pos) <= limit]
    if near:
        rp = min(near, key=lambda x: abs(x - trigger_pos))
        return ("weak_medium", rp, rp - trigger_pos)

    rp = min(reqs, key=lambda x: abs(x - trigger_pos))
    return ("weak", rp, rp - trigger_pos)


# ---------------------------------------------------------------- 抑制判定

def suppress_flags(text, sents, clauses, pos, suppress, esc):
    """判定某处命中是否应被抑制。

    抑制词须与命中词落在**同一小句**内才生效。
    若命中处所在整句出现「规避／绕过 ＋ 审核」类组合，则属加重情节：
    不抑制，且须上调——这是为防止抑制规则把「私信我绕过平台审核」反向降级。
    """
    neg_pat = None
    for r in suppress:
        if r.get("id") == "negation":
            neg_pat = r["pattern"]
            break

    escalated = False
    if esc and esc.get("pattern"):
        se = span_at(sents, pos)
        if se:
            for m in re.finditer(esc["pattern"], se[2]):
                cl = span_at(clauses, se[0] + m.start())
                if neg_pat and cl and re.search(neg_pat, cl[2]):
                    continue  # 「不能绕过审核」是否定表述，不算加重
                escalated = True
                break

    hits = []
    if not escalated:
        cl = span_at(clauses, pos)
        if cl:
            for r in suppress:
                if re.search(r["pattern"], cl[2]):
                    hits.append(r["id"])
    return {"escalated": escalated, "rules": hits}


# ---------------------------------------------------------------- 主扫描

def scan(text, wl, cat_items=None):
    findings = []
    sents = split_sentences(text)
    clauses = split_clauses(text)
    suppress = wl.get("suppress_rules", {}).get("rules", [])
    esc = wl.get("suppress_rules", {}).get("escalate_exceptions")
    ccfg = dict(DEFAULT_COOCCUR)
    ccfg.update(wl.get("cooccur", {}))

    def add(tier, item_id, label, hit, pos, level, advice, extra=None):
        sent = sentence_of(sents, pos)
        rec = {
            "tier": tier,
            "id": item_id,
            "label": label,
            "hit": hit,
            "pos": pos,
            "sentence": sent,
            "level": level,
            "advice": advice,
        }
        rec.update(suppress_flags(text, sents, clauses, pos, suppress, esc))
        if extra:
            rec.update(extra)
        findings.append(rec)

    def combo(tier, item, m, cat_label=None):
        """B／C 档通用处理：按共现强度定级。"""
        strength, rpos, dist = cooccur_strength(
            m.start(), item["require"], text, sents, clauses, ccfg)
        label = item["label"] if not cat_label else "<%s> %s" % (cat_label, item["label"])
        if strength == "strong":
            level, advice = item["level"], item["advice"]
        elif strength == "weak_medium":
            level = DOWNGRADE.get(item["level"], "green")
            advice = ("跨句共现（相距约 %d 字），降一级并须人工复核：%s"
                      % (abs(dist or 0), item["advice"]))
        elif strength == "weak":
            level = "green"
            advice = ("与行动词相距约 %d 字，超出 %d 字的有效共现距离，"
                      "不构成该条目风险。" % (abs(dist or 0),
                                            ccfg.get("cross_sentence_max_chars", 100)))
        else:
            level = "green"
            advice = "单独出现，未与所需语境共现，不构成风险；留意搭配。"
        add(tier, item["id"], label, m.group(0), m.start(), level, advice,
            extra={"cooccur": strength, "cooccur_distance": dist})

    # ---- A 档：出现即判 ----
    for item in wl["tier_a"]["items"]:
        for m in iter_matches(item["pattern"], text):
            add("A", item["id"], item["label"], m.group(0), m.start(),
                item["level"], item["advice"])

    # ---- B 档：组合触发 ----
    for item in wl["tier_b"]["items"]:
        for m in iter_matches(item["trigger"], text):
            combo("B", item, m)

    # ---- C 档：视上下文 ----
    for item in wl["tier_c"]["items"]:
        for m in iter_matches(item["trigger"], text):
            combo("C", item, m)

    # ---- D 档：绝对化用语（含构式与白名单）----
    d = wl["tier_d"]
    white = d.get("whitelist", [])
    seen = set()

    def d_skip(m):
        """白名单：向后再取 4 字比对，命中即不计（如「最后」「最重要」）。

        白名单项按正则前缀匹配，因此可写负向断言，
        例如「第一(?!名)」用于放行序数用法而保留「第一名」。
        """
        seg = text[m.start(): m.start() + len(m.group(0)) + 4]
        for w in white:
            try:
                if re.match(w, seg):
                    return True
            except re.error:
                if seg.startswith(w):
                    return True
        return False

    def d_add(m):
        key = (m.start(), m.end())
        if key in seen:
            return
        seen.add(key)
        add("D", "absolute_language", d["label"], m.group(0), m.start(),
            d["level"], d["advice"])

    for m in iter_matches(d["pattern"], text):
        if not d_skip(m):
            d_add(m)
    if d.get("construct"):
        for m in iter_matches(d["construct"], text):
            if not d_skip(m):
                d_add(m)

    # ---- 品类附加条目 ----
    for ckey, clabel, items in (cat_items or []):
        for item in items:
            for m in iter_matches(item["pattern"], text):
                add("CAT:%s" % ckey, item["id"], "<%s> %s" % (clabel, item["label"]),
                    m.group(0), m.start(), item["level"], item["advice"])

    # ---- 误报抑制：否定语境或规则引用，按小句下调一级 ----
    for f in findings:
        if f.get("escalated"):
            f["advice"] = ("⚠ 与「规避审核」类表述相邻，属加重情节，"
                           "不适用降级，须优先处理：" + f["advice"])
            continue
        if not f.get("rules"):
            continue
        if f["level"] == "red":
            f["level"] = "yellow"
            f["advice"] = "疑似否定语境或规则引用，降一级复核：" + f["advice"]
        elif f["level"] == "yellow":
            f["level"] = "green"
            f["advice"] = "疑似否定语境或规则引用，降为提醒。"

    return findings


def ai_disclosure_check(text, wl):
    """AI 生成声明检测。

    ⚠️ 已声明时零命中是**正确行为**，不报。真正的缺口是：
    文本看起来含 AI 生成成分、却没有声明时，提示补充。
    """
    cfg = wl.get("ai_disclosure")
    if not cfg:
        return {"needs_hint": False, "has_declaration": False}
    if re.search(cfg["declarations"], text):
        return {"needs_hint": False, "has_declaration": True,
                "note": "已检出 AI 生成声明。"}
    m = re.search(cfg["signals"], text)
    if m:
        return {"needs_hint": True, "has_declaration": False,
                "hit": m.group(0), "advice": cfg["advice"]}
    return {"needs_hint": False, "has_declaration": False}


# ---------------------------------------------------------------- 归并与展示

def group_findings(items):
    """按（档位, 条目）归并，同一规则命中多个词只算一处。"""
    groups, order = {}, []
    for f in items:
        k = (f["tier"], f["id"])
        if k not in groups:
            groups[k] = {"label": f["label"], "hits": [], "sents": [],
                         "advice": f["advice"], "escalated": f.get("escalated", False)}
            order.append(k)
        g = groups[k]
        if f["hit"] not in g["hits"]:
            g["hits"].append(f["hit"])
        if f["sentence"] and f["sentence"] not in g["sents"]:
            g["sents"].append(f["sentence"])
        if f.get("escalated"):
            g["escalated"] = True
    return [groups[k] for k in order]


def field_report(field, text, findings, source, show_field):
    reds = [f for f in findings if f["level"] == "red"]
    yels = [f for f in findings if f["level"] == "yellow"]
    grns = [f for f in findings if f["level"] == "green"]

    if show_field:
        out("─── 字段：%s（%d 字） ───" % (field, len(text)))
        out("")

    if not findings:
        out("✅ 该部分未命中任何词表条目。")
        out("")
        return

    def dump(title, items):
        if not items:
            return
        gs = group_findings(items)
        out("%s（%d 处）" % (title, len(gs)))
        for n, g in enumerate(gs, 1):
            flag = "　⚠加重情节" if g.get("escalated") else ""
            out("%d. [%s] 命中：%s%s" % (n, g["label"], "、".join(g["hits"]), flag))
            for s in g["sents"][:2]:
                if len(s) > 60:
                    s = s[:60] + "…"
                out("   所在句：%s" % s)
            out("   处置：%s" % g["advice"])
        out("")

    dump("🔴 必改", reds)
    dump("🟡 建议改", yels)
    dump("🟢 提醒", grns)

    out("小结：必改 %d 处，建议改 %d 处，提醒 %d 处。"
        % (len(group_findings(reds)), len(group_findings(yels)),
           len(group_findings(grns))))
    out("")


def similarity_note(blocks, threshold=0.30):
    """批量场景下的同质化检测：字符 3-gram Jaccard 相似度。"""
    def grams(t):
        t = re.sub(r"\s+", "", t)
        if len(t) < 3:
            return set()
        return set(t[i:i + 3] for i in range(len(t) - 2))

    pairs = []
    gs = [(f, grams(t)) for f, t, _ in blocks]
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            (fa, ga), (fb, gb) = gs[i], gs[j]
            if not ga or not gb:
                continue
            inter = len(ga & gb)
            union = len(ga | gb)
            sim = inter / union if union else 0.0
            if sim >= threshold:
                pairs.append({"a": fa, "b": fb, "similarity": round(sim, 3)})
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs


def report(blocks, wl, source, cat_labels, ai_flags, sim_pairs, coverage_note):
    out("")
    out("【抖音合规初审 · 分级词表扫描】")
    out("扫描对象：%s" % source)
    if cat_labels:
        out("套用品类：%s" % "、".join(cat_labels))
    else:
        out("套用品类：通用标准（未指定品类）")
    out("词表版本：%s　｜　规则依据截至 %s"
        % (wl["meta"]["version"], wl["meta"]["updated"]))
    out("")

    totals = {"red": 0, "yellow": 0, "green": 0}
    any_hit = False
    for field, text, findings in blocks:
        if findings:
            any_hit = True
        for f in findings:
            totals[f["level"]] = totals.get(f["level"], 0) + 1
        field_report(field, text, findings, source, len(blocks) > 1)

    out("═" * 52)
    by_rule = {k: 0 for k in ("red", "yellow", "green")}
    for _f, _t, fs in blocks:
        for lv in ("red", "yellow", "green"):
            by_rule[lv] += len(group_findings([f for f in fs if f["level"] == lv]))
    out("总计：按规则计 —— 必改 %d 处，建议改 %d 处，提醒 %d 处"
        % (by_rule["red"], by_rule["yellow"], by_rule["green"]))
    out("　　　按命中词计 —— 必改 %d 个，建议改 %d 个，提醒 %d 个"
        % (totals["red"], totals["yellow"], totals["green"]))
    if not any_hit:
        out("表述层面未命中任何条目。")
    out("")

    # ---- AI 声明检查 ----
    hints = [f for f in ai_flags if f["needs_hint"]]
    declared = [f for f in ai_flags if f.get("has_declaration")]
    if hints:
        out("⚠ AI 生成声明检查：检出 AI 生成相关内容，但**未检出声明**。")
        for h in hints[:3]:
            out("   命中信号：%s（字段：%s）" % (h["hit"], h["field"]))
        out("   %s" % wl["ai_disclosure"]["advice"])
        out("")
    elif declared:
        out("✅ AI 生成声明检查：已检出声明，无需补充。")
        out("")

    # ---- 同质化提示 ----
    if sim_pairs:
        out("🔁 同质化提示（字符 3-gram 相似度 ≥ 0.30）：")
        for p in sim_pairs[:10]:
            mark = "高度相似" if p["similarity"] >= 0.55 else "较相似"
            out("   %s ↔ %s ：%.2f（%s）" % (p["a"], p["b"], p["similarity"], mark))
        out("   官方口径将「模板化同质化」列为低质内容类型；多条同套路文案须改结构。")
        out("")

    out("注意：本扫描只覆盖表述。以下几项须人工完成——")
    out("  1. 画面形态（纯录屏／无真人出镜／模板图文）")
    out("     见 SKILL.md 3.2 节；操作指引见 references/video-guide.md")
    out("  2. 选题本身是否属被点名品类（AI 课程、股市预测、债务优化、兼职赚钱、线上开店投资）")
    out("     见 SKILL.md 4.2 节")
    out("  3. 简介、置顶评论、评论区话术（全链路）")
    out("  4. 账号状态是否需上调一档（近期违规记录／新号起量期）")
    out("")
    out("⚠ 覆盖范围声明：%s" % coverage_note)
    out("")


# ---------------------------------------------------------------- 输入收集

def read_json_draft(path):
    """若文件是含 fields 的结构化草稿，返回 (fields_dict, category) 或 None。"""
    if not path.lower().endswith(".json"):
        return None
    try:
        with io.open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    fields = doc.get("fields")
    if isinstance(fields, dict) and fields:
        return ({str(k): str(v) for k, v in fields.items()},
                doc.get("category"))
    return None


def collect_blocks(args):
    """返回 ([(field, text, source_desc)], cats, source_label)。"""
    blocks = []

    if args.json_input:
        with io.open(args.json_input, encoding="utf-8") as f:
            doc = json.load(f)
        cats = args.category
        if not cats and doc.get("category"):
            cats = [c for c in re.split(r"[,，\s]+", str(doc["category"])) if c]
        fields = doc.get("fields") or {}
        for name, val in fields.items():
            blocks.append((name, str(val), args.json_input))
        if not fields:
            blocks.append(("（未命名）", str(doc.get("text", "")), args.json_input))
        return blocks, cats, args.json_input

    if args.text:
        return [("（命令行输入）", args.text, "（命令行输入）")], args.category, "（命令行输入）"

    if args.file:
        p = args.file
        if os.path.isdir(p):
            names = []
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith(TEXT_EXT) and not fn.startswith("."):
                        names.append(os.path.join(root, fn))
            cats = args.category
            for fp in names:
                rel = os.path.relpath(fp, p)
                draft = read_json_draft(fp)
                if draft:
                    # 结构化草稿：逐字段当作独立块，字段名带文件名前缀以示来源
                    fields, fcat = draft
                    if not cats and fcat:
                        cats = [c for c in re.split(r"[,，\s]+", str(fcat)) if c]
                    for name, val in fields.items():
                        blocks.append(("%s ｜ %s" % (rel, name), val, fp))
                    continue
                try:
                    with io.open(fp, encoding="utf-8", errors="replace") as f:
                        blocks.append((rel, f.read(), fp))
                except Exception as exc:
                    out("  ⚠ 读取失败 %s：%s" % (fp, exc))
            return blocks, cats, "目录 %s（%d 个文件）" % (p, len(names))
        draft = read_json_draft(p)
        if draft:
            fields, fcat = draft
            cats = args.category or ([c for c in re.split(r"[,，\s]+", str(fcat)) if c]
                                     if fcat else "")
            for name, val in fields.items():
                blocks.append((name, val, p))
            return blocks, cats, p
        with io.open(p, encoding="utf-8", errors="replace") as f:
            return [("（整篇）", f.read(), p)], args.category, p

    text = sys.stdin.read()
    return [("（标准输入）", text, "（标准输入）")], args.category, "（标准输入）"


def main():
    ap = argparse.ArgumentParser(description="抖音合规分级词表扫描（通用版）")
    ap.add_argument("file", nargs="?", help="文案文件路径，或含多个文案的目录")
    ap.add_argument("--text", help="直接传入文案内容")
    ap.add_argument("--json-input", help="多字段输入 JSON（含 category 与 fields）")
    ap.add_argument("--category", default="", help="品类，可逗号分隔：ai,education")
    ap.add_argument("--wordlist", default=WORDLIST, help="词表路径")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out", help="把 JSON 结果写入文件")
    ap.add_argument("--no-similarity", action="store_true",
                    help="批量模式下关闭同质化相似度计算")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    wl = load_wordlist(args.wordlist)
    blocks, cats, source = collect_blocks(args)
    if isinstance(cats, str):
        cats = [c for c in re.split(r"[,，\s]+", cats) if c]

    cat_items = resolve_categories(wl, cats or [])
    cat_labels = [lab for _k, lab, _i in cat_items]

    results = []
    totals = {"red": 0, "yellow": 0, "green": 0}
    display_blocks = []
    ai_flags = []
    for field, text, src in blocks:
        if not text.strip():
            continue
        findings = scan(text, wl, cat_items)
        for f in findings:
            totals[f["level"]] = totals.get(f["level"], 0) + 1
        ai = ai_disclosure_check(text, wl)
        ai["field"] = field
        ai_flags.append(ai)
        results.append({
            "field": field,
            "source": src,
            "chars": len(text),
            "counts_by_rule": {
                "red": len(group_findings([f for f in findings if f["level"] == "red"])),
                "yellow": len(group_findings([f for f in findings if f["level"] == "yellow"])),
                "green": len(group_findings([f for f in findings if f["level"] == "green"])),
            },
            "ai_disclosure": ai,
            "findings": findings,
        })
        display_blocks.append((field, text, findings))

    if not results:
        out("⚠ 输入为空。")
        return 2

    sim_pairs = []
    if len(display_blocks) > 1 and not args.no_similarity:
        sim_pairs = similarity_note(display_blocks)

    if args.json or args.out:
        payload = {
            "source": source,
            "categories": cat_labels,
            "wordlist_version": wl["meta"]["version"],
            "totals_by_hit": totals,
            "ai_disclosure": [a for a in ai_flags if a.get("needs_hint") or a.get("has_declaration")],
            "similarity": sim_pairs,
            "coverage_warning": {
                "statement": COVERAGE_NOTE,
                "limits": [
                    "只覆盖表述，不覆盖画面形态、账号状态、发布节奏",
                    "词表为分级清单，未收录的表述同样可能违规",
                    "不构成合规结论，不替代平台审核",
                ],
                "rule_as_of": wl["meta"]["updated"],
            },
            "blocks": results,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            out("已写出：%s" % args.out)
        if args.json:
            out(text)
    else:
        report(display_blocks, wl, source, cat_labels, ai_flags, sim_pairs, COVERAGE_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
