#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抖音合规分级词表扫描（通用版）。

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
  D 档 统一 🟡
  品类附加条目：--category 指定后叠加，比通用条目更严
  含误报抑制：否定语境、引用规则本身，等级下调

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

TEXT_EXT = (".txt", ".md", ".json", ".srt")


def out(*a):
    print(*a)


def load_wordlist(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def split_sentences(text):
    """按中文标点与换行分句，返回 [(start, end, sentence)]。"""
    parts = []
    pos = 0
    for seg in re.split(r"(?<=[。！？；!?;\n])", text):
        if seg:
            parts.append((pos, pos + len(seg), seg))
            pos += len(seg)
    if not parts:
        parts = [(0, len(text), text)]
    return parts


def sentence_of(sents, start):
    for s, e, t in sents:
        if s <= start < e:
            return t.strip()
    return ""


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


def scan(text, wl, cat_items=None):
    findings = []
    sents = split_sentences(text)
    suppress = wl.get("suppress_rules", {}).get("rules", [])

    def suppressed(sent):
        """返回命中的抑制规则 id 列表。"""
        hits = []
        for r in suppress:
            if re.search(r["pattern"], sent):
                hits.append(r["id"])
        return hits

    def add(tier, item_id, label, hit, sent, level, advice):
        findings.append({
            "tier": tier,
            "id": item_id,
            "label": label,
            "hit": hit,
            "sentence": sent,
            "level": level,
            "advice": advice,
            "suppressed": suppressed(sent),
        })

    # ---- A 档：出现即判 ----
    for item in wl["tier_a"]["items"]:
        for m in iter_matches(item["pattern"], text):
            sent = sentence_of(sents, m.start())
            add("A", item["id"], item["label"], m.group(0), sent,
                item["level"], item["advice"])

    # ---- B 档：组合触发 ----
    for item in wl["tier_b"]["items"]:
        req_ok = bool(re.search(item["require"], text))
        for m in iter_matches(item["trigger"], text):
            sent = sentence_of(sents, m.start())
            local_ok = bool(re.search(item["require"], sent)) or req_ok
            level = item["level"] if local_ok else "green"
            advice = item["advice"] if local_ok else \
                "单独出现，未与行动词或产品名共现，不构成风险；留意搭配。"
            add("B", item["id"], item["label"], m.group(0), sent, level, advice)

    # ---- C 档：视上下文 ----
    for item in wl["tier_c"]["items"]:
        req_ok = bool(re.search(item["require"], text))
        for m in iter_matches(item["trigger"], text):
            sent = sentence_of(sents, m.start())
            local_ok = bool(re.search(item["require"], sent)) or req_ok
            level = item["level"] if local_ok else "green"
            advice = item["advice"] if local_ok else "语境未达判定条件，仅作留意。"
            add("C", item["id"], item["label"], m.group(0), sent, level, advice)

    # ---- D 档：绝对化用语 ----
    d = wl["tier_d"]
    for m in iter_matches(d["pattern"], text):
        sent = sentence_of(sents, m.start())
        add("D", "absolute_language", d["label"], m.group(0), sent,
            d["level"], d["advice"])

    # ---- 品类附加条目 ----
    for ckey, clabel, items in (cat_items or []):
        for item in items:
            for m in iter_matches(item["pattern"], text):
                sent = sentence_of(sents, m.start())
                add("CAT:%s" % ckey, item["id"], "<%s> %s" % (clabel, item["label"]),
                    m.group(0), sent, item["level"], item["advice"])

    # ---- 误报抑制：否定语境或引用规则本身时下调一级 ----
    for f in findings:
        if not f["suppressed"]:
            continue
        if f["level"] == "red":
            f["level"] = "yellow"
            f["advice"] = "疑似否定语境或规则引用，降一级复核：" + f["advice"]
        elif f["level"] == "yellow":
            f["level"] = "green"
            f["advice"] = "疑似否定语境或规则引用，降为提醒。"

    return findings


def group_findings(items):
    """按（档位, 条目）归并，同一规则命中多个词只算一处。"""
    groups, order = {}, []
    for f in items:
        k = (f["tier"], f["id"])
        if k not in groups:
            groups[k] = {"label": f["label"], "hits": [], "sents": [],
                         "advice": f["advice"]}
            order.append(k)
        g = groups[k]
        if f["hit"] not in g["hits"]:
            g["hits"].append(f["hit"])
        if f["sentence"] and f["sentence"] not in g["sents"]:
            g["sents"].append(f["sentence"])
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
            out("%d. [%s] 命中：%s" % (n, g["label"], "、".join(g["hits"])))
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


def report(blocks, wl, source, cat_labels):
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
    out("注意：本扫描只覆盖表述。以下几项须人工完成——")
    out("  1. 画面形态（纯录屏／无真人出镜／模板图文）")
    out("     见 SKILL.md 3.2 节；必要时跑 extract_video.py 并看图")
    out("  2. 选题本身是否属被点名品类（AI 课程、股市预测、债务优化、兼职赚钱、线上开店投资）")
    out("     见 SKILL.md 4.2 节")
    out("  3. 简介、置顶评论、评论区话术（全链路）")
    out("  4. 账号状态是否需上调一档（近期违规记录／新号起量期）")
    out("")


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
    for field, text, src in blocks:
        if not text.strip():
            continue
        findings = scan(text, wl, cat_items)
        for f in findings:
            totals[f["level"]] = totals.get(f["level"], 0) + 1
        results.append({
            "field": field,
            "source": src,
            "chars": len(text),
            "counts_by_rule": {
                "red": len(group_findings([f for f in findings if f["level"] == "red"])),
                "yellow": len(group_findings([f for f in findings if f["level"] == "yellow"])),
                "green": len(group_findings([f for f in findings if f["level"] == "green"])),
            },
            "findings": findings,
        })
        display_blocks.append((field, text, findings))

    if not results:
        out("⚠ 输入为空。")
        return 2

    if args.json or args.out:
        payload = {
            "source": source,
            "categories": cat_labels,
            "wordlist_version": wl["meta"]["version"],
            "totals_by_hit": totals,
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
        report(display_blocks, wl, source, cat_labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
