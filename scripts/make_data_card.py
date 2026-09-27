#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/make_data_card.py —— 生成数据卡片（Week 4 验收项）

计划书要求的数据工程验收：
  * 五个任务的样本数与配比统计
  * counting 类答案分布直方图（确认没有**单值坍缩**）
  * image_id 在 train/val/test 之间零交集（泄漏自检）
  * 问答重复率 < 1%

额外做两件这个项目特有的事：
  * **答案长度分布**：答案太长会让"停不下来"的问题更严重
  * **空间关系任务的左右平衡**：若"左边"远多于"右边"，
    模型可以靠猜多数类拿分（spatial 的 top_share 必须接近 50%）

用法：
    & $env:MINIVLM_PY scripts/make_data_card.py \
        --data data/processed/baseline_10k.jsonl --out report/data_stats.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = ["counting", "existence", "listing", "attribute", "spatial"]
TASK_ZH = {
    "counting": "计数", "existence": "存在性", "listing": "列举",
    "attribute": "主要物体", "spatial": "空间关系",
}


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def norm(s: str) -> str:
    return re.sub(r"[\s，。、？！：；,.\?!:;]", "", s)


def try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        cands = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"]
        avail = {f.name for f in font_manager.fontManager.ttflist}
        pick = next((c for c in cands if c in avail), None)
        if pick:
            plt.rcParams["font.sans-serif"] = [pick]
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except Exception:  # noqa: BLE001
        return None


def main() -> int:  # noqa: C901
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if s is not None and hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

    ap = argparse.ArgumentParser(description="生成数据卡片")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--splits", type=str, default="data/processed/splits.json")
    ap.add_argument("--out", type=str, default="report/data_stats.md")
    ap.add_argument("--figdir", type=str, default="report/figures")
    args = ap.parse_args()

    data_path = ROOT / args.data
    records = load_jsonl(data_path)
    print(f"读取 {len(records)} 条 <- {data_path}")

    splits_path = ROOT / args.splits
    sp = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else None

    figdir = ROOT / args.figdir
    figdir.mkdir(parents=True, exist_ok=True)

    # ---------------- 基础统计 ----------------
    n_images = len({r["image"] for r in records})
    per_task = Counter(r["task"] for r in records)
    ans_counter = Counter(r["answer"] for r in records)
    top_ans, top_cnt = ans_counter.most_common(1)[0]
    majority = top_cnt / len(records)

    # 重复率要分两个层次看，否则会误判：
    #   * (question, answer) 组合重复是**正常的** —— 模板化问答里
    #     "图中是否有人？->是的，图中有人。" 会在很多张不同图片上出现。
    #   * 真正要防的是"同一张图 + 同一个问题 + 同一个答案"重复入库（真重复）。
    qa = Counter((norm(r["question"]), norm(r["answer"])) for r in records)
    dup_qa = sum(c - 1 for c in qa.values() if c > 1)
    dup_rate = dup_qa / len(records)          # 模板重复（供参考，不作为验收）

    triples = Counter((r["image"], norm(r["question"]), norm(r["answer"]))
                      for r in records)
    dup_true = sum(c - 1 for c in triples.values() if c > 1)
    dup_true_rate = dup_true / len(records)   # 真重复（验收用这个）

    # 同一问题的重复（不同图可能问同样的问题，这是正常的）
    q_only = Counter(norm(r["question"]) for r in records)

    # 答案长度
    def ans_len(r): return len(r["answer"])
    lens = sorted(ans_len(r) for r in records)
    mean_len = sum(lens) / len(lens)
    p50 = lens[len(lens) // 2]
    p95 = lens[int(len(lens) * 0.95)]

    lines: list[str] = []
    add = lines.append
    add(f"# 数据卡片：{data_path.name}")
    add("")
    add(f"> 由 `scripts/make_data_card.py` 自动生成，所有数字为脚本实测")
    add("")
    add("## 1. 总览")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 样本数 | **{len(records):,}** |")
    add(f"| 唯一图片数 | **{n_images:,}** |")
    add(f"| 每图平均样本数 | {len(records)/n_images:.2f} |")
    add(f"| 任务类别数 | {len(per_task)} |")
    add(f"| 唯一答案数 | **{len(ans_counter):,}** |")
    add(f"| 最高频答案 | `{top_ans}`（{top_cnt} 次） |")
    add(f"| 多数类基线准确率 | **{majority:.2%}** ← 模型必须显著超过它 |")
    add(f"| 真重复（同图+同问+同答） | {dup_true}（**{dup_true_rate:.2%}**，验收要求 < 1%） |")
    add(f"| 问答模板重复率 | {dup_rate:.2%}（模板化数据的正常现象，不作验收） |")
    add(f"| 唯一问题数 | {len(q_only):,} |")
    add(f"| 答案长度 均值 / 中位 / P95 | {mean_len:.1f} / {p50} / {p95} 字 |")
    add("")

    # ---------------- 任务分布 ----------------
    add("## 2. 任务分布")
    add("")
    add("| 任务 | 说明 | 样本数 | 占比 | 唯一答案数 | 最高频答案占比 |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    task_rows = []
    for t in TASKS:
        sub = [r for r in records if r["task"] == t]
        if not sub:
            continue
        c = Counter(r["answer"] for r in sub)
        ta, tc = c.most_common(1)[0]
        share = tc / len(sub)
        task_rows.append((t, len(sub), share))
        add(f"| `{t}` | {TASK_ZH[t]} | {len(sub):,} | {len(sub)/len(records):.1%} | "
            f"{len(c):,} | {share:.1%} |")
    add("")

    warnings: list[str] = []
    for t, n, share in task_rows:
        if share > 0.5:
            warnings.append(f"任务 `{t}` 的最高频答案占比 {share:.1%} > 50%，存在单值坍缩风险")
    if majority > 0.3:
        warnings.append(f"全局最高频答案占比 {majority:.1%} > 30%，建议配平")
    if dup_true_rate >= 0.01:
        warnings.append(f"真重复率 {dup_true_rate:.2%} ≥ 1%，不满足验收标准")

    # ---------------- 空间关系左右平衡 ----------------
    add("## 3. 空间关系任务的左右平衡（本项目特有检查）")
    add("")
    spatial = [r for r in records if r["task"] == "spatial"]
    if spatial:
        left_n = sum(1 for r in spatial if r["answer"].endswith("左边。"))
        right_n = len(spatial) - left_n
        add("`spatial` 任务的答案里「左边 / 右边」必须接近 1:1 —— "
            "否则模型可以靠「永远答左边」拿到接近 100% 的分数，"
            "指标就失去意义。")
        add("")
        add("| 答案 | 数量 | 占比 |")
        add("| --- | ---: | ---: |")
        add(f"| 左边 | {left_n} | {left_n/len(spatial):.1%} |")
        add(f"| 右边 | {right_n} | {right_n/len(spatial):.1%} |")
        add("")
        add(f"**猜「左边」的基线 = {left_n/len(spatial):.1%}** —— "
            f"评测 spatial 时必须与这个数对比，而不是与 5% 的随机基线对比。")
        add("")
        if max(left_n, right_n) / len(spatial) > 0.7:
            warnings.append(
                f"spatial 任务左右失衡：{max(left_n,right_n)/len(spatial):.1%} 是同一边"
            )
        # 生成方式本身的偏置说明
        add("> 说明：生成器取「中心点 x 差最大的一对不同类别物体」，"
            "并**随机**以左边或右边的物体作为被问主体（各 50%）。"
            "早期版本总是以左侧物体为主体，导致答案 100% 是「左边」，"
            "模型可以靠猜多数类拿满分 —— 已在生成器里修正。")
        add("")

    # ---------------- 答案分布图 ----------------
    plt = try_plt()
    figs: list[tuple[str, str]] = []

    if plt is not None:
        # 图 1：任务分布 + 答案长度分布
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ts = [r[0] for r in task_rows]
        axes[0].bar(ts, [r[1] for r in task_rows], color="tab:blue")
        axes[0].set_title("各任务样本数")
        axes[0].set_ylabel("样本数")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].hist([ans_len(r) for r in records], bins=range(0, max(lens) + 2),
                     color="tab:orange", edgecolor="white")
        axes[1].set_title("答案长度分布")
        axes[1].set_xlabel("答案字符数")
        axes[1].set_ylabel("样本数")
        axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        f1 = figdir / f"w4_data_overview_{data_path.stem}.png"
        fig.savefig(f1, bbox_inches="tight")
        plt.close(fig)
        figs.append((f1.name, "任务分布与答案长度分布"))

        # 图 2：counting 答案分布（单值坍缩检查）+ spatial 左右平衡
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        counting = [r for r in records if r["task"] == "counting"]
        if counting:
            cnt = Counter(r["meta"].get("count") for r in counting)
            xs = sorted(k for k in cnt if k is not None)
            axes[0].bar([str(x) for x in xs], [cnt[x] for x in xs], color="tab:green")
            axes[0].set_title("counting 任务：答案数值分布（检查单值坍缩）")
            axes[0].set_xlabel("物体数量")
            axes[0].set_ylabel("样本数")
            axes[0].grid(alpha=0.3, axis="y")
        if spatial:
            axes[1].bar(["左边", "右边"], [left_n, right_n], color="tab:red")
            axes[1].set_title("spatial 任务：左右平衡")
            axes[1].set_ylabel("样本数")
            axes[1].axhline(len(spatial) / 2, color="gray", ls="--", lw=1,
                            label="完美平衡线")
            axes[1].legend()
            axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        f2 = figdir / f"w4_answer_dist_{data_path.stem}.png"
        fig.savefig(f2, bbox_inches="tight")
        plt.close(fig)
        figs.append((f2.name, "counting 数值分布与 spatial 左右平衡"))

        add("## 4. 分布图")
        add("")
        for name, cap in figs:
            add(f"![{cap}](figures/{name})")
            add("")
            add(f"*{cap}*")
            add("")

    # ---------------- 划分检查 ----------------
    add("## 5. 划分与泄漏检查")
    add("")
    if sp:
        s = sp["stats"]
        add(f"划分方式：按 **{sp['by']}**（不是按样本），种子 {sp['seed']}")
        add("")
        add("| 部分 | 图片数 | 样本数 | 占比 |")
        add("| --- | ---: | ---: | ---: |")
        for part in ("train", "val", "test"):
            add(f"| {part} | {s['n_images'][part]:,} | {s['n_samples'][part]:,} | "
                f"{s['n_samples'][part]/len(records):.1%} |")
        add("")
        add(f"- 图片跨 split 交集：**{s['leak_check']}**（脚本硬校验，有交集会直接报错）")
        add("")
        add("| 部分 | 任务分布 |")
        add("| --- | --- |")
        for part in ("train", "val", "test"):
            dist = s["task_dist"][part]
            tot = sum(dist.values()) or 1
            pretty = "  ".join(f"{k}={v}({v/tot:.0%})" for k, v in sorted(dist.items()))
            add(f"| {part} | {pretty} |")
        add("")
        add("> 为什么必须按 image_id 划分：同一张图会贡献多条问答"
            "（counting / existence / listing / ...）。若按样本随机划分，"
            "同一张图的问题会同时出现在训练集与验证集，模型只要「记住这张图」，"
            "验证指标就会虚高 —— 这是最典型的数据泄漏。")
    else:
        add("⚠️ 未找到划分文件，跳过")
    add("")

    # ---------------- 警告汇总 ----------------
    add("## 6. 验收结论")
    add("")
    checks = [
        ("样本数 ≥ 10000（Final 档）", len(records) >= 10000),
        ("五类任务齐全且配比均衡（各 20%±3%）",
         len(task_rows) == 5 and all(abs(n / len(records) - 0.2) < 0.03
                                     for _, n, _ in task_rows)),
        ("多数类基线 < 10%", majority < 0.10),
        ("重复问答率 < 1%（同图+同问+同答）", dup_true_rate < 0.01),
        ("按 image_id 划分且无泄漏",
         bool(sp) and sp["stats"]["leak_check"] == "passed"),
        ("spatial 左右接近平衡（多数边 < 70%）",
         bool(spatial) and max(left_n, right_n) / max(len(spatial), 1) < 0.70),
    ]
    add("| 验收项 | 结果 |")
    add("| --- | :---: |")
    for name, ok in checks:
        add(f"| {name} | {'✅' if ok else '❌'} |")
    add("")

    if warnings:
        add("### 需要关注")
        add("")
        for w in warnings:
            add(f"- ⚠️ {w}")
        add("")
    else:
        add("**无警告。**")
        add("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n数据卡片已写出 -> {out}")
    print(f"  样本 {len(records):,}，图片 {n_images:,}，多数类基线 {majority:.2%}")
    print(f"  真重复（同图+同问+同答）{dup_true_rate:.2%}；"
          f"问答模板重复率 {dup_rate:.2%}（正常现象）")
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    if warnings:
        for w in warnings:
            print(f"  [warn] {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
