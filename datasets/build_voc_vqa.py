#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
datasets/build_voc_vqa.py —— 基于 VOC 标注程序化构造中文视觉问答数据

为什么程序化构造：
  真实 VQA 数据（COCO-QA / VQAv2）需要下载几十 GB 且多为英文；
  而 VOC 的 XML 里已经有**逐物体的类别与 bbox**，足以程序化生成
  计数 / 存在性 / 列举 / 属性 / 空间关系 五类问答，答案 100% 准确、零成本。

五类任务（与计划书 §3.2 对应）：
  counting   图中有几个 X？          答案来自 bbox 计数（过滤 difficult）
  existence  图中是否有 X？          答案 有/没有（负样本按类别分布采样）
  listing    图中有什么物体？         答案来自类别集合
  attribute  图片主要是什么？         答案来自"非 difficult 且面积最大"的物体
  spatial    X 在 Y 的左边还是右边？  答案来自两个 bbox 中心点的 x 比较

⚠️ 重要设计（避免模型学会作弊）：
  * 负样本（existence 答"没有"）必须按类别频率采样，
    否则模型可以靠"永远答有"拿高分
  * 生成后必须检查答案分布，防止单值坍缩（Week 4 验收项）
  * 空间关系只取左右（x 轴）：Week 2 实测 CLIP 做中心裁剪，
    上下关系更容易因裁剪而失效

用法：
    & $env:MINIVLM_PY datasets/build_voc_vqa.py --n 1000 --out data/processed/debug_1k.jsonl
    & $env:MINIVLM_PY datasets/build_voc_vqa.py --n 10000 --out data/processed/baseline_10k.jsonl
    & $env:MINIVLM_PY datasets/build_voc_vqa.py --n 1000 --outdata/processed/x.jsonl --stats
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.voc_source import VOC_CLASSES, build_index  # noqa: E402

# 类别的中文名（用于生成自然的中文问句）
CLASS_ZH = {
    "aeroplane": "飞机", "bicycle": "自行车", "bird": "鸟", "boat": "船",
    "bottle": "瓶子", "bus": "公交车", "car": "汽车", "cat": "猫",
    "chair": "椅子", "cow": "牛", "diningtable": "餐桌", "dog": "狗",
    "horse": "马", "motorbike": "摩托车", "person": "人",
    "pottedplant": "盆栽", "sheep": "羊", "sofa": "沙发",
    "train": "火车", "tvmonitor": "电视",
}

# 数字的中文写法
NUM_ZH = {0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五",
          6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def _num_zh(n: int) -> str:
    return NUM_ZH.get(n, str(n))


def _load_objects(entry: dict) -> list[dict]:
    """从 VOC 索引条目重新读 XML，拿到每个物体的类别 + bbox。

    voc_source.build_index 只保留了类别集合，这里补上 bbox（空间关系需要）。
    """
    import xml.etree.ElementTree as ET

    ann = Path(entry["path"]).parent.parent / "Annotations" / f"{entry['stem']}.xml"
    if not ann.exists():
        return []
    root = ET.parse(ann).getroot()
    objs = []
    for o in root.findall("object"):
        if o.findtext("difficult") == "1":
            continue
        name = o.findtext("name")
        bb = o.find("bndbox")
        if not name or bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        objs.append({
            "name": name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "area": max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)),
        })
    return objs


def build_samples(
    n: int,
    seed: int = 42,
    entries: list[dict] | None = None,
    balance: bool = True,
) -> list[dict]:
    """生成 n 条问答样本，五类任务配额均衡。

    ⚠️ 设计修正记录：最初版本用「每图最多 max_per_image 条，按固定顺序取」，
    结果 max_per_image=3 时排在后面的 attribute / spatial **一条都生成不出来**
    （被 made < max_per_image 卡死）。正确做法是先统计每类任务可用的图片数，
    再按配额均衡分配，而不是按顺序取前几个。
    """
    rng = random.Random(seed)
    if entries is None:
        entries = build_index()

    # 全局类别频率，用于给 existence 采负样本（关键：防止"永远答有"）
    global_freq: Counter = Counter()
    for e in entries:
        for c in e["classes"]:
            global_freq[c] += 1
    all_classes = [c for c in VOC_CLASSES if global_freq[c] > 0]

    pool = sorted(entries, key=lambda x: x["stem"])
    rng.shuffle(pool)

    # 一次性读入所有图片的物体列表（避免多趟解析 XML）
    objects_map: dict[str, list[dict]] = {}
    for entry in pool:
        objs = _load_objects(entry)
        if objs:
            objects_map[entry["stem"]] = objs
    usable = [e for e in pool if e["stem"] in objects_map]
    if not usable:
        return []

    tasks = ["counting", "existence", "listing", "attribute", "spatial"]

    def eligible(entry: dict, task: str) -> bool:
        objs = objects_map[entry["stem"]]
        if task == "spatial":
            return len({o["name"] for o in objs}) >= 2
        return True

    # 每类任务选一批图片；图不够的任务降低目标量
    per_task_target = n // len(tasks)
    quotas = {t: per_task_target for t in tasks}
    if balance:
        # 按各任务可用图片数缩放（每图最多贡献 1 条该任务）
        for t in tasks:
            avail = sum(1 for e in usable if eligible(e, t))
            quotas[t] = min(per_task_target, avail)
        # 把因上限损失的量补给还有余量的任务
        deficit = n - sum(quotas.values())
        if deficit > 0:
            for t in sorted(tasks, key=lambda k: sum(1 for e in usable if eligible(e, k)) - quotas[k],
                            reverse=True):
                avail = sum(1 for e in usable if eligible(e, t))
                add = min(deficit, avail - quotas[t])
                quotas[t] += add
                deficit -= add
                if deficit <= 0:
                    break

    samples: list[dict] = []
    for task in tasks:
        want = quotas[task]
        if want <= 0:
            continue
        cands = [e for e in usable if eligible(e, task)]
        rng.shuffle(cands)
        for entry in cands:
            if want <= 0:
                break
            objs = objects_map[entry["stem"]]
            s = _make_one(entry, task, objs, global_freq, all_classes, rng)
            if s is not None:
                samples.append(s)
                want -= 1

    rng.shuffle(samples)          # 打散任务顺序，避免 batch 内全同类
    return samples[:n]


def _make_one(entry: dict, task: str, objs: list[dict], global_freq: Counter,
              all_classes: list[str], rng: random.Random) -> dict | None:
    """为一张图生成指定任务的一条样本。"""
    counts = Counter(o["name"] for o in objs)
    present = sorted(counts)

    if task == "counting":
        cls = rng.choice(present)
        k = counts[cls]
        return _mk(entry, task, f"图中一共有几个{CLASS_ZH[cls]}？",
                   f"图中有{_num_zh(k)}个{CLASS_ZH[cls]}。",
                   {"class": cls, "count": k})

    if task == "existence":
        # 一半正样本、一半负样本；负样本按全局类别频率采样
        if rng.random() < 0.5 or not [c for c in all_classes if c not in counts]:
            cls = rng.choice(present)
            return _mk(entry, task, f"图中是否有{CLASS_ZH[cls]}？",
                       f"是的，图中有{CLASS_ZH[cls]}。", {"class": cls, "label": 1})
        absent = [c for c in all_classes if c not in counts]
        w = [global_freq[c] for c in absent]
        cls = rng.choices(absent, weights=w, k=1)[0]
        return _mk(entry, task, f"图中是否有{CLASS_ZH[cls]}？",
                   f"图中没有{CLASS_ZH[cls]}。", {"class": cls, "label": 0})

    if task == "listing":
        zh = [CLASS_ZH[c] for c in present]
        return _mk(entry, task, "图中有什么物体？",
                   "图中有" + "、".join(zh) + "。", {"classes": present})

    if task == "attribute":
        main = max(objs, key=lambda o: o["area"])
        return _mk(entry, task, "图片主要是什么？",
                   f"图片主要是{CLASS_ZH[main['name']]}。",
                   {"class": main["name"], "area": round(main["area"], 1)})

    if task == "spatial":
        # 取中心点 x 差最大的一对不同类别物体（左右关系最明确）
        best = None
        for i, a_obj in enumerate(objs):
            for b_obj in objs[i + 1:]:
                if a_obj["name"] == b_obj["name"]:
                    continue
                d = abs(a_obj["cx"] - b_obj["cx"])
                if best is None or d > best[0]:
                    best = (d, a_obj, b_obj)
        if best is None:
            return None
        _, o1, o2 = best
        left, right = (o1, o2) if o1["cx"] < o2["cx"] else (o2, o1)

        # ⚠️ 必须**随机**决定被问的主体是左边还是右边那个物体。
        # 第一版总是以左侧物体为主体，于是 2000 条 spatial 样本的答案
        # 100% 是「X 在 Y 的左边」（实测答案结尾统计只有 ('左边。', 2000)）。
        # 这样模型只要学会"永远答左边"就能拿满分，空间关系任务完全失效 ——
        # 指标看起来漂亮，实际什么都没学会。
        if rng.random() < 0.5:
            subj, other, ans = left, right, "左边"
        else:
            subj, other, ans = right, left, "右边"

        return _mk(entry, task,
                   f"{CLASS_ZH[subj['name']]}在{CLASS_ZH[other['name']]}的左边还是右边？",
                   f"{CLASS_ZH[subj['name']]}在{CLASS_ZH[other['name']]}的{ans}。",
                   {"subject": subj["name"], "other": other["name"], "answer": ans,
                    "left": left["name"], "right": right["name"],
                    "x_subject": round(subj["cx"], 1),
                    "x_other": round(other["cx"], 1)})

    return None


def _mk(entry: dict, task: str, question: str, answer: str, meta: dict) -> dict:
    return {
        "id": f"{entry['stem']}_{task}",
        "image": entry["path"],
        "task": task,
        "question": question,
        "answer": answer,
        "meta": meta,
    }


def answer_stats(samples: list[dict]) -> dict:
    """答案分布统计——用于发现"单值坍缩"（模型可以靠猜最高频答案作弊）。"""
    per_task: dict[str, dict] = {}
    for task in sorted({s["task"] for s in samples}):
        sub = [s for s in samples if s["task"] == task]
        counter = Counter(s["answer"] for s in sub)
        top_ans, top_cnt = counter.most_common(1)[0]
        per_task[task] = {
            "n": len(sub),
            "unique_answers": len(counter),
            "top_answer": top_ans,
            "top_share": top_cnt / len(sub),
        }
    overall = Counter(s["answer"] for s in samples)
    top_all, top_all_cnt = overall.most_common(1)[0]
    return {
        "n": len(samples),
        "per_task": per_task,
        "global_top_answer": top_all,
        "global_top_share": top_all_cnt / max(len(samples), 1),
        "majority_baseline_acc": top_all_cnt / max(len(samples), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="程序化构造 VOC 中文 VQA 数据")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stats", action="store_true", help="打印答案分布统计")
    ap.add_argument("--no-balance", action="store_true", help="关闭五类任务均衡配额")
    args = ap.parse_args()

    print("建立 VOC 索引 ...")
    entries = build_index()
    print(f"  {len(entries)} 张可用图片")

    print(f"生成 {args.n} 条问答（五类任务均衡配额）...")
    samples = build_samples(args.n, seed=args.seed, entries=entries,
                            balance=not args.no_balance)
    if not samples:
        print("生成失败：没有可用样本")
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n已写出 {len(samples)} 条 -> {out}")
    stats = answer_stats(samples)
    print(f"\n任务分布:")
    for task, v in stats["per_task"].items():
        print(f"  {task:<11} n={v['n']:<5} 唯一答案={v['unique_answers']:<5} "
              f"最高频占比={v['top_share']:.1%}")
    print(f"\n全局最高频答案: {stats['global_top_answer']!r} "
          f"({stats['global_top_share']:.1%})")
    print(f"多数类基线准确率: {stats['majority_baseline_acc']:.1%}  "
          f"<- 模型必须显著超过这个数才有意义")

    if stats["global_top_share"] > 0.3:
        print("\n[警告] 全局最高频答案占比 > 30%，存在单值坍缩风险，建议配平")
    for task, v in stats["per_task"].items():
        if v["n"] >= 50 and v["top_share"] > 0.5:
            print(f"[警告] 任务 {task} 的最高频答案占比 {v['top_share']:.1%}，建议配平")

    if args.stats:
        print("\n=== 完整统计 ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 抽样展示
    print("\n=== 抽样 6 条 ===")
    step = max(1, len(samples) // 6)
    for s in samples[::step][:6]:
        print(f"  [{s['task']}] {s['question']}")
        print(f"        -> {s['answer']}   ({Path(s['image']).name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
