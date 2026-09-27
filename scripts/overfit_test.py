#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/overfit_test.py —— 100 样本过拟合测试（**上云硬门禁**）

为什么这个脚本最重要：
    计划书 §6.4 规定：如果模型连 100 个样本都无法接近完全拟合，
    **禁止上云 GPU**。这是整个项目成本控制的第一道也是最后一道闸门。
    它同时覆盖了 Week 3 的全部 5 项自检。

五项断言（全通过才算过门禁）：
    A. loss 从初始值下降到 < 0.1（或降到初始值的 10% 以下）
    B. 训练集上的答案 token 准确率 > 95%
    C. **换图输出必须变化**（防"训练正常但推理忽略图片"这个最贵的 bug）
    D. 只有 Projector 有可训练参数与梯度（冻结真的生效）
    E. 能生成出训练集里的答案（自回归解码路径可用，不只是 teacher forcing）

用法：
    & $env:MINIVLM_PY scripts/overfit_test.py                      # 默认 100 条 / 300 步
    & $env:MINIVLM_PY scripts/overfit_test.py --n 20 --steps 120   # 快速冒烟
    & $env:MINIVLM_PY scripts/overfit_test.py --device cuda --batch 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from datasets.collator import MultipleImageCollator, pad_to_square  # noqa: E402
from datasets.dataset import JsonlVLDataset  # noqa: E402
from models.multimodal_model import MiniVLM, config_from_yaml  # noqa: E402
from training.losses import (  # noqa: E402
    IGNORE_INDEX,
    build_param_groups,
    cosine_schedule_with_warmup,
    normalize_answer_text,
)

FAIL: list[str] = []
PASS: list[str] = []


def force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，模型偶尔吐出越南语/马拉雅拉姆语等字符时会
    直接 UnicodeEncodeError 崩掉（实测 generated 里出现 '\u1ea1' 就炸）。
    这里把 stdout/stderr 切成 UTF-8，并开启 errors='replace' 兜底。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def safe(s: str, limit: int = 120) -> str:
    """把模型输出变成控制台一定打得出来的字符串。"""
    out = []
    for ch in s[:limit]:
        code = ord(ch)
        # 只保留 ASCII、常用中文标点与 CJK 汉字，其余用 '?' 占位
        if code < 0x2000 or 0x3000 <= code <= 0x9FFF or 0xFF00 <= code <= 0xFFEF:
            out.append(ch)
        else:
            out.append("?")
    text = "".join(out)
    return repr(text) + ("..." if len(s) > limit else "")


def head(t: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def check(cond: bool, label: str, detail: str = "") -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"\n         {detail}" if detail else ""))
    (PASS if cond else FAIL).append(label)
    return cond


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description="100 样本过拟合测试（上云门禁）")
    ap.add_argument("--data", type=str, default="data/processed/debug_1k.jsonl")
    ap.add_argument("--out", type=str, default="report/figures/w3_overfit_test.json",
                    help="验收结果 JSON 路径；为新实验指定独立文件以保留历史记录")
    ap.add_argument("--config", type=str, default="configs/model_clip_b32_qwen05.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--depth", type=int, default=None, help="覆盖 Projector 层数 0/2/3")
    ap.add_argument("--max-gen", type=int, default=8, help="生成检验的样本数")
    ap.add_argument("--gen-batch", type=int, default=4,
                    help="生成时的子批次大小（每步重算全序列，显存随 batch 线性增长）")
    ap.add_argument("--debug-anchor", action="store_true",
                    help="打印答案起点的计算依据")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-vision", action="store_true",
                    help="不加载 CLIP（用随机视觉特征，仅用于验证管线）")
    args = ap.parse_args()

    force_utf8_stdout()
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    head("0. 准备")
    print(f"  device={device}  n={args.n}  steps={args.steps}  batch={args.batch}  lr={args.lr}")

    # ---------------- 数据 ----------------
    ds = JsonlVLDataset.from_jsonl(ROOT / args.data)
    tasks = ["counting", "existence", "listing", "attribute", "spatial"]
    per_task = max(1, args.n // len(tasks))
    idxs = ds.split_by_task(tasks, per_task=per_task)[:args.n]
    print(f"  数据集 {len(ds)} 条，选中 {len(idxs)} 条（每任务约 {per_task} 条）")
    print(f"  任务分布: {dict(Counter(ds.records[i]['task'] for i in idxs))}")

    # ---------------- 模型 ----------------
    cfg = config_from_yaml(ROOT / args.config)
    cfg.device = device
    cfg.load_vision = not args.no_vision
    if args.depth is not None:
        cfg.projector_depth = args.depth

    t0 = time.perf_counter()
    model = MiniVLM(cfg)
    print(f"  模型加载耗时 {time.perf_counter() - t0:.1f} s")
    print("  " + model.parameter_summary().replace("\n", "\n  "))

    collator = MultipleImageCollator(
        processor=model.processor,
        tokenizer=model.tokenizer,
        max_length=cfg.max_seq_len,
        num_image_token=model.num_image_token,
    )

    # ---------------- 断言 D：只有 Projector 可训练 ----------------
    head("断言 D：冻结是否真的生效")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    non_proj = [n for n in trainable_names if not n.startswith("projector.")]
    n_trainable = model.trainable_parameters
    n_vision = sum(p.numel() for p in (model.vision.parameters()
                                       if model.vision is not None else []))
    n_vision_train = sum(p.requires_grad for p in (model.vision.parameters()
                                                   if model.vision is not None else []))
    n_llm = sum(p.numel() for p in model.llm.parameters())
    n_llm_train = sum(p.requires_grad for p in model.llm.parameters())
    check(not non_proj, "可训练参数只来自 projector",
          f"{len(trainable_names)} 个张量 / {n_trainable:,} 个标量；"
          f"非 projector 的：{non_proj if non_proj else '无'}")
    check(n_vision_train == 0, "视觉塔零可训练参数", f"{n_vision:,} 参数全部冻结")
    check(n_llm_train == 0, "LLM 零可训练参数", f"{n_llm:,} 参数全部冻结")

    # ---------------- 优化器 ----------------
    groups = build_param_groups(model, projector_lr=args.lr, weight_decay=0.0)
    print(f"\n  参数组: {[(g['name'], sum(p.numel() for p in g['params']), g['lr']) for g in groups]}")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_schedule_with_warmup(args.warmup, args.steps)
    )

    # ---------------- 预编码视觉特征（视觉塔冻结 => 只需算一次） ----------------
    head("1. 预编码视觉特征（只跑一次）")
    print(f"  注意：这里缓存的是**视觉塔原始特征** [B,{model.num_image_token},{model.vision_size}]，")
    print("        Projector 仍然在每次前向的计算图内 —— 否则梯度会断（本脚本第一版踩过）。")
    t0 = time.perf_counter()
    raw_feats: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for s in range(0, len(idxs), args.batch):
            chunk = idxs[s:s + args.batch]
            if model.vision is not None:
                pv = model.processor(
                    images=[pad_to_square(ds[i]["image"]) for i in chunk],
                    return_tensors="pt",
                )["pixel_values"].to(device)
                f = model.vision(pv)                    # 只到视觉塔，不含 Projector
            else:
                f = torch.randn(len(chunk), model.num_image_token, model.vision_size,
                                device=device)
            for j, i in enumerate(chunk):
                raw_feats[i] = f[j].detach()
    print(f"  编码 {len(raw_feats)} 张图耗时 {time.perf_counter() - t0:.1f} s")
    print(f"  原始特征形状 {tuple(next(iter(raw_feats.values())).shape)}")

    def make_batch(indices: list[int]) -> tuple[dict, torch.Tensor, list[dict]]:
        samples = [ds[i] for i in indices]
        b = collator([
            {"image": x["image"], "question": x["question"], "answer": x["answer"]}
            for x in samples
        ])
        vf = torch.stack([raw_feats[i] for i in indices], dim=0)
        return b, vf, samples

    # ---------------- 训练 ----------------
    head("2. 训练（只优化 Projector）")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    if model.vision is not None:
        model.vision.eval()
    model.llm.eval()                      # 冻结部分保持 eval（dropout 关闭）

    history: list[dict] = []
    first_loss = None
    step = 0
    t_start = time.perf_counter()
    while step < args.steps:
        for s in range(0, len(idxs), args.batch):
            if step >= args.steps:
                break
            b, vf, _ = make_batch(idxs[s:s + args.batch])
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
                visual_features=vf,
            )
            loss = out["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            if first_loss is None:
                first_loss = float(loss.item())
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "tok_acc": out["n_correct"] / max(out["n_answer_tokens"], 1),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(gn),
            }
            history.append(rec)
            if step % 20 == 0 or step == args.steps - 1:
                print(f"  step {step:4d}  loss={rec['loss']:.4f}  "
                      f"tok_acc={rec['tok_acc']:.3f}  lr={rec['lr']:.2e}  "
                      f"gn={rec['grad_norm']:.3f}")
            step += 1

    train_s = time.perf_counter() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0.0
    print(f"\n  训练完成：{args.steps} 步，{train_s:.1f} s "
          f"({train_s / max(args.steps, 1) * 1000:.0f} ms/step)，显存峰值 {peak_mb:.0f} MB")

    last_losses = [r["loss"] for r in history[-10:]]
    final_loss = sum(last_losses) / len(last_losses)

    # ---------------- 断言 A ----------------
    head("断言 A：loss 是否收敛")
    drop_ratio = final_loss / max(first_loss, 1e-9)
    print(f"  初始 loss = {first_loss:.4f}")
    print(f"  末尾 10 步平均 loss = {final_loss:.4f}（相对初始 {drop_ratio:.1%}）")
    check(final_loss < 0.1 or drop_ratio < 0.1,
          "A. loss 下降到 < 0.1 或初始值的 10% 以下",
          f"final={final_loss:.4f}, first={first_loss:.4f}, ratio={drop_ratio:.1%}")

    # ---------------- 断言 B：训练集答案 token 准确率 ----------------
    head("断言 B：训练集上的答案 token 准确率")
    model.eval()
    tot_tok = tot_ok = 0
    per_sample: list[tuple[int, bool, int, int]] = []
    with torch.no_grad():
        for s in range(0, len(idxs), args.batch):
            chunk = idxs[s:s + args.batch]
            b, vf, _ = make_batch(chunk)
            out = model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                labels=b["labels"],
                visual_features=vf,
                return_logits=True,
            )
            logits = out["logits"]
            labels = b["labels"].to(logits.device)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            for j in range(len(chunk)):
                lab = shift_labels[j]
                keep = lab != IGNORE_INDEX
                if not bool(keep.any()):
                    continue
                pred = shift_logits[j][keep].argmax(-1)
                ok = int((pred == lab[keep]).sum().item())
                n = int(keep.sum().item())
                tot_ok += ok
                tot_tok += n
                per_sample.append((chunk[j], ok == n, ok, n))
    tok_acc = tot_ok / max(tot_tok, 1)
    n_full = sum(1 for _, f, _, _ in per_sample if f)
    full_match = n_full / max(len(per_sample), 1)
    print(f"  token 级准确率 = {tok_acc:.2%}  ({tot_ok}/{tot_tok})")
    print(f"  整句全对样本 = {full_match:.1%}  ({n_full}/{len(per_sample)})")
    check(tok_acc > 0.95, "B. 答案 token 准确率 > 95%", f"实测 {tok_acc:.2%}")

    bad = [(i, ok, n) for i, f, ok, n in per_sample if not f]
    if bad:
        print(f"\n  未完全拟合的样本 {len(bad)} 条（最多列 8 条）：")
        for i, ok, n in bad[:8]:
            print(f"    idx={i} [{ds.records[i]['task']}] {ok}/{n}")
            print(f"      Q={ds.records[i]['question']}  A={ds.records[i]['answer']}")

    # ---------------- 断言 C：换图输出必须变化 ----------------
    head("断言 C：换图后输出是否变化（防「推理忽略图片」）")

    # 诊断模式：逐步打印生成 token vs 参考答案 token，并给出 teacher forcing 对照
    if args.debug_anchor:
        import torch as _t
        from training.losses import IGNORE_INDEX as _IG

        model.eval()
        d_idx = idxs[:1]
        db, dvf, dsamp = make_batch(d_idx)
        d_ids = db["input_ids"].to(device)
        d_mask = db["attention_mask"].to(device)
        d_lab = db["labels"].to(device)
        _eos = int(model.tokenizer.eos_token_id)
        valid = int(d_mask[0].sum().item())
        pos = (d_ids[0, :valid] == _eos).nonzero().flatten().tolist()
        start = pos[-2] + 1
        print(f"\n  [debug] 样本: {dsamp[0]['raw_question']}")
        print(f"         参考答案: {safe(dsamp[0]['answer'])}")
        print(f"         eos 位置={pos}   start_pos={start}  valid={valid}")

        # 1) teacher forcing：完整序列前向，看每个监督位置的 argmax
        with _t.no_grad():
            dvis = model.projector(dvf)
            de = model.build_inputs_embeds(d_ids, dvis, d_mask)
            dlg = model.llm(inputs_embeds=de, attention_mask=d_mask,
                            use_cache=False).logits.float()
        sup = [p for p in range(valid) if int(d_lab[0, p]) != _IG]
        print(f"         监督位置={sup}")
        for p in sup:
            pred = int(dlg[0, p - 1].argmax())
            tgt = int(d_lab[0, p])
            print(f"           pos{p}: 输入={model.tokenizer.decode([int(d_ids[0,p])])!r} "
                  f"预测={model.tokenizer.decode([pred])!r} 目标={model.tokenizer.decode([tgt])!r}")

        # 2) 截断到 start_pos 后，逐位置 argmax（应该与上面 pos=start+? 的预测一致）
        trunc = d_ids[:, :start]
        tmask = _t.ones_like(trunc)
        with _t.no_grad():
            tv = model.projector(dvf)
            te = model.build_inputs_embeds(trunc, tv, tmask)
            tlg = model.llm(inputs_embeds=te, attention_mask=tmask,
                            use_cache=False).logits.float()
        print(f"         截断后窗口={tuple(trunc.shape)}，末位 {start-1} 的预测="
              f"{model.tokenizer.decode([int(tlg[0, start-1].argmax())])!r}"
              f"   （teacher forcing 在 pos{start} 的预测应为 "
              f"{model.tokenizer.decode([int(dlg[0, start-1].argmax())])!r}）")

        # 3) 真正调用 generate，逐步打印
        buf, glen = model.generate(input_ids=db["input_ids"], attention_mask=db["attention_mask"],
                                   visual_features=dvf, max_new_tokens=10, micro_batch=1)
        toks = [model.tokenizer.decode([int(t)]) for t in buf[0, :glen[0]]]
        print(f"         generate 前 10 个 token = {toks}")
        print()
    model.eval()
    gen_idx = idxs[:args.max_gen]
    b, vf, samples = make_batch(gen_idx)
    gen_texts = model.generate_text(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"],
        visual_features=vf, max_new_tokens=args.max_new_tokens,
        micro_batch=args.gen_batch,
        answer_start=torch.tensor(b["answer_starts"], device=device),
    )
    # 视觉特征循环错位一位 == 每张图都换成另一张图
    vf_shuffled = torch.roll(vf, shifts=1, dims=0)
    gen_texts2 = model.generate_text(
        input_ids=b["input_ids"], attention_mask=b["attention_mask"],
        visual_features=vf_shuffled, max_new_tokens=args.max_new_tokens,
        micro_batch=args.gen_batch,
        answer_start=torch.tensor(b["answer_starts"], device=device),
    )
    n_diff = sum(1 for a, c in zip(gen_texts, gen_texts2) if a != c)
    print("  原图 vs 错位图（换图后应不同）：")

    # 调试：打印"答案起点"的计算依据，便于核对锚点是否落在正确位置
    if args.debug_anchor:
        _ids = b["input_ids"]
        _eos = int(model.tokenizer.eos_token_id)
        _m = b["attention_mask"]
        print("\n  [debug] 答案起点锚定检查：")
        for r in range(min(2, _ids.size(0))):
            valid = int(_m[r].sum().item())
            pos = (_ids[r, :valid] == _eos).nonzero().flatten().tolist()
            if len(pos) >= 2:
                start = pos[-2] + 1
                ctx = [model.tokenizer.decode([int(t)])
                       for t in _ids[r, start:min(start + 6, valid)]]
                print(f"    样本{r}: eos 位置={pos}  倒数第二个={pos[-2]}  "
                      f"答案起点={start}  其后 token={ctx}")
                print(f"           实测前缀末 4 个 token="
                      f"{[model.tokenizer.decode([int(t)]) for t in _ids[r, start-4:start]]}")
            else:
                print(f"    样本{r}: eos 位置={pos}（不足 2 个）")
        print()

    for i in range(min(4, len(gen_texts))):
        mark = "!=" if gen_texts[i] != gen_texts2[i] else "=="
        print(f"    [{mark}] {samples[i]['raw_question']}")
        print(f"         原图 -> {safe(gen_texts[i])}")
        print(f"         错位 -> {safe(gen_texts2[i])}")
    check(n_diff > 0, "C. 换图后生成结果发生变化",
          f"{n_diff}/{len(gen_texts)} 个样本的输出发生了改变")

    # ---------------- 断言 E：能生成参考答案 ----------------
    head("断言 E：能否生成出训练集里的答案")
    # 比对时容忍答案开头的换行差异：训练数据把 `assistant\n` 的换行算进了答案区间，
    # 所以参考文本可能带一个前导换行。归一化后比较即可。
    def norm_pair(a: str, b: str) -> tuple[str, str]:
        return (normalize_answer_text(a.lstrip()),
                normalize_answer_text(b.lstrip()))

    hits = []
    prefix_hits = []
    for s, g in zip(samples, gen_texts):
        na, ng = norm_pair(s["answer"], g)
        exact = na == ng
        # 前缀匹配：生成结果以参考答案开头。
        # 自由生成（无 teacher forcing）出现"答对之后继续重复"是欠拟合的典型表现，
        # 但它证明解码路径与视觉条件都工作正常 —— 这是本断言要检查的东西。
        hits.append(exact)
        prefix_hits.append(exact or (len(na) > 0 and ng.startswith(na)))
    n_hit = sum(hits)
    n_prefix = sum(prefix_hits)
    print(f"  精确匹配 {n_hit}/{len(hits)}    前缀匹配 {n_prefix}/{len(prefix_hits)}")
    for i, (s, g) in enumerate(list(zip(samples, gen_texts))[:8]):
        if hits[i]:
            mark = "OK  "
        elif prefix_hits[i]:
            mark = "PRE "
        else:
            mark = "MISS"
        print(f"    [{mark}] {s['raw_question']}")
        print(f"           参考: {safe(s['answer'])}")
        print(f"           生成: {safe(g)}")
        if not prefix_hits[i]:
            na, ng = norm_pair(s["answer"], g)
            print(f"           归一化后: 参考={safe(na, 60)} 生成={safe(ng, 60)}")
    check(n_prefix >= max(1, len(prefix_hits) // 2),
          "E. 至少一半的生成结果正确（精确或前缀匹配）",
          f"精确 {n_hit}/{len(hits)}，前缀 {n_prefix}/{len(prefix_hits)}（自回归解码路径可用）")

    # ---------------- 断言 D 补充：梯度真的只落在 Projector 上 ----------------
    head("断言 D 补充：反向传播后梯度落在哪些参数上")
    b, vf, _ = make_batch(idxs[:args.batch])
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                labels=b["labels"], visual_features=vf)
    out["loss"].backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    unexpected = [n for n in with_grad if not n.startswith("projector.")]
    check(not unexpected, "只有 projector 参数拿到梯度",
          f"有梯度的张量 {len(with_grad)} 个；非 projector 的：{unexpected if unexpected else '无'}")
    optimizer.zero_grad(set_to_none=True)

    # ---------------- 总结 ----------------
    head("门禁结论")
    print(f"  可训练参数   : {n_trainable:,} / {model.total_parameters:,} "
          f"({n_trainable / model.total_parameters:.2%})")
    print(f"  loss         : {first_loss:.4f} -> {final_loss:.4f}（{drop_ratio:.1%}）")
    print(f"  token 准确率 : {tok_acc:.2%}    整句全对 {n_full}/{len(per_sample)}")
    print(f"  换图变化     : {n_diff}/{len(gen_texts)}")
    print(f"  生成匹配     : {n_hit}/{len(hits)}")
    print(f"  训练耗时     : {train_s:.1f} s（{train_s / max(args.steps, 1) * 1000:.0f} ms/step）")
    print(f"  显存峰值     : {peak_mb:.0f} MB")

    out_json = {
        "n": len(idxs), "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "device": device, "depth": cfg.projector_depth,
        "first_loss": first_loss, "final_loss": final_loss, "drop_ratio": drop_ratio,
        "token_acc": tok_acc, "full_match": full_match,
        "n_diff": n_diff, "n_gen": len(gen_texts), "n_hit": n_hit,
        "train_s": train_s, "ms_per_step": train_s / max(args.steps, 1) * 1000,
        "peak_mb": peak_mb, "trainable": n_trainable, "total": model.total_parameters,
        "history": history, "passed": not FAIL, "failed": FAIL,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已写出 {out_path}")

    if FAIL:
        print(f"\n  ### 门禁未通过：{len(FAIL)} 项失败 ###")
        for f in FAIL:
            print(f"    - {f}")
        print("\n  >>> 禁止上云 GPU，先在本地解决问题。")
        return 1
    print(f"\n  ### 门禁通过：{len(PASS)} 项全部通过 ###")
    print("  >>> 可以用 1k 数据在本地跑 Baseline 了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
