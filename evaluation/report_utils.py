#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluation/report_utils.py —— 把实验结果落成 markdown 报告

为什么单独写这个：
  项目要求"每一个数字都必须来自真实记录"。手工把数字抄进报告极易出错、
  也无法复现。所以让脚本在跑实验的过程中**直接生成报告**，
  数字与结论同源，跑一次就更新一次。

用法：
    rep = Report("Week 2 实验报告", subtitle="...")
    rep.section("实验 A")
    rep.table(["列1", "列2"], [["a", 1], ["b", 2]])
    rep.note("一句话结论")
    rep.save("report/figures/w2_feature_shapes.md")
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "NaN"
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


class Report:
    def __init__(self, title: str, subtitle: str = "", generated_by: str = "") -> None:
        self.lines: list[str] = [f"# {title}"]
        if subtitle:
            self.lines.append(f"\n> {subtitle}")
        meta = [f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        if generated_by:
            meta.append(f"生成脚本：`{generated_by}`")
        meta.append("所有数字由脚本实测产生，非手工填写")
        self.lines.append("\n" + "  \n".join(meta))

    # ------------------------------------------------------------------
    def section(self, title: str, level: int = 2) -> "Report":
        self.lines.append(f"\n{'#' * level} {title}\n")
        return self

    def text(self, s: str) -> "Report":
        self.lines.append(s + "\n")
        return self

    def note(self, s: str) -> "Report":
        self.lines.append(f"> **结论**：{s}\n")
        return self

    def warn(self, s: str) -> "Report":
        self.lines.append(f"> ⚠️ {s}\n")
        return self

    def code(self, s: str, lang: str = "") -> "Report":
        self.lines.append(f"```{lang}\n{s}\n```\n")
        return self

    def bullets(self, items: Iterable[str]) -> "Report":
        for it in items:
            self.lines.append(f"- {it}")
        self.lines.append("")
        return self

    def table(
        self,
        header: Sequence[str],
        rows: Iterable[Sequence[Any]],
        aligns: Sequence[str] | None = None,
    ) -> "Report":
        header = list(header)
        self.lines.append("| " + " | ".join(str(h) for h in header) + " |")
        if aligns:
            sep = []
            for a in aligns:
                sep.append({"l": ":---", "c": ":---:", "r": "---:"}.get(a, ":---"))
            self.lines.append("| " + " | ".join(sep) + " |")
        else:
            self.lines.append("| " + " | ".join(":---" for _ in header) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
        self.lines.append("")
        return self

    def figure(self, rel_path: str, caption: str = "", width: int | None = None) -> "Report":
        w = f" width={width}" if width else ""
        self.lines.append(f"![{caption}]({rel_path}){w}\n")
        if caption:
            self.lines.append(f"*{caption}*\n")
        return self

    # ------------------------------------------------------------------
    def save(self, path: str | Path, echo: bool = False) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.lines).rstrip() + "\n"
        p.write_text(body, encoding="utf-8")
        print(f"\n[报告] 已写出 {p}  ({len(body)} 字符)")
        if echo:
            print(body)
        return p

    def __str__(self) -> str:
        return "\n".join(self.lines)
