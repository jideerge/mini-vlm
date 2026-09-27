"""Conservative content metrics for this project's Chinese VOC answer templates.

Not a general semantic judge: unsupported wording, uncertainty, contradictions,
and unknown objects are unparsed and count as incorrect in content accuracy.
"""
import re
import unicodedata
from collections import Counter

CONTENT_METRICS_VERSION = 2

CLASS_ZH = dict(zip(
    "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor".split(),
    "飞机 自行车 鸟 船 瓶子 公交车 汽车 猫 椅子 牛 餐桌 狗 马 摩托车 人 盆栽 羊 沙发 火车 电视".split()))
ZH_CLASS = {v: k for k, v in CLASS_ZH.items()}
OBJECT = re.compile("|".join(sorted(ZH_CLASS, key=len, reverse=True)))
NUMBER = re.compile(r"[0-9]+|[零一二两三四五六七八九十百]+")


def compact(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if not c.isspace()
                   and not unicodedata.category(c).startswith("P"))


def number_value(text):
    if text.isdigit():
        return int(text)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    if text in digits:
        return digits[text]
    if re.fullmatch("[一二三四五六七八九]?十[一二三四五六七八九]?", text):
        a, b = text.split("十")
        return digits.get(a, 1) * 10 + digits.get(b, 0)
    return None


def remove_fillers(text, fillers):
    return re.sub("|".join(re.escape(x) for x in sorted(fillers, key=len, reverse=True)), "", text)


def parse_content(task, text, meta):
    raw = unicodedata.normalize("NFKC", text)
    if task == "counting":
        if re.search(r"[-−]\d|\d[.]\d", raw):
            return None
        # Preserve clause boundaries: "1,2" must not turn into the number 12.
        text = "".join("|" if unicodedata.category(c).startswith("P") else c
                       for c in raw if not c.isspace())
    else:
        text = compact(raw)
    if not text or re.search("可能|也许|不确定|似乎|或者|还是|不是|并非|不在|不对", text):
        return None
    # Remove explicit positive quantity phrases only when attached to a known object.
    # Never erase zero, negative/decimal quantities, uncertain counts, or negated counts.
    if task in ("attribute", "listing", "existence") and not re.search("没|无|不", text):
        if re.search(r"[-−]\d|\d[.]\d", raw):
            return None
        quantity = re.compile(r"([0-9]+|[零一二两三四五六七八九十百]+)(?:个|只|辆|匹|头|张|把|棵|架|艘|台)(" + OBJECT.pattern + ")")
        def remove_positive(match):
            value = number_value(match.group(1))
            return match.group(2) if value is not None and value > 0 else match.group(0)
        text = quantity.sub(remove_positive, text)
    classes = {ZH_CLASS[m.group()] for m in OBJECT.finditer(text)}
    rest = OBJECT.sub("", text)
    if task in ("listing", "attribute"):
        if re.search("没|无|不", text):
            return None
        rest = remove_fillers(rest, ["图中包括", "图片中包括", "包括", "图片主要是", "主要是", "图片中有", "图中有", "图片里有",
                                     "图里有", "图中", "图片", "中有", "有", "和", "以及", "与", "是"])
        if rest or not classes or (task == "attribute" and len(classes) != 1):
            return None
        return sorted(classes) if task == "listing" else next(iter(classes))
    if task == "counting":
        if re.search("没|无|不", text):
            return None
        target = meta.get("class")
        if classes and target and classes != {target}:
            return None
        fillers = ["一共有", "总共有", "共有", "图中有", "图片中有", "图中", "图片", "中有", "有", "个", "只", "辆", "匹", "头", "张", "把", "棵", "架", "艘", "台", "共"]
        rest = re.sub("|".join(sorted(fillers, key=len, reverse=True)), "|", rest)
        values = [number_value(m.group()) for m in NUMBER.finditer(rest)]
        if NUMBER.sub("", rest).replace("|", "") or not values or None in values or len(set(values)) != 1:
            return None
        return values[0]
    if task == "existence":
        target = meta.get("class")
        if classes and target and classes != {target}:
            return None
        signs = []
        for match in re.finditer("没有|不存在|不是的|是的|存在|否|无|有|是", rest):
            signs.append(0 if match.group() in ("没有", "不存在", "不是的", "否", "无") else 1)
        residue = re.sub("没有|不存在|不是的|是的|存在|否|无|有|是", "", rest)
        residue = remove_fillers(residue, ["图片中", "图中", "图片里", "图里", "图片", "中"])
        if residue or not signs or len(set(signs)) != 1:
            return None
        return signs[0]
    if task == "spatial":
        # Full relations must name the subject and object in the question's order.
        subj, other = CLASS_ZH.get(meta.get("subject")), CLASS_ZH.get(meta.get("other"))
        if subj and other:
            text = text.replace(subj + "在" + other + "的", "")
            text = text.replace(subj + "在" + other, "")
        directions = re.findall("左边|右边|左侧|右侧", text)
        if re.sub("左边|右边|左侧|右侧", "", text) or not directions:
            return None
        values = {"左边" if d.startswith("左") else "右边" for d in directions}
        return next(iter(values)) if len(values) == 1 else None
    return None


def score_content(record, generated):
    task, meta = record.get("task"), record.get("meta", {})
    expected = {"counting": meta.get("count"), "existence": meta.get("label"),
                "attribute": meta.get("class"), "spatial": meta.get("answer"),
                "listing": sorted(meta["classes"]) if "classes" in meta else None}.get(task)
    if expected is None:
        expected = parse_content(task, record["answer"], meta)
    predicted = parse_content(task, generated, meta)
    parsed = expected is not None and predicted is not None
    correct = parsed and predicted == expected
    ref, pred = [re.sub(r"[\s，。、？！：；,.\?!:;]", "", x)
                 for x in (record["answer"], generated)]
    exact = ref == pred
    clauses = [compact(x) for x in re.split(r"[。！？.!?\n]+", generated)]
    clauses = [x for x in clauses if x]
    repeated = any(n > 1 for n in Counter(clauses).values()) or bool(ref and pred.count(ref) > 1)
    result = {"parsed": parsed, "correct": bool(correct), "predicted": predicted,
              "expected": expected,
              "status": "correct" if correct else ("wrong" if parsed else "unparsed"),
              "correct_nonexact": bool(correct and not exact),
              "reference_prefix_extra": bool(ref and pred.startswith(ref) and pred != ref),
              "repetition_detected": repeated}
    if task == "listing":
        a, b = set(predicted or []), set(expected or [])
        overlap = len(a & b) if parsed else 0
        precision = overlap / len(a) if a else 0.0
        recall = overlap / len(b) if b else 0.0
        result.update(precision=precision, recall=recall,
                      f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return result


def summarize_content(scores):
    n = len(scores)
    rates = {key: sum(bool(s[key]) for s in scores) / n if n else 0.0
             for key in ("parsed", "correct", "correct_nonexact", "reference_prefix_extra", "repetition_detected")}
    result = {"n": n, **rates, "status_counts": dict(Counter(s["status"] for s in scores))}
    for key in ("precision", "recall", "f1"):
        values = [s[key] for s in scores if key in s]
        if values:
            result["listing_macro_" + key] = sum(values) / len(values)
    return result
