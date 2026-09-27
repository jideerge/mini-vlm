"""Answer-only cross entropy with optional emphasis on the left/right token."""
import re
import torch
import torch.nn.functional as F

def direction_weights(batch, records, tokenizer, factor=6.0):
    if factor < 1:
        raise ValueError("factor must be >= 1")
    labels=batch["labels"]
    weights=torch.ones_like(labels,dtype=torch.float32)
    for row,record in enumerate(records):
        if record["task"] != "spatial":
            continue
        matches=list(re.finditer("左边|右边",record["answer"]))
        if len(matches)!=1:
            raise ValueError("spatial answer must contain one direction")
        pos=matches[0].start()
        encoded=tokenizer(record["answer"],add_special_tokens=False,return_offsets_mapping=True)
        start=int(batch["answer_starts"][row]); selected=0
        for j,((lo,hi),tid) in enumerate(zip(encoded["offset_mapping"],encoded["input_ids"])):
            if lo <= pos < hi:
                at=start+j
                if at>=labels.shape[1] or int(labels[row,at])!=tid:
                    raise ValueError("direction label truncated or misaligned")
                weights[row,at]=factor;selected+=1
        if not selected:
            raise ValueError("no direction token found")
    return weights

def weighted_answer_loss(logits, labels, weights):
    labels=labels.to(logits.device);weights=weights.to(logits.device)
    if weights.shape != labels.shape:
        raise ValueError("weights must match labels")
    targets=labels[:,1:];mask=targets!=-100
    if not bool(mask.any()):
        raise ValueError("no answer labels")
    selected_weights=weights[:,1:][mask]
    if not bool(torch.isfinite(selected_weights).all()) or not bool((selected_weights>0).all()):
        raise ValueError("answer weights must be finite and positive")
    losses=F.cross_entropy(logits[:,:-1,:][mask].float(),targets[mask],reduction="none")
    return (losses*selected_weights).sum()/selected_weights.sum()
