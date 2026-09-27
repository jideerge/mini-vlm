"""Fixed checkpoint validation after train-only mirror memorization."""
import json
from scripts.train_mirror_pilot import evaluate, write
from scripts.evaluate_spatial_flip import (ROOT, mirrored_record, load_one, sha256, torch,
    MiniVLM, config_from_yaml, MultipleImageCollator, Trainer, TrainingConfig, load_records, split_records)
OUT=ROOT/'outputs/b16_mirror_overfit32_val'
def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists()
    checkpoints={'baseline':ROOT/'outputs/b16_spatial_clean_cloud/checkpoint_best.pt',
        'overfit192':ROOT/'outputs/b16_mirror_overfit32/checkpoint_best_training.pt'}
    hashes={k:sha256(p) for k,p in checkpoints.items()}
    parts=split_records(load_records(ROOT/'data/processed/spatial_clean_9333.jsonl'),ROOT/'data/processed/splits.json')
    val=[r for r in parts['val'] if r['task']=='spatial']; mirrors=[mirrored_record(r) for r in val]
    guard=[r for r in parts['val'] if r['task']!='spatial']
    ids=set(json.loads((ROOT/'outputs/b16_mirror_overfit32/protocol.json').read_text(encoding='utf-8'))['train_ids'])
    train_images={r['image'] for r in parts['train'] if r['id'] in ids}
    assert len(train_images)==32 and not train_images & {r['image'] for r in val+guard}
    assert len(val)==62 and len(guard)==399 and len({r['image'] for r in val})==62
    protocol={'checkpoint_hashes':hashes,'selection':'Fixed step192 selected on training only; no tuning or training in this run. Existing validation set, not a new untouched test set.',
        'n_spatial_pairs':62,'n_other_tasks':399,'control':'One fixed next-image cyclic shift on original spatial questions.',
        'test_evaluated':False,'val_ids':[r['id'] for r in val+guard]}
    write(OUT/'protocol.json',protocol)
    cfg=config_from_yaml(ROOT/'configs/model_clip_b16_qwen05.yaml');cfg.device='cuda'
    model=MiniVLM(cfg)
    collator=MultipleImageCollator(processor=model.processor,tokenizer=model.tokenizer,max_length=cfg.max_seq_len,num_image_token=model.num_image_token)
    t=Trainer(model,collator,[],[],load_one,TrainingConfig(device='cuda',batch_size=2,cache_features_on_gpu=False,output_dir=str(OUT)))
    unique={r['image']:r for r in val+mirrors+guard}
    t.precompute_features(list(unique.values()));model.vision.cpu();torch.cuda.empty_cache()
    report={'protocol':protocol}
    for name,path in checkpoints.items():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if name=='overfit192':assert payload['global_step']==192
        model.projector.load_state_dict(payload['projector'],strict=True)
        print('EVALUATE',name,flush=True)
        report[name]=evaluate(t,val,mirrors,guard)
        report[name]['checkpoint_step']=payload['global_step']
        write(OUT/'results.partial.json',report)
        print(name,report[name]['summary'],flush=True)
        print('other_tasks',report[name]['guard']['content_per_task'],flush=True)
    assert all(sha256(p)==hashes[k] for k,p in checkpoints.items())
    report['checkpoints_unchanged']=True
    write(OUT/'results.json',report)
    print('DONE',flush=True)
if __name__=='__main__':main()
