"""Local validation-only paired-mirror projector pilot with matched control."""
import json, random, time, gc
from pathlib import Path
from scripts.evaluate_spatial_flip import (ROOT, mirrored_record, load_one, swap_direction, sha256,
    torch, MiniVLM, config_from_yaml, MultipleImageCollator, Trainer, TrainingConfig, load_records, split_records)

OUT=ROOT/'outputs/b16_mirror_pilot'

def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def evaluate(trainer, originals, mirrors, guard):
    a=trainer.evaluate_generation(originals,detail_limit=None,max_new_tokens=24,micro_batch=2)
    b=trainer.evaluate_generation(mirrors,detail_limit=None,max_new_tokens=24,micro_batch=2)
    wrong=originals[1:]+originals[:1]
    c=trainer.evaluate_generation(originals,visual_records=wrong,detail_limit=None,max_new_tokens=24,micro_batch=2)
    g=trainer.evaluate_generation(guard,detail_limit=None,max_new_tokens=24,micro_batch=2)
    pairs=[]
    for x,y in zip(a['details'],b['details']):
        px,py=x['content']['predicted'],y['content']['predicted']
        pairs.append({'id':x['id'],'original':x,'mirror':y,'flipped':px in ('左边','右边') and py==swap_direction(px)})
    summary={'n':len(originals),'original_correct':sum(p['original']['content']['correct'] for p in pairs),
       'mirror_correct':sum(p['mirror']['content']['correct'] for p in pairs),
       'both_correct':sum(p['original']['content']['correct'] and p['mirror']['content']['correct'] for p in pairs),
       'flipped':sum(p['flipped'] for p in pairs),'wrong_image_correct':sum(d['content']['correct'] for d in c['details']),
       'guard_n':len(guard),'guard_correct':sum(d['content']['correct'] for d in g['details'])}
    return {'summary':summary,'pairs':pairs,'wrong_image':c,'guard':g}

def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists(), 'Use a separate output directory for a new experiment'
    ckpt=ROOT/'outputs/b16_spatial_clean_cloud/checkpoint_best.pt'
    source_hash=sha256(ckpt)
    payload=torch.load(ckpt,map_location='cpu',weights_only=True)
    initial={k:v.clone() for k,v in payload['projector'].items()}
    parts=split_records(load_records(ROOT/'data/processed/spatial_clean_9333.jsonl'),ROOT/'data/processed/splits.json')
    train=[r for r in parts['train'] if r['task']=='spatial']
    random.Random(42).shuffle(train); train=train[:256]
    val=[r for r in parts['val'] if r['task']=='spatial']
    flip_train=[mirrored_record(r) for r in train]
    flip_val=[mirrored_record(r) for r in val]
    guard=[]
    for task in ['attribute','counting','existence','listing']:
        choices=[r for r in parts['val'] if r['task']==task]
        random.Random(42).shuffle(choices); guard.extend(choices[:10])
    assert len(val)==62 and len(guard)==40
    assert not {r['image'] for r in train}&{r['image'] for r in val+guard}
    assert len({r['image'] for r in val})==len(val)
    protocol={'seed':42,'source_sha256':source_hash,'source_step':payload['global_step'],
       'train_ids':[r['id'] for r in train],'val_ids':[r['id'] for r in val],'guard_ids':[r['id'] for r in guard],
       'steps_per_arm':64,'batch':2,'accum':4,'lr':1e-4,'warmup':8,
       'selection':'Lowest paired spatial validation loss among baseline, step32 and step64. No test evaluation.',
       'control':'Each selected original repeated twice; treatment original plus actual mirrored image with swapped answer. Same source weights, optimizer, seed and sample slots.',
       'gate':'Exploratory pass: treatment paired accuracy >=60%, both-correct >=25%, flipped >=25%; paired accuracy >= control +5 percentage points; guard loses at most 4/40 vs baseline; original exceeds fixed wrong-image control by >=10 percentage points.'}
    write(OUT/'protocol.json',protocol)
    cfg=config_from_yaml(ROOT/'configs/model_clip_b16_qwen05.yaml'); cfg.device='cuda'
    model=MiniVLM(cfg); model.projector.load_state_dict(initial)
    assert all(not p.requires_grad for p in model.vision.parameters())
    assert all(not p.requires_grad for p in model.llm.parameters())
    collator=MultipleImageCollator(processor=model.processor,tokenizer=model.tokenizer,max_length=cfg.max_seq_len,num_image_token=model.num_image_token)
    tcfg=TrainingConfig(device='cuda',batch_size=2,cache_features_on_gpu=False,output_dir=str(OUT/'baseline'))
    baseline=Trainer(model,collator,[],val+flip_val,load_one,tcfg)
    print('Precompute train/validation only',flush=True)
    baseline.precompute_features(train+flip_train+val+flip_val+guard)
    cache=baseline.feature_cache
    rms=[(cache[a['image']]-cache[b['image']]).square().mean().sqrt().item() for a,b in zip(train,flip_train)]
    assert min(rms)>1e-4
    # Cached features let us move the frozen vision encoder off GPU during training.
    model.vision.cpu(); torch.cuda.empty_cache()
    base_loss=baseline.validate()['val_loss']
    print('Evaluate baseline',flush=True)
    report={'protocol':protocol,'feature_rms_min':min(rms),'baseline_val_loss':base_loss,'baseline':evaluate(baseline,val,flip_val,guard)}
    write(OUT/'results.partial.json',report)
    print('baseline',report['baseline']['summary'],flush=True)
    for name, second in [('original_control',train),('mirror',flip_train)]:
        model.projector.load_state_dict(initial)
        rows=[r for pair in zip(train,second) for r in pair]
        conf=TrainingConfig(device='cuda',batch_size=2,grad_accum_steps=4,max_steps=64,warmup_steps=8,
            projector_lr=1e-4,eval_every=32,log_every=8,cache_features_on_gpu=False,output_dir=str(OUT/name))
        trainer=Trainer(model,collator,rows,val+flip_val,load_one,conf); trainer.feature_cache=cache
        trainer.best_val_loss=base_loss; trainer.best_step=0
        trainer.save_checkpoint(trainer.out_dir/'checkpoint_best.pt')
        print('TRAIN',name,flush=True)
        trainer.train(log_fn=lambda s:print(s,flush=True))
        best=torch.load(trainer.out_dir/'checkpoint_best.pt',map_location='cpu',weights_only=True)
        model.projector.load_state_dict(best['projector'])
        print('Evaluate selected',name,best['global_step'],flush=True)
        result=evaluate(trainer,val,flip_val,guard)
        result['selected_step']=best['global_step']; result['selected_val_loss']=best['best_val_loss']
        report[name]=result
        write(OUT/'results.partial.json',report)
        print(name,result['summary'],flush=True)
        del trainer; gc.collect(); torch.cuda.empty_cache()
    m=report['mirror']['summary']; c=report['original_control']['summary']; b=report['baseline']['summary']; n=m['n']
    pa=lambda x:(x['original_correct']+x['mirror_correct'])/(2*n)
    report['gate_checks']={'paired_accuracy':pa(m)>=0.6,'both_correct':m['both_correct']/n>=0.25,
        'flipped':m['flipped']/n>=0.25,'beats_control':pa(m)>=pa(c)+0.05,
        'guard':m['guard_correct']>=b['guard_correct']-4,'wrong_image_gap':(m['original_correct']-m['wrong_image_correct'])/n>=0.1}
    report['gate_pass']=all(report['gate_checks'].values())
    assert sha256(ckpt)==source_hash
    report['source_checkpoint_unchanged']=True
    write(OUT/'results.json',report)
    print('DONE',report['gate_checks'],flush=True)

if __name__=='__main__': main()
