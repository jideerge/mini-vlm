"""Training-only 32-pair memorization diagnostic; never evaluates val/test."""
import random, time, json
from scripts.train_mirror_pilot import write
from scripts.evaluate_spatial_flip import (ROOT, mirrored_record, load_one, swap_direction, sha256,
    torch, MiniVLM, config_from_yaml, MultipleImageCollator, Trainer, TrainingConfig, load_records, split_records)
OUT=ROOT/'outputs/b16_mirror_overfit32'

def score(trainer, original, mirror):
    a=trainer.evaluate_generation(original,detail_limit=None,max_new_tokens=24,micro_batch=2)
    b=trainer.evaluate_generation(mirror,detail_limit=None,max_new_tokens=24,micro_batch=2)
    pairs=[]
    for x,y in zip(a['details'],b['details']):
        px,py=x['content']['predicted'],y['content']['predicted']
        pairs.append({'id':x['id'],'original':x,'mirror':y,'flipped':px in ('左边','右边') and py==swap_direction(px)})
    s={'n_pairs':len(pairs),'original_correct':sum(x['original']['content']['correct'] for x in pairs),
       'mirror_correct':sum(x['mirror']['content']['correct'] for x in pairs),
       'both_correct':sum(x['original']['content']['correct'] and x['mirror']['content']['correct'] for x in pairs),
       'flipped':sum(x['flipped'] for x in pairs)}
    return {'summary':s,'pairs':pairs}

def main():
    torch.set_num_threads(4); torch.manual_seed(42)
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite an existing experiment'
    ckpt=ROOT/'outputs/b16_spatial_clean_cloud/checkpoint_best.pt'; original_hash=sha256(ckpt)
    parts=split_records(load_records(ROOT/'data/processed/spatial_clean_9333.jsonl'),ROOT/'data/processed/splits.json')
    records=[r for r in parts['train'] if r['task']=='spatial']
    random.Random(42).shuffle(records); records=records[:32]
    mirrors=[mirrored_record(r) for r in records]
    assert len({r['image'] for r in records})==32
    assert not {r['image'] for r in records}&{r['image'] for p in ['val','test'] for r in parts[p]}
    rows=[r for pair in zip(records,mirrors) for r in pair]
    for a,b in zip(records,mirrors):
        assert a['question']==b['question'] and a['answer']==swap_direction(b['answer'])
    protocol={'source_sha256':original_hash,'seed':42,'train_ids':[r['id'] for r in records],
        'scope':'Training-only memorization diagnostic, no val/test images or scores. Only projector is trained.',
        'max_steps':384,'lr':1e-3,'warmup':8,'batch':2,'accum':4,'eval_every':64,
        'pair_sampling':'Each microbatch contains one original and its actual mirror; shuffle 32 pairs every epoch. 8 updates/epoch.',
        'stop':'Stop at a scheduled evaluation if all 32 pairs are simultaneously correct; otherwise stop after 384 steps. Near-fit threshold 31/32 both correct.',
        'selection':'Best training both-correct, then total correct, then lowest training loss; explicitly not validation selection.'}
    write(OUT/'protocol.json',protocol)
    cfg=config_from_yaml(ROOT/'configs/model_clip_b16_qwen05.yaml'); cfg.device='cuda'
    model=MiniVLM(cfg)
    payload=torch.load(ckpt,map_location='cpu',weights_only=True)
    model.projector.load_state_dict(payload['projector'])
    assert all(not p.requires_grad for p in model.vision.parameters())
    assert all(not p.requires_grad for p in model.llm.parameters())
    assert all(n.startswith('projector.') for n,p in model.named_parameters() if p.requires_grad)
    collator=MultipleImageCollator(processor=model.processor,tokenizer=model.tokenizer,max_length=cfg.max_seq_len,num_image_token=model.num_image_token)
    conf=TrainingConfig(device='cuda',batch_size=2,grad_accum_steps=4,max_steps=384,warmup_steps=8,projector_lr=1e-3,
        cache_features_on_gpu=False,output_dir=str(OUT))
    # Trainer.validate is deliberately called on training rows here; reported as training loss.
    t=Trainer(model,collator,rows,rows,load_one,conf)
    t.precompute_features(rows)
    feature_rms=[(t.feature_cache[a['image']]-t.feature_cache[b['image']]).square().mean().sqrt().item() for a,b in zip(records,mirrors)]
    assert min(feature_rms)>1e-4
    model.vision.cpu(); torch.cuda.empty_cache()
    log=(OUT/'run.log').open('w',encoding='utf-8')
    def emit(s):
        print(s,flush=True); log.write(s+'\n'); log.flush()
    history=[]; evals=[]; best_key=None; best_step=None
    def assess(step):
        nonlocal best_key,best_step
        v=t.validate(); result=score(t,records,mirrors)
        result.update({'step':step,'train_loss':v['val_loss'],'train_token_accuracy':v['val_tok_acc']})
        evals.append(result); s=result['summary']
        key=(s['both_correct'],s['original_correct']+s['mirror_correct'],-v['val_loss'])
        if best_key is None or key>best_key:
            best_key=key; best_step=step
            t.save_checkpoint(OUT/'checkpoint_best_training.pt',extra={'diagnostic':'train_only','training_score':s})
        write(OUT/'progress.json',{'protocol':protocol,'evaluations':evals,'history':history,'best_step':best_step})
        emit('EVAL '+json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=True))
        return s['both_correct']==32
    assess(0)
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    initial={k:v.detach().cpu().clone() for k,v in model.projector.state_dict().items()}
    first_gradient=None; epoch=0; reached=False
    while t.global_step<384 and not reached:
        epoch+=1; order=torch.randperm(32).tolist()
        for start_pair in range(0,32,4):
            model.train(); model.llm.eval(); model.vision.eval()
            t.optimizer.zero_grad(set_to_none=True); loss_sum=0
            for i in order[start_pair:start_pair+4]:
                b,vf=t.make_batch([records[i],mirrors[i]])
                out=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],labels=b['labels'],visual_features=vf)
                assert torch.isfinite(out['loss']), 'Nonfinite loss'
                (out['loss']/4).backward(); loss_sum+=out['loss'].item()/4
            norm=torch.nn.utils.clip_grad_norm_(model.projector.parameters(),1.0)
            assert torch.isfinite(norm) and norm>0, 'Invalid projector gradient'
            if first_gradient is None:first_gradient=float(norm)
            t.optimizer.step(); t.scheduler.step(); t.global_step+=1
            if t.global_step%16==0:
                entry={'step':t.global_step,'epoch':epoch,'mean_micro_loss':loss_sum,'grad_norm':float(norm),'lr':t.scheduler.get_last_lr()[0]}
                history.append(entry); emit('TRAIN '+json.dumps(entry))
            if t.global_step%64==0:
                reached=assess(t.global_step)
                if reached:break
    t.save_checkpoint(OUT/'checkpoint_last.pt',extra={'diagnostic':'train_only'})
    delta=sum((v.detach().cpu()-initial[k]).square().sum().item() for k,v in model.projector.state_dict().items())**0.5
    assert delta>0 and sha256(ckpt)==original_hash
    best=next(x for x in evals if x['step']==best_step)
    report={'protocol':protocol,'evaluations':evals,'history':history,'best_step':best_step,'best_summary':best['summary'],
       'near_fit_pass':best['summary']['both_correct']>=31,'perfect_fit':best['summary']['both_correct']==32,
       'completed_steps':t.global_step,'epochs':epoch,'elapsed_training_and_periodic_eval_s':time.perf_counter()-start,
       'peak_gpu_mb':torch.cuda.max_memory_allocated()/2**20,'first_gradient_norm':first_gradient,
       'projector_parameter_delta_l2':delta,'feature_rms_min':min(feature_rms),'source_checkpoint_unchanged':True}
    write(OUT/'results.json',report); emit('DONE '+json.dumps({k:v for k,v in report.items() if k not in ['protocol','evaluations','history']},ensure_ascii=True));log.close()

if __name__=='__main__':main()
