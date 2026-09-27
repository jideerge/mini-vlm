"""Matched local ordinary/6x-direction supervision pilot; validation only."""
import json,random,time,gc
from scripts.train_mirror_pilot import evaluate,write
from scripts.evaluate_spatial_flip import (ROOT,mirrored_record,load_one,sha256,torch,MiniVLM,config_from_yaml,MultipleImageCollator,Trainer,TrainingConfig,load_records,split_records)
from training.direction_loss import direction_weights,weighted_answer_loss
OUT=ROOT/'outputs/b16_direction_weighted_pilot'
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    source=ROOT/'outputs/b16_spatial_clean_cloud/checkpoint_best.pt';source_hash=sha256(source)
    parts=split_records(load_records(ROOT/'data/processed/spatial_clean_9333.jsonl'),ROOT/'data/processed/splits.json')
    prev=json.loads((ROOT/'outputs/b16_mirror_mixed512/protocol.json').read_text(encoding='utf-8'))
    byid={r['id']:r for r in parts['train']}
    spatial=[byid[i] for i in prev['train_ids'][:256]];mirrors=[mirrored_record(r) for r in spatial]
    replay=[]
    for task in ['attribute','counting','existence','listing']:
        replay.extend([byid[i] for i in prev['replay_ids'] if byid[i]['task']==task][:128])
    val=[r for r in parts['val'] if r['task']=='spatial'];flipval=[mirrored_record(r) for r in val];guard=[r for r in parts['val'] if r['task']!='spatial']
    assert len(spatial)==256 and len(replay)==512 and len(val)==62 and len(guard)==399
    assert not {r['image'] for r in spatial+replay}&{r['image'] for r in val+guard}
    baseline=json.loads((ROOT/'outputs/b16_mirror_overfit32_val/results_content_v2.json').read_text(encoding='utf-8'))['baseline']
    protocol={'seed':42,'source_sha256':source_hash,'spatial_ids':[r['id'] for r in spatial],'replay_ids':[r['id'] for r in replay],
      'max_steps':256,'epochs':2,'lr':2e-4,'warmup':16,'batch':2,'accum':4,'direction_factors':[1,6],
      'sampling':'Each update: two original/mirror pairs and four replay examples. Identical per-arm order; 128 updates per epoch.',
      'selection':'Among baseline and final step256, require other-task correct >=313/399, then maximize both-correct spatial pairs, then total paired correctness. Ties prefer baseline.',
      'gate':'Final weighted arm must reach 60% paired accuracy and 25% both-correct, gain >=5pp paired accuracy over matched control, retain >=313/399 other tasks, and beat fixed wrong image >=10pp on originals.',
      'metrics_version':2,'test_evaluated':False,'baseline_source':'outputs/b16_mirror_overfit32_val/results_content_v2.json'}
    write(OUT/'protocol.json',protocol)
    cfg=config_from_yaml(ROOT/'configs/model_clip_b16_qwen05.yaml');cfg.device='cuda';model=MiniVLM(cfg)
    initial=torch.load(source,map_location='cpu',weights_only=True)['projector'];model.projector.load_state_dict(initial)
    assert all(n.startswith('projector.') for n,p in model.named_parameters() if p.requires_grad)
    collator=MultipleImageCollator(processor=model.processor,tokenizer=model.tokenizer,max_length=cfg.max_seq_len,num_image_token=model.num_image_token)
    helper=Trainer(model,collator,[],[],load_one,TrainingConfig(device='cuda',batch_size=2,cache_features_on_gpu=False,output_dir=str(OUT)))
    helper.precompute_features(list({r['image']:r for r in spatial+mirrors+replay+val+flipval+guard}.values()))
    cache=helper.feature_cache;model.vision.cpu();torch.cuda.empty_cache()
    # Actual tokenizer and collator alignment check, including unchanged question masks.
    sample=[spatial[0],mirrors[0]];batch,_=helper.make_batch(sample)
    w=direction_weights(batch,sample,model.tokenizer,6)
    assert (w[batch['labels']==-100]==1).all() and (w>1).sum()>=2
    aligned=[model.tokenizer.decode(batch['labels'][i][w[i]>1].tolist()) for i in range(2)]
    write(OUT/'token_alignment.json',{'weighted_text':aligned,'only_answer_positions':True})
    report={'protocol':protocol,'baseline':baseline,'token_alignment':aligned}
    for name,factor in [('ordinary',1.),('direction6',6.)]:
        model.projector.load_state_dict(initial);torch.manual_seed(42)
        conf=TrainingConfig(device='cuda',batch_size=2,grad_accum_steps=4,max_steps=256,warmup_steps=16,projector_lr=2e-4,cache_features_on_gpu=False,output_dir=str(OUT/name))
        t=Trainer(model,collator,[],[],load_one,conf);t.feature_cache=cache
        rng=random.Random(42);history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        print('TRAIN',name,flush=True)
        with (OUT/name/'run.log').open('w',encoding='utf-8') as log:
            for epoch in range(2):
                si=list(range(256));ri=list(range(512));rng.shuffle(si);rng.shuffle(ri)
                for k in range(128):
                    model.train();model.llm.eval();model.vision.eval();t.optimizer.zero_grad(set_to_none=True)
                    micro=[[spatial[i],mirrors[i]] for i in si[2*k:2*k+2]]
                    rr=[replay[i] for i in ri[4*k:4*k+4]];micro.extend([rr[:2],rr[2:]])
                    average=0.
                    for rows in micro:
                        batch,vf=t.make_batch(rows)
                        weights=direction_weights(batch,rows,model.tokenizer,factor)
                        out=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],visual_features=vf,return_logits=True)
                        loss=weighted_answer_loss(out['logits'],batch['labels'],weights)
                        assert torch.isfinite(loss);(loss/4).backward();average+=loss.item()/4
                        del out,loss
                    norm=torch.nn.utils.clip_grad_norm_(model.projector.parameters(),1.0);assert torch.isfinite(norm)
                    t.optimizer.step();t.scheduler.step();t.global_step+=1
                    if t.global_step%32==0:
                        entry={'step':t.global_step,'epoch':epoch+1,'loss':average,'grad_norm':float(norm)};history.append(entry)
                        line=json.dumps(entry);print(name,line,flush=True);log.write(line+'\n');log.flush()
        t.save_checkpoint(OUT/name/'checkpoint_final.pt',extra={'direction_factor':factor})
        training={'elapsed_s':time.perf_counter()-start,'peak_mb':torch.cuda.max_memory_allocated()/2**20,'history':history}
        write(OUT/name/'training.json',training)
        print('EVALUATE',name,flush=True)
        result=evaluate(t,val,flipval,guard);m=result['summary'];b=baseline['summary']
        rank=lambda x:(x['both_correct'],x['original_correct']+x['mirror_correct'])
        select_final=m['guard_correct']>=313 and rank(m)>rank(b)
        report[name]={'evaluation':result,'training':training,'selected_step':256 if select_final else 0}
        write(OUT/'results.partial.json',report);print(name,m,'selected',report[name]['selected_step'],flush=True)
        del t;gc.collect();torch.cuda.empty_cache()
    m=report['direction6']['evaluation']['summary'];c=report['ordinary']['evaluation']['summary']
    pa=lambda x:(x['original_correct']+x['mirror_correct'])/124
    checks={'paired_accuracy':pa(m)>=.6,'both_correct':m['both_correct']/62>=.25,'beats_control':pa(m)>=pa(c)+.05,'other_tasks':m['guard_correct']>=313,'wrong_image_gap':(m['original_correct']-m['wrong_image_correct'])/62>=.1}
    assert sha256(source)==source_hash
    report.update(gate_checks=checks,gate_pass=all(checks.values()),source_checkpoint_unchanged=True)
    write(OUT/'results.json',report);print('DONE',checks,flush=True)
if __name__=='__main__':main()
