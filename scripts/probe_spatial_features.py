"""Exploratory frozen-CLIP spatial probe; no VLM checkpoint is changed."""
import os, sys, json, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor
from models.vision_encoder import VisionEncoder
from datasets.collator import pad_to_square
from datasets.image_paths import resolve_image_path
from scripts.evaluate import load_records, split_records

OUT = ROOT / 'outputs/b16_spatial_probe'

def extract(parts):
    processor = CLIPImageProcessor.from_pretrained(ROOT / 'checkpoints/pretrained/clip-vit-base-patch16', local_files_only=True)
    encoder = VisionEncoder(str(ROOT / 'checkpoints/pretrained/clip-vit-base-patch16')).cuda().eval()
    result = {}
    with torch.inference_mode():
        for part, records in parts.items():
            descriptors, globals_ = [], []
            for start in range(0, len(records), 8):
                images = []
                for r in records[start:start+8]:
                    with Image.open(resolve_image_path(r['image'])) as source:
                        im = source.convert('RGB')
                    images.extend([pad_to_square(im), pad_to_square(ImageOps.mirror(im))])
                pixels = processor(images=images, return_tensors='pt')['pixel_values'].cuda()
                features = encoder(pixels).reshape(-1, 14, 14, 768)
                # Explicit coarse spatial readout, no boxes/coordinates from labels.
                delta = features[:, :, 7:].mean((1,2)) - features[:, :, :7].mean((1,2))
                descriptors.append(delta.cpu())
                globals_.append(features.mean((1,2)).cpu())
                if start % 160 == 0:
                    print(f'extract {part} {start}/{len(records)}', flush=True)
            result[part] = {'spatial': torch.cat(descriptors), 'global': torch.cat(globals_)}
    del encoder
    torch.cuda.empty_cache()
    return result

def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {p:[r for r in rs if r['task']=='spatial'] for p,rs in split_records(load_records(ROOT/'data/processed/spatial_clean_9333.jsonl'), ROOT/'data/processed/splits.json').items()}
    image_sets = {p:{Path(r['image']).name for r in rs} for p,rs in parts.items()}
    assert not (image_sets['train'] & image_sets['val'] or image_sets['train'] & image_sets['test'] or image_sets['val'] & image_sets['test'])
    assert all(len(image_sets[p]) == len(rs) for p, rs in parts.items()), 'Controls require unique images per split'
    assert all(r['meta']['answer'] in ('左边','右边') for rs in parts.values() for r in rs)
    classes = sorted({r['meta'][k] for r in parts['train'] for k in ['subject','other']})
    pairs, labels = {}, {}
    for p, rs in parts.items():
        pairs[p] = torch.tensor([[classes.index(r['meta']['subject']),classes.index(r['meta']['other'])] for r in rs]).repeat_interleave(2,0)
        y = torch.tensor([int(r['meta']['answer']=='右边') for r in rs], dtype=torch.float32)
        labels[p] = torch.stack([y,1-y],1).flatten().cuda()
        pairs[p] = pairs[p].cuda()
    features = extract(parts)
    report = {'seed':42,'counts':{p:len(rs) for p,rs in parts.items()},'image_disjoint':True,'classes':classes,'protocol':'Frozen B16, original+actual mirror train pairs; class-conditioned linear half-image difference probe. No boxes used. Select regularization/epoch on paired validation BCE; evaluate test after selection. Exploratory previously observed test set; not a replacement VLM evaluation.', 'models':{}}
    for kind in ['spatial','global']:
        scale = features['train'][kind].square().mean(0).sqrt().clamp_min(0.01)
        xs = {p:(d[kind]/scale).cuda() for p,d in features.items()}
        best = None
        for reg in [0.001,0.01,0.1]:
            w = torch.nn.Parameter(torch.zeros(len(classes),768,device='cuda'))
            opt = torch.optim.Adam([w],lr=0.01)
            for epoch in range(1,301):
                pred = ((w[pairs['train'][:,0]]-w[pairs['train'][:,1]])*xs['train']).sum(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,labels['train']) + reg*w.square().sum()/len(classes)
                opt.zero_grad(); loss.backward(); opt.step()
                if epoch % 5 == 0:
                    with torch.no_grad():
                        vp = ((w[pairs['val'][:,0]]-w[pairs['val'][:,1]])*xs['val']).sum(1)
                        vl = torch.nn.functional.binary_cross_entropy_with_logits(vp,labels['val']).item()
                        if best is None or vl < best[0]: best=(vl,reg,epoch,w.detach().clone())
        w = best[3]
        def predict(x,p): return (((w[pairs[p][:,0]]-w[pairs[p][:,1]])*x).sum(1)>0).long().cpu()
        def metrics(pred,p):
            y=labels[p].long().cpu(); ok=pred==y
            return {'original_correct':int(ok[::2].sum()),'mirror_correct':int(ok[1::2].sum()),'both_correct':int((ok[::2]&ok[1::2]).sum()),'direction_changed':int((pred[::2]!=pred[1::2]).sum()),'n_pairs':len(y)//2,'paired_accuracy':ok.float().mean().item()}
        pred = predict(xs['test'],'test')
        # Every cyclic shift pairs questions with a different test image, keeping orientation.
        controls=[]
        for shift in range(1,len(parts['test'])):
            idx=torch.arange(len(parts['test'])).roll(shift)
            wrong=xs['test'].reshape(-1,2,768)[idx].reshape(-1,768)
            controls.append(metrics(predict(wrong,'test'),'test')['paired_accuracy'])
        result={'selected_reg':best[1],'selected_epoch':best[2],'val_bce':best[0],'val':metrics(predict(xs['val'],'val'),'val'),'test':metrics(pred,'test'),'wrong_image_paired_accuracy_mean':sum(controls)/len(controls),'wrong_image_range':[min(controls),max(controls)],'details':[{'id':r['id'],'target_right':int(labels['test'][2*i].item()),'original_pred_right':int(pred[2*i]),'mirror_pred_right':int(pred[2*i+1])} for i,r in enumerate(parts['test'])]}
        report['models'][kind]=result
        torch.save({'weights':w.cpu(),'scale':scale,'classes':classes,'kind':kind},OUT/f'{kind}_probe.pt')
        print(kind,json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
    # Question lookup uses train originals only; paired score is necessarily 50%.
    from collections import Counter, defaultdict
    counts=defaultdict(Counter)
    for r in parts['train']: counts[r['question']][r['meta']['answer']]+=1
    correct=0
    for r in parts['test']:
        c=counts[r['question']]; answer='右边' if c['右边']>=c['左边'] else '左边'
        correct+=answer==r['meta']['answer']
    report['question_only']={'original_correct':correct,'mirror_correct':len(parts['test'])-correct,'both_correct':0,'direction_changed':0,'paired_accuracy':0.5}
    report['data_sha256']=hashlib.sha256((ROOT/'data/processed/spatial_clean_9333.jsonl').read_bytes()).hexdigest()
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',report['counts'],report['question_only'],flush=True)

if __name__=='__main__': main()
