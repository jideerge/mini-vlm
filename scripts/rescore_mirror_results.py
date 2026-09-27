"""Re-score saved outputs without inference; preserve original report."""
import copy,json
from pathlib import Path
from collections import defaultdict
from evaluation.metrics import score_content,summarize_content,CONTENT_METRICS_VERSION
ROOT=Path(__file__).resolve().parents[1]
def rescore(result):
    r=copy.deepcopy(result)
    def detail(d):
        d['content']=score_content({'task':d['task'],'answer':d['reference'],'meta':d['meta']},d['generated'])
    for p in r['pairs']:
        detail(p['original']);detail(p['mirror'])
    for key in ['guard','wrong_image']:
        groups=defaultdict(list)
        for d in r[key]['details']:detail(d);groups[d['task']].append(d['content'])
        r[key]['content_per_task']={k:summarize_content(v) for k,v in groups.items()}
        r[key]['content']=summarize_content([d['content'] for d in r[key]['details']])
    s=r['summary']
    s['original_correct']=sum(p['original']['content']['correct'] for p in r['pairs'])
    s['mirror_correct']=sum(p['mirror']['content']['correct'] for p in r['pairs'])
    s['both_correct']=sum(p['original']['content']['correct'] and p['mirror']['content']['correct'] for p in r['pairs'])
    s['guard_correct']=sum(d['content']['correct'] for d in r['guard']['details'])
    s['wrong_image_correct']=sum(d['content']['correct'] for d in r['wrong_image']['details'])
    return r
if __name__=='__main__':
    source=ROOT/'outputs/b16_mirror_overfit32_val/results.json'
    old=json.loads(source.read_text(encoding='utf-8'))
    r={'content_metrics_version':CONTENT_METRICS_VERSION,'source':str(source),'old_summaries':{k:old[k]['summary'] for k in ['baseline','overfit192']}}
    for k in ['baseline','overfit192']:
        r[k]=rescore(old[k]);print(k,r[k]['summary'],r[k]['guard']['content']['status_counts'])
    (source.parent/'results_content_v2.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
