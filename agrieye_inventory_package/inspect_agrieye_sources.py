#!/usr/bin/env python3
"""Preserve exports, inventory polygons, and compare source membership. Needs PyYAML."""
import argparse,collections,csv,hashlib,io,json,math,re,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import yaml
IMAGE_EXT={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(name): return ' '.join(str(name).upper().split())
def source_key(n):
    # Candidate ancestry only: Roboflow hash removal does not prove pixel identity.
    s=PurePosixPath(n).name.split('.rf.')[0].lower()
    return re.sub(r'(?:_jpg|_jpeg|_png|\.jpg|\.jpeg|\.png)+$','',s)
def csv_write(p,rows,fields):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,obj): p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def stage(src,dst):
    if dst.exists(): raise ValueError(f'Refusing to overwrite staged source: {dst}')
    dst.mkdir(parents=True)
    if src.is_file():
        shutil.copy2(src,dst/src.name)
        with zipfile.ZipFile(src) as z:
            seen={}
            for i in z.infolist():
                n=PurePosixPath(i.filename)
                if n.is_absolute() or '..' in n.parts: raise ValueError('Unsafe ZIP path')
                if i.is_dir():continue
                b=z.read(i)
                if i.filename in seen:
                    if seen[i.filename]!=sha(b):raise ValueError('Conflicting duplicate ZIP entry')
                    continue
                seen[i.filename]=sha(b);p=dst/'export'/str(n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        return dst/'export'
    shutil.copytree(src,dst/'export');return dst/'export'
def tree(project,depth=3):
    lines=[str(project)]
    if not project.is_dir():return lines+['[project not accessible in this runtime]']
    def walk(p,level):
        if level>depth:return
        for c in sorted(p.iterdir(),key=lambda q:q.name):
            if c.name.startswith('.'):continue
            prefix='  '*level
            if c.is_dir():
                count=sum(1 for x in c.iterdir() if x.is_file())
                lines.append(f'{prefix}{c.name}/ ({count} direct files)')
                # Summarize image/label/run bulk without enumerating thousands of files.
                if c.name not in {'images','labels','runs','weights','__pycache__'}:walk(c,level+1)
            elif level<=1:lines.append(prefix+c.name)
    walk(project,1);return lines
class Source:
    def __init__(self,p):
        self.p=Path(p);self.z=zipfile.ZipFile(p) if self.p.is_file() else None
        self.entries=self.z.infolist() if self.z else []
        self.files={i.filename:i for i in self.entries if not i.is_dir()} if self.z else {q.relative_to(self.p).as_posix():q for q in self.p.rglob('*') if q.is_file()}
    def read(self,n):return self.z.read(self.files[n]) if self.z else self.files[n].read_bytes()
    def close(self):
        if self.z:self.z.close()
def scan(path,tag):
    s=Source(path)
    try:
        cfg=yaml.safe_load(s.read('data.yaml'));ns=cfg['names']
        names={int(k):v for k,v in ns.items()} if isinstance(ns,dict) else dict(enumerate(ns))
        rows=[];issues=[];matched=set();counts=collections.Counter();splitcounts=collections.Counter();instance=collections.Counter()
        for n in sorted(s.files):
            parts=PurePosixPath(n).parts
            if len(parts)<3 or parts[1]!='images' or PurePosixPath(n).suffix.lower() not in IMAGE_EXT:continue
            split=parts[0];lab=n.replace('/images/','/labels/',1).rsplit('.',1)[0]+'.txt';ids=[];polys=[];invalid=0;boxes=0;total=0
            b=s.read(n);lb=s.read(lab) if lab in s.files else None
            if lb is not None:
                matched.add(lab)
                for j,line in enumerate(lb.decode('utf-8').splitlines(),1):
                    if not line.strip():continue
                    total+=1
                    try:
                        tok=line.split();f=float(tok[0]);assert f.is_integer() and int(f) in names
                        cid=int(f);v=list(map(float,tok[1:]));assert all(math.isfinite(x) and 0<=x<=1 for x in v)
                        if len(v)==4:boxes+=1
                        else:
                            assert len(v)>=6 and len(v)%2==0
                            points=list(zip(v[::2],v[1::2]));assert abs(sum(x*y2-x2*y for (x,y),(x2,y2) in zip(points,points[1:]+points[:1])))>0
                            polys.append([canonical(names[cid]),v]);instance[(split,cid)]+=1
                        ids.append(cid)
                    except (AssertionError,ValueError):invalid+=1;issues.append({'file':lab,'line':j,'issue':'invalid class/shape/coordinates/area'})
            else:issues.append({'file':n,'line':'','issue':'missing label'})
            for cid in set(ids):counts[(split,cid)]+=1
            splitcounts[split]+=1
            rows.append(dict(dataset=tag,split=split,image=n,image_sha256=sha(b),label=lab,label_sha256=sha(lb) if lb is not None else '',label_status='missing' if lb is None else 'empty' if not total else 'nonempty',annotation_rows=total,observed_ids=';'.join(map(str,sorted(set(ids)))),polygons=len(polys),boxes=boxes,invalid_rows=invalid,source_key=source_key(n),tungro_named='tungro' in PurePosixPath(n).name.lower(),semantic_polygon_hash=sha(json.dumps(sorted(polys,key=lambda x:json.dumps(x)),separators=(',',':')).encode()) if not invalid and not boxes and lb is not None else ''))
        orphans=sorted(n for n in s.files if '/labels/' in n and n.endswith('.txt') and n not in matched)
        summary=dict(dataset=tag,source=str(Path(path).resolve()),names=names,images=len(rows),splits=dict(splitcounts),empty_labels=sum(r['label_status']=='empty' for r in rows),missing_labels=sum(r['label_status']=='missing' for r in rows),polygon_rows=sum(r['polygons'] for r in rows),box_rows=sum(r['boxes'] for r in rows),invalid_rows=sum(r['invalid_rows'] for r in rows),tungro_named_images=sum(r['tungro_named'] for r in rows),orphan_labels=orphans,duplicate_zip_paths={k:v for k,v in collections.Counter(i.filename for i in s.entries).items() if v>1},yaml=cfg)
        classrows=[dict(dataset=tag,split=sp,class_id=cid,class_name=name,images_with_class=counts[(sp,cid)],polygon_instances=instance[(sp,cid)]) for sp in [*sorted(splitcounts),'ALL'] for cid,name in names.items()]
        for r in classrows:
            if r['split']=='ALL':r['images_with_class']=sum(v for (sp,cid),v in counts.items() if cid==r['class_id']);r['polygon_instances']=sum(v for (sp,cid),v in instance.items() if cid==r['class_id'])
        return summary,rows,classrows,issues
    finally:s.close()
def from_audit(path):
    with zipfile.ZipFile(path) as z:
        summary=json.loads(z.read('dataset_1/summary.json'));cfg=summary['yaml_content'];ns=cfg['names'];names={int(k):v for k,v in ns.items()} if isinstance(ns,dict) else dict(enumerate(ns))
        rows=list(csv.DictReader(io.StringIO(z.read('dataset_1/image_manifest.csv').decode())))
        oldclasses=list(csv.DictReader(io.StringIO(z.read('dataset_1/class_counts.csv').decode())))
    for r in rows:
        r.update(dataset='final',source_key=source_key(r['image']),tungro_named='tungro' in r['image'].lower(),semantic_polygon_hash='')
        for key in ['annotation_rows','polygons','boxes']:r[key]=int(r[key])
        r['invalid_rows']=int(r['annotation_rows'])-int(r['valid_rows'])
    classrows=[dict(dataset='final',split=r['split'],class_id=int(r['class_id']),class_name=r['class_name'],images_with_class=int(r['images_with_valid_geometry']),polygon_instances=int(r['valid_geometry_rows'])) for r in oldclasses]
    for cid,name in names.items():
        rr=[r for r in classrows if r['class_id']==cid];classrows.append(dict(dataset='final',split='ALL',class_id=cid,class_name=name,images_with_class=sum(r['images_with_class'] for r in rr),polygon_instances=sum(r['polygon_instances'] for r in rr)))
    return dict(dataset='final',source=summary['dataset_root'],names=names,images=len(rows),splits=dict(collections.Counter(r['split'] for r in rows)),yaml=cfg,evidence='Previous uploaded manifest; final image/label bytes unavailable here, so semantic polygon matching and fresh verification are unavailable.'),rows,classrows,[]
def compare(raw,final,out):
    # Candidate matches can be one-to-many; all edges retained rather than forced into pairs.
    byhash=collections.defaultdict(list);bykey=collections.defaultdict(list)
    for r in raw:byhash[r['image_sha256']].append(r);bykey[r['source_key']].append(r)
    edges=[];seen=set();finalonly=[]
    for f in final:
        candidates=byhash[f['image_sha256']];kind='identical_bytes'
        if not candidates:candidates=bykey[f['source_key']];kind='same_source_key_different_bytes_candidate'
        if not candidates:finalonly.append(f);continue
        for r in candidates:
            seen.add((r['dataset'],r['image']))
            sem='not_compared'
            if kind=='identical_bytes' and r.get('semantic_polygon_hash') and f.get('semantic_polygon_hash'):sem='equal' if r['semantic_polygon_hash']==f['semantic_polygon_hash'] else 'different'
            edges.append(dict(raw_dataset=r['dataset'],raw_split=r['split'],raw_image=r['image'],final_split=f['split'],final_image=f['image'],match_kind=kind,raw_ids=r['observed_ids'],final_ids=f['observed_ids'],label_bytes_equal=r['label_sha256']==f['label_sha256'],semantic_polygons=sem))
    rawonly=[r for r in raw if (r['dataset'],r['image']) not in seen]
    csv_write(out/'matches.csv',edges,['raw_dataset','raw_split','raw_image','final_split','final_image','match_kind','raw_ids','final_ids','label_bytes_equal','semantic_polygons'])
    fields=['dataset','split','image','image_sha256','label','label_status','observed_ids','source_key','tungro_named']
    csv_write(out/'raw_without_final_match.csv',rawonly,fields);csv_write(out/'final_without_raw_match.csv',finalonly,fields)
    groups=collections.defaultdict(list)
    for r in raw+final:groups[r['image_sha256']].append({k:r[k] for k in ['dataset','split','image']})
    dump(out/'exact_duplicate_image_groups.json',{k:v for k,v in groups.items() if len(v)>1})
    return dict(raw_images=len(raw),final_images=len(final),matched_final_images=len(final)-len(finalonly),raw_without_any_final_match=len(rawonly),final_without_any_raw_match=len(finalonly),match_edges=dict(collections.Counter(e['match_kind'] for e in edges)),note='Source-key matches are ancestry candidates, not proof of the augmentation or annotation geometry. Multiple final tiles/augmentations may map to one raw image. No files or class IDs were changed.')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--disease',type=Path,required=True);ap.add_argument('--healthy',type=Path,required=True)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--final',type=Path);group.add_argument('--final-audit',type=Path,help='Use earlier checker ZIP when final image bytes are unavailable')
    ap.add_argument('--stage',action='store_true',help='Copy all source exports to fresh project data/raw_author snapshot; no training configuration changes');ap.add_argument('--output',type=Path)
    args=ap.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=args.output or args.project/'reports'/('source_inventory_'+stamp)
    if args.stage and not args.project.is_dir():raise FileNotFoundError(f'Project directory does not exist: {args.project}')
    if out.exists():raise ValueError(f'Use a fresh output folder: {out}')
    out.mkdir(parents=True);(out/'directory_tree_before.txt').write_text('\n'.join(tree(args.project)))
    paths={'raw_disease':args.disease,'raw_healthy':args.healthy}
    if args.final:paths['final']=args.final
    for p in paths.values():
        if not p.exists():raise FileNotFoundError(p)
    if args.stage:
        for tag,p in list(paths.items()):paths[tag]=stage(p,args.project/'data'/'raw_author'/stamp/tag)
    summaries=[];allrows=[];classes=[];issues=[]
    for tag,p in paths.items():
        s,r,c,i=scan(p,tag);summaries.append(s);allrows.extend(r);classes.extend(c);issues.extend(i)
    if args.final_audit:
        s,r,c,i=from_audit(args.final_audit);summaries.append(s);allrows.extend(r);classes.extend(c)
    dump(out/'dataset_summaries.json',summaries)
    csv_write(out/'class_counts.csv',classes,['dataset','split','class_id','class_name','images_with_class','polygon_instances'])
    csv_write(out/'image_manifest.csv',allrows,list(allrows[0]))
    csv_write(out/'annotation_issues.csv',issues,['file','line','issue'])
    maps=[dict(dataset=s['dataset'],class_id=cid,class_name=name,meaning=canonical(name)) for s in summaries for cid,name in s['names'].items()]
    csv_write(out/'class_id_mapping.csv',maps,['dataset','class_id','class_name','meaning'])
    result=compare([r for r in allrows if r['dataset']!='final'],[r for r in allrows if r['dataset']=='final'],out)
    dump(out/'comparison_summary.json',result)
    dump(out/'invocation.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},staged_paths={k:str(v) for k,v in paths.items()},raw_modified=False,training_config_modified=False))
    (out/'directory_tree_after.txt').write_text('\n'.join(tree(args.project)))
    print(json.dumps(result,indent=2));print(f'REPORTS: {out}')
if __name__=='__main__':main()
