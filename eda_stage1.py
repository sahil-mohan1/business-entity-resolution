# -*- coding: utf-8 -*-
import os, re, sys, warnings, datetime
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
# Ensure stdout can handle non-ASCII output on Windows terminals
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE, 'dataset', 'test')

FILES = {
    'train_source1': os.path.join(TRAIN_DIR, 'train_source1.tsv'),
    'train_source2': os.path.join(TRAIN_DIR, 'train_source2.tsv'),
    'train_source3': os.path.join(TRAIN_DIR, 'train_source3.tsv'),
    'train_gt':      os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'),
    'test_source1':  os.path.join(TEST_DIR,  'test_source1.tsv'),
    'test_source2':  os.path.join(TEST_DIR,  'test_source2.tsv'),
    'test_source3':  os.path.join(TEST_DIR,  'test_source3.tsv'),
}
SAMPLE_ROWS = 50_000
SEED = 42

def read_sample(path, n=SAMPLE_ROWS):
    try:
        return pd.read_csv(path, sep=chr(9), nrows=n, encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=chr(9), nrows=n, encoding='latin-1', on_bad_lines='skip', low_memory=False)

def read_full(path):
    try:
        return pd.read_csv(path, sep=chr(9), encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=chr(9), encoding='latin-1', on_bad_lines='skip', low_memory=False)

def file_size_mb(path): return os.path.getsize(path)/(1024**2)

def row_count(path):
    with open(path,'rb') as f: return sum(1 for _ in f)-1

def jaccard(a,b):
    if not isinstance(a,str) or not isinstance(b,str): return 0.0
    ta=set(a.lower().split()); tb=set(b.lower().split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta&tb)/len(ta|tb)

def missing_pct(df): return (df.isnull().sum()/len(df)*100).round(2)

def fbeta(p,r,b=0.5):
    if p+r==0: return 0.0
    return (1+b*b)*p*r/(b*b*p+r)

def sec(t): return '\n'+'='*70+'\n  '+t+'\n'+'='*70

lines=[]
def p(*a):
    t=' '.join(str(x) for x in a); print(t); lines.append(t)

p('='*70); p('  STAGE 1 EDA REPORT')
p('  Generated: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
p('='*70)

p(sec('1. FILE OVERVIEW'))
total_rows={}
for name,path in FILES.items():
    try:
        nr=row_count(path); total_rows[name]=nr
        p('  %-22s %8.1f MB  %12s rows'%(name,file_size_mb(path),format(nr,',')))
    except Exception as e:
        p('  %-22s ERROR: %s'%(name,e))

p(sec('2. COLUMN NAMES'))
samples={}
for name,path in FILES.items():
    df=read_sample(path); samples[name]=df
    p('\n  ['+name+']  '+str(list(df.columns)))

p(sec('3. MISSING VALUES'))
for name in ['train_source1','train_source2','train_source3','test_source1','test_source2','test_source3']:
    df=samples[name]; miss=missing_pct(df)
    p('\n  ['+name+']')
    for col,pct in miss.items():
        warn='  <-- WARN' if pct>5 else ''
        p('    %-30s %6.2f%%%s'%(col,pct,warn))

p(sec('4. GROUND TRUTH ANALYSIS'))
gt=read_full(FILES['train_gt'])
p('\n  GT columns: '+str(list(gt.columns)))
p('  GT rows: %s'%format(len(gt),','))
gtc=list(gt.columns)
s1c=[c for c in gtc if 'source1' in c.lower()]
s2c=[c for c in gtc if 'source2' in c.lower()]
s3c=[c for c in gtc if 'source3' in c.lower()]
p('  S1 col: %s  S2 col: %s  S3 col: %s'%(s1c,s2c,s3c))
match_counts=pd.Series([],dtype=int); gt_s1u=0
if s1c:
    sc1=s1c[0]; match_counts=gt.groupby(sc1).size(); gt_s1u=len(match_counts)
    p('\n  Unique S1 with >=1 match: %s'%format(gt_s1u,','))
    p('  Total GT rows: %s'%format(len(gt),','))
    dist=match_counts.value_counts().sort_index()
    p('\n  Match-count distribution:')
    for k,v in dist.head(20).items():
        bar='#'*min(int(v/max(dist.values)*30),30)
        p('    %3d match(es) -> %8s  %s'%(k,format(v,','),bar))
    p('\n  Max=%d  Mean=%.3f  Median=%.1f'%(match_counts.max(),match_counts.mean(),match_counts.median()))
s1t=total_rows.get('train_source1',0); sing=s1t-gt_s1u
p('\n  S1 total: %s  matched: %s  singletons: %s (%.1f%%)'%(format(s1t,','),format(gt_s1u,','),format(sing,','),sing/s1t*100 if s1t else 0))

p(sec('5. COUNTRY DISTRIBUTION'))
for name in ['train_source1','train_source2','train_source3']:
    df=samples[name]; cc_l=[c for c in df.columns if 'country' in c.lower()]
    if cc_l:
        cc=cc_l[0]; vc=df[cc].value_counts(dropna=False).head(15)
        p('\n  ['+name+'] sample=%s'%format(len(df),','))
        for val,cnt in vc.items():
            p('    %-30s %7s  (%.1f%%)'%(str(val),format(cnt,','),cnt/len(df)*100))
    else:
        p('\n  ['+name+'] -- no country column')

p(sec('6. NAME AND ADDRESS LENGTH DISTRIBUTIONS'))
for name in ['train_source1','train_source2','train_source3']:
    df=samples[name]
    nc_l=[c for c in df.columns if 'business_name' in c.lower()]
    ac_l=[c for c in df.columns if 'address' in c.lower()]
    p('\n  ['+name+']')
    if nc_l:
        nc=nc_l[0]; lns=df[nc].dropna().str.len(); tok=df[nc].dropna().str.split().str.len()
        p('    name_length  mean:%.1f median:%.0f min:%.0f max:%.0f std:%.1f'%(lns.mean(),lns.median(),lns.min(),lns.max(),lns.std()))
        p('    name_tokens  mean:%.1f median:%.0f max:%.0f'%(tok.mean(),tok.median(),tok.max()))
    if ac_l:
        ac=ac_l[0]; al=df[ac].dropna().str.len(); at=df[ac].dropna().str.split().str.len()
        p('    addr_length  mean:%.1f median:%.0f min:%.0f max:%.0f'%(al.mean(),al.median(),al.min(),al.max()))
        p('    addr_tokens  mean:%.1f median:%.0f'%(at.mean(),at.median()))

p(sec('7. NOISE PROFILE - TOKEN OVERLAP ON MATCHED PAIRS'))
gt_s=gt.sample(min(5000,len(gt)),random_state=SEED) if len(gt)>0 else gt
def build_map(df):
    ic=[c for c in df.columns if 'entity_id' in c.lower()]
    if not ic: return {},None
    return df.set_index(ic[0]).to_dict('index'),ic[0]
s1m,_=build_map(samples['train_source1'])
s2m,_=build_map(samples['train_source2'])
s3m,_=build_map(samples['train_source3'])
if s1c and (s2c or s3c) and s1m:
    nj,aj,ec=[],[],[]
    for _,row in gt_s.iterrows():
        s1i=row[s1c[0]] if s1c else None
        s2i=row[s2c[0]] if s2c else None
        s3i=row[s3c[0]] if s3c else None
        oi=s2i if (s2i and not pd.isna(s2i)) else s3i
        om=s2m if (s2i and not pd.isna(s2i)) else s3m
        if s1i not in s1m or oi not in om: continue
        r1=s1m[s1i]; r2=om[oi]
        nj.append(jaccard(r1.get('business_name','') or '',r2.get('business_name','') or ''))
        aj.append(jaccard(r1.get('business_address','') or '',r2.get('business_address','') or ''))
        c1=str(r1.get('country','')).strip().lower(); c2=str(r2.get('country','')).strip().lower()
        ec.append(int(c1==c2 and c1!=''))
    if nj:
        p('\n  Matched pairs found in sample: %s'%format(len(nj),','))
        p('  Name Jaccard  mean:%.3f median:%.3f <0.2:%.1f%% =1.0:%.1f%%'%(np.mean(nj),np.median(nj),sum(j<0.2 for j in nj)/len(nj)*100,sum(j==1.0 for j in nj)/len(nj)*100))
        p('  Addr Jaccard  mean:%.3f median:%.3f <0.2:%.1f%% =1.0:%.1f%%'%(np.mean(aj),np.median(aj),sum(j<0.2 for j in aj)/len(aj)*100,sum(j==1.0 for j in aj)/len(aj)*100))
        if ec: p('  Country exact match: %.1f%% of pairs'%(sum(ec)/len(ec)*100))
    else:
        p('  No pairs found in 50k sample - IDs not overlapping. Consider SAMPLE_ROWS=200000.')
else:
    p('  Skipped.')

p(sec('8. ENCODING AND CHARACTER ISSUES'))
na_re=re.compile(r'[^\x00-\x7F]')
for name in ['train_source1','train_source2','train_source3']:
    df=samples[name]; nc_l=[c for c in df.columns if 'business_name' in c.lower()]
    if nc_l:
        nc=nc_l[0]; s=df[nc].dropna().astype(str)
        h=s.apply(lambda x:bool(na_re.search(x)))
        p('\n  ['+name+'] non-ASCII: %s/%s (%.1f%%)'%(format(h.sum(),','),format(len(s),','),h.mean()*100))
        for ex in s[h].head(3).tolist(): p('    e.g.: '+ascii(ex[:80]))

p(sec('9. NAIVE BASELINE F0.5'))
p('\n  Strategy A - Predict NOTHING:   F0.5=0.000 (recall=0)')
p('  Strategy B - Predict EVERYTHING: F0.5~0.000 (precision~0)')
p('\n  Hypothetical targets:')
for pr,rc in [(0.95,0.90),(0.90,0.90),(0.80,0.90),(0.95,0.70)]:
    p('    P=%.2f R=%.2f -> F0.5=%.4f'%(pr,rc,fbeta(pr,rc)))

p(sec('10. SAMPLE RECORDS'))
for name in ['train_source1','train_source2','train_source3']:
    df=samples[name]; p('\n  ['+name+']')
    for i,(_,row) in enumerate(df.head(3).iterrows()):
        p('  Row %d:'%i)
        for col,val in row.items(): p('    %s: %s'%(col,str(val)[:100]))
        p('')

rp=os.path.join(BASE,'eda_summary.md')
with open(rp,'w',encoding='utf-8') as ff:
    ff.write('# Stage 1 EDA Summary\n\n`\n'+'\n'.join(lines)+'\n`\n')
p('\n  Report saved: '+rp)
p('='*70)
