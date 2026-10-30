#!/usr/bin/env python3
"""Audits KORE B HWP-derived text against all 777 knowledge rows.
All HWP files read by glob('*.hwp'); excludes unrelated PDF uploads.
Normalization discards editorial brackets/amendment citations and whitespace,
not substantive article wording. Does not establish website/human verification.
"""
from pathlib import Path
import csv, re, html, json, unicodedata, hashlib, collections, zipfile, shutil, sys
ROOT = Path(__file__).resolve().parent
HWP_DIR=Path('/mnt/data')
RAW=ROOT/'raw'; DATA=ROOT/'data'; DOC=ROOT/'docs'/'research';DOC.mkdir(parents=True,exist_ok=True)

def readcsv(p):
 with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,header):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)

def clean(s):
 s=html.unescape(unicodedata.normalize('NFC',s))
 # Official HWP uses <개정 ...>, <2020. 2. 4.> as editorial annotations.
 s=re.sub(r'<[^<>\n]{0,200}>','',s)
 # [본조신설], [제39조에서 이동] and [시행일: ...] are editorial metadata.
 # Line-aware to avoid corrupting text if an HWP extraction drops a closing ].
 s=re.sub(r'\[(?:본조|전문|제목|종전|제\d+조|시행일:)[^\n]*','',s)
 return re.sub(r'\s+','',s)

rows=readcsv(DATA/'knowledge.csv'); logs=readcsv(DATA/'source_log.csv')
rawfiles={p.name.split('_20')[0]:p for p in RAW.glob('*_본문_단일판본.txt')}
assert len(rawfiles)==10 and len(rows)==len(logs)==777
byid={r['id']:r for r in rows}
assert len(byid)==777 and set(byid)=={x['id'] for x in logs}
raw_norm={k:clean(p.read_text(encoding='utf-8')) for k,p in rawfiles.items()}
assert all(x['verified_by']=='' and x['license_checked']=='no' for x in logs)

# Independently compare HWP XML with saved extracted original text (does not open PDF).
import xml.etree.ElementTree as ET
hwpfiles=list(HWP_DIR.glob('*.hwp'))
assert len(hwpfiles)==10, 'Need original 10 HWP inputs in /mnt/data'
versions=readcsv(DOC/'법령10종_판본목록.csv')
file_issues=[]
for v in versions:
 matched=[f for f in hwpfiles if f.name==v['source_file']]
 if len(matched)!=1:
  # HWP filenames may be stored with differing Unicode normal forms.
  matched=[f for f in hwpfiles if unicodedata.normalize('NFC',f.name)==unicodedata.normalize('NFC',v['source_file'])]
 assert len(matched)==1,v['document']
 f=matched[0]
 assert hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256'],v['document']
 ps=[''.join(p.itertext()).replace('\xa0',' ').strip() for p in ET.parse(f).getroot().iter('P')]
 ps=[re.sub(r'[ \t]+',' ',s).strip() for s in ps if s]
 first=next(i for i,s in enumerate(ps) if re.match(r'^제1장(?:\s|$)',s) or s.startswith('제1조('))
 meta=next(x for x in ps[:12] if '[시행' in x and ('제'+v['law_number']+'호') in x)
 stem=next(x['id'].rsplit('_',1)[0] for x in rows if x['title'].startswith(v['document']+' 제'))
 extracted=RAW/f'{stem}_{v["effective_date"]}_추출원문_부칙포함.txt'
 expected=meta+'\n\n'+'\n'.join(ps[first:])+'\n'
 assert extracted.read_text(encoding='utf-8')==expected,v['document']
 file_issues.append({'document':v['document'],'version':v['effective_date'],'hwp_sha256_matches':True,'hwp_xml_to_saved_raw_exact':True,'hwp_filename':f.name})

# Check full-row normalized membership and article-by-article coverage.
per_row=[]
for r in rows:
 pre=r['id'].rsplit('_',1)[0];t=clean(r['text']); source=raw_norm[pre]
 hits=source.count(t)
 per_row.append({'id':r['id'],'law':r['title'].split(' 제')[0], 'title':r['title'], 'norm_chars':len(t),'full_text_found_in_uploaded_hwp':('yes' if hits else 'no'), 'number_of_matches':hits,'issues':('|'.join(z for test,z in [('&nbsp;' in r['text'],'HTML_ENTITY'),('[시행일:' in r['text'],'FUTURE_EFFECTIVE_MARK'),(bool(re.search(r'\[(?:제\d+조|본조|전문|제목)',r['text'])),'EDITORIAL_NOTE')] if test))})
assert all(x['full_text_found_in_uploaded_hwp']=='yes' for x in per_row)
writecsv(DOC/'AI_정밀검증_행별.csv',per_row,list(per_row[0]))

r_by=collections.defaultdict(list)
for r in rows:r_by[r['id'].rsplit('_',1)[0]].append(r)
summary=[];deleted=[];article_issues=[]
for pre,rf in sorted(rawfiles.items()):
 s=rf.read_text(encoding='utf-8')
 starts=list(re.finditer(r'(?m)^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*\([^\n]*?\)|\s*삭제)',s))
 articles=[]
 for i,m in enumerate(starts):
  chunk=s[m.start():starts[i+1].start() if i+1<len(starts) else len(s)]
  chunk=re.sub(r'(?m)^제\s*\d+\s*(?:장|절)[^\n]*$', '',chunk)
  t=clean(chunk)
  candidates=[r for r in r_by[pre] if clean(r['text']) in t]
  coverage=bytearray(len(t))
  for r in candidates:
   v=clean(r['text']);start=t.find(v)
   if start>=0:coverage[start:start+len(v)]=b'\1'*len(v)
  covered=sum(coverage)
  articles.append((m.group(0),len(t),covered,len(candidates)))
  if covered!=len(t):
   if re.match(r'^제\s*\d+\s*조(?:\s*의\s*\d+)?\s*삭제',m.group(0)) and not candidates:
    deleted.append({'law':next(r['title'].split(' 제')[0] for r in r_by[pre]),'article':m.group(0),'reason':'deleted-only article intentionally not indexed'})
   else:
    article_issues.append({'prefix':pre,'article':m.group(0),'chars':len(t),'covered':covered})
 assert not article_issues,article_issues
 total=sum(a[1] for a in articles)
 covered=sum(a[2] for a in articles)
 summary.append({'law':r_by[pre][0]['title'].split(' 제')[0],'prefix':pre,'csv_rows':len(r_by[pre]),'source_article_headings':len(articles),'deleted_only_articles':sum(x[3]==0 for x in articles),'substantive_article_coverage':'all','coverage_pct_including_deleted':round(100*covered/total,3),'hwp_sha256_checked':'yes','official_version_metadata_checked':'yes','official_full_text_compared':'no'})
assert len(deleted)==47,(len(deleted),deleted)
writecsv(DOC/'AI_정밀검증_법령별.csv',summary,list(summary[0]))
writecsv(DOC/'AI_정밀검증_삭제조문.csv',deleted,list(deleted[0]))

# User never confirmed research source licenses; preserve all human markers untouched.
issues=[]
for r in rows:
 v=r['text']
 if '&nbsp;' in v:issues.append({'id':r['id'],'type':'HTML_ENTITY','severity':'fixable','action':'replace &nbsp; with whitespace','details':'Text extraction artifact only'})
 if '[시행일:' in v:issues.append({'id':r['id'],'type':'FUTURE_EFFECTIVE','severity':'quarantine','action':'exclude row from evaluation pending A approval','details':v.split('[시행일:')[-1][:110]})
 if re.search(r'\[(?:제\d+조|본조|전문|제목)',v):issues.append({'id':r['id'],'type':'EDITORIAL_NOTE','severity':'cosmetic','action':'strip editorial bracket notes in suggested cleaned export','details':re.findall(r'\[(?:제\d+조|본조|전문|제목)[^\n]*',v)[-1][:120]})
writecsv(DOC/'AI_정밀검증_예외.csv',issues,list(issues[0]))

# Proposal only: baseline /data files remain unchanged; never mark as human verified.
proposal=ROOT/'suggested'/'data'
proposal.mkdir(parents=True,exist_ok=True)
reviewed=[];excluded_ids={'law_pipa_decree_081'}
for r in rows:
 if r['id'] in excluded_ids:continue
 nr=r.copy()
 nr['text']=html.unescape(nr['text'])
 nr['text']=re.sub(r'\[(?:제\d+조|본조|전문|제목)[^\n]*','',nr['text'])
 nr['text']=nr['text'].strip()
 reviewed.append(nr)
logreview=[x for x in logs if x['id'] not in excluded_ids]
assert len(reviewed)==len(logreview)==776
assert {x['id'] for x in reviewed}=={x['id'] for x in logreview}
writecsv(proposal/'knowledge.csv',reviewed,list(rows[0]))
writecsv(proposal/'source_log.csv',logreview,list(logs[0]))

stats={'hwp_verified':len(file_issues),'knowledge_rows':len(rows),'exact_normalized_legal_body_matches':sum(x['full_text_found_in_uploaded_hwp']=='yes' for x in per_row),'source_articles':sum(x['source_article_headings'] for x in summary),'deleted_articles_intentionally_unindexed':len(deleted),'substantive_articles_uncovered':len(article_issues),'proposal_knowledge_rows':len(reviewed),'proposal_excluded_row_ids':sorted(excluded_ids),'issue_counts':dict(collections.Counter(i['type'] for i in issues)),'id_duplicates':len(rows)-len(byid),'source_log_id_sets_match':set(byid)=={x['id'] for x in logs},'all_human_verified':'no','full_official_web_text_compared':'no'}
(DOC/'AI_정밀검증_결과.json').write_text(json.dumps({'stats':stats,'files':file_issues,'per_law':summary},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
