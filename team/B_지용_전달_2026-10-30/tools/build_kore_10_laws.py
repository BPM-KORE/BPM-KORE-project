#!/usr/bin/env python3
"""Rebuild KORE B's 10 uploaded HWPML statutes dataset without touching unrelated uploads."""
from pathlib import Path
import csv,zipfile,io,hashlib,xml.etree.ElementTree as ET,re,json,sys,subprocess,collections,shutil
ROOT=Path('/mnt/data'); OUT=ROOT/'KORE_B_법령10종_통합_2026-10-30'
for d in ['raw','data','docs/research','tools']:(OUT/d).mkdir(parents=True,exist_ok=True)
CODE=ROOT/'kore_project_code/kore-project'; sys.path.insert(0,str(CODE))
from kore.ingest import ingest_text,SOURCE_LOG_COLUMNS,split_text
from kore.schemas import KNOWLEDGE_REQUIRED,KNOWLEDGE_OPTIONAL

registry = [
 ('개인정보 보호법','21445','20260911','law_pipa','https://www.law.go.kr/법령/개인정보보호법','core'),
 ('개인정보 보호법 시행령','36671','20260911','law_pipa_decree','https://www.law.go.kr/법령/개인정보보호법시행령','core'),
 ('공공기관의 정보공개에 관한 법률','19408','20231117','law_foia','https://www.law.go.kr/법령/공공기관의정보공개에관한법률','core'),
 ('공공기관의 정보공개에 관한 법률 시행령','35948','20260102','law_foia_decree','https://www.law.go.kr/법령/공공기관의정보공개에관한법률시행령','core'),
 ('공공데이터의 제공 및 이용 활성화에 관한 법률','21392','20260828','law_opendata','https://www.law.go.kr/법령/공공데이터의제공및이용활성화에관한법률','core'),
 ('공공데이터의 제공 및 이용 활성화에 관한 법률 시행령','35947','20260102','law_opendata_decree','https://www.law.go.kr/법령/공공데이터의제공및이용활성화에관한법률시행령','core'),
 ('전자정부법','21394','20260828','law_egov','https://www.law.go.kr/법령/전자정부법','core'),
 ('전자정부법 시행령','36605','20260828','law_egov_decree','https://www.law.go.kr/법령/전자정부법시행령','core'),
 ('정보통신공사업법','21555','20261022','law_ictconstruction','https://www.law.go.kr/법령/정보통신공사업법','distractor'),
 ('공공기관의 운영에 관한 법률','21857','20261002','law_pubinst','https://www.law.go.kr/법령/공공기관의운영에관한법률','distractor'),
]
files=list(ROOT.glob('*.hwp')); assert len(files)==10,len(files)
# Map by law-number uniqueness. Absolutely do not read any .pdf file.
known={}
for f in files:
 matched=[r for r in registry if f'(제{r[1]}호)' in f.name or f'제{r[1]}호' in f.name]
 # Unicode decomposed file names may not match; use NFC
 if not matched:
  import unicodedata
  normal=unicodedata.normalize('NFC',f.name)
  matched=[r for r in registry if f'제{r[1]}호' in normal]
 assert len(matched)==1,(f.name,matched)
 known[matched[0][1]]=f

# Preserve original previously-delivered 264 rows; IDs never change.
BASEZIP=ROOT/'KORE_B_개인정보보호법_시행령_통합_2026-10-30.zip'
with zipfile.ZipFile(BASEZIP) as z:
  prev=list(csv.DictReader(io.TextIOWrapper(z.open('data/knowledge.csv'),encoding='utf-8-sig',newline='')))
  prevlog=list(csv.DictReader(io.TextIOWrapper(z.open('data/source_log.csv'),encoding='utf-8-sig',newline='')))
cols=list(KNOWLEDGE_REQUIRED)+list(KNOWLEDGE_OPTIONAL)
assert len(prev)==264 and len(prevlog)==264
assert {r['id'] for r in prev}=={r['id'] for r in prevlog}
assert set(prev[0])==set(cols)
assert all(not r['verified_by'] and r['license_checked']=='no' for r in prevlog)

keep_future_exclusions={
 '21445':{(28,12),(28,13),(28,14),(28,15)},
 # Another future version of electronic-government article 43-2 appears later; first is current.
}
art_re=re.compile(r'^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*\(([^)]*)\)|\s+삭제)')
metadata_re=re.compile(r'\[시행\s+(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.\].*?\[(법률|대통령령)\s+제(\d+)호')

def paras_from_file(f):
    root=ET.parse(f).getroot()
    res=[''.join(p.itertext()).replace('\xa0',' ').strip() for p in root.iter('P')]
    return [re.sub(r'[ \t]+',' ',s).strip() for s in res if s]

allrows=prev[:];alllogs=prevlog[:];audits=[];verify=[]
for name,no,effect,prefix,url,group in registry:
 f=known[no]; ps=paras_from_file(f)
 meta=next((s for s in ps[:12] if '[시행' in s and f'제{no}호' in s),None)
 assert meta and re.search(r'\[시행\s+\d{4}',meta),(no,ps[:12])
 m=metadata_re.search(meta)
 assert m,meta
 display_date=''.join([m.group(1),m.group(2).zfill(2),m.group(3).zfill(2)])
 assert display_date==effect and m.group(5)==no, (no,display_date,meta)
 start=next(i for i,s in enumerate(ps) if re.match(r'^제1장(?:\s|$)',s) or s.startswith('제1조('))
 stop=next((i for i,s in enumerate(ps[start:],start) if re.match(r'^부칙\s*<',s)),len(ps))
 full=ps[start:]
 main=ps[start:stop]
 heading_indices=[]
 for i,s in enumerate(main):
  a=art_re.match(s)
  if a:
   k=(int(a.group(1)),int(a.group(2) or 0))
   heading_indices.append((i,k,s))
 assert heading_indices and heading_indices[0][1]==(1,0),(name,heading_indices[:3])
 assert len(heading_indices)>20,(name,len(heading_indices))
 seen=set();keep=main[:heading_indices[0][0]]; removed=[]
 for i,(pidx,key,title) in enumerate(heading_indices):
  end=heading_indices[i+1][0] if i+1<len(heading_indices) else len(main)
  block=main[pidx:end]
  if key in keep_future_exclusions.get(no,set()):
   removed.append({'article':f'{key[0]}'+(f'-{key[1]}' if key[1] else ''),'reason':'2027-03-09 staged-effective article (excluded previously)'}); continue
  if key in seen:
   removed.append({'article':f'{key[0]}'+(f'-{key[1]}' if key[1] else ''),'reason':'repeated article header; later alternative staged-effective version excluded'});continue
  seen.add(key); keep.extend(block)
 # Preserve all untouched body text incl appendix separately; explicitly title as *text extraction* not guaranteed complete HWP tables.
 stem=f'{prefix}_{effect}'
 (OUT/'raw'/f'{stem}_추출원문_부칙포함.txt').write_text(meta+'\n\n'+'\n'.join(full)+'\n',encoding='utf-8')
 (OUT/'raw'/f'{stem}_본문_단일판본.txt').write_text(meta+'\n\n'+'\n'.join(keep)+'\n',encoding='utf-8')
 # Legacy two: use unchanged rows exactly.
 if no in ['21445','36671']:
  rows=[r for r in prev if r['source_url']==url]
  assert rows,(name,prefix)
  result=None;dropped=[]
 else:
  result=ingest_text('\n'.join(keep),domain='law',id_prefix=prefix,source_url=url,license='저작권법 제7조(법령)',mode='article',title_prefix=name,reliability=0.8,lang='ko')
  rows=[{k:str(getattr(item,k)) for k in cols} for item in result.items]
  dropped=result.dropped
  assert len(rows)>10,(name,len(rows))
  allrows+=rows
  allogsnote=f'사용자가 제공한 국가법령정보센터 HWPML 판본(시행 {effect}, 제{no}호)을 2026-10-30 추출; 부칙 및 반복 미래 시행 조문은 지식 CSV에서 제외; 표·별표/서식 추출 여부 미확인; 원문 사람 대조 필요; 신뢰도 0.8은 공식출처 분류용 임시값'
  alllogs += [dict(id=r['id'],source_url=url,accessed_date='2026-10-30',license_checked='no',verified_by='',note=allogsnote) for r in rows]
 # Global verify registry samples.
 head_ids=[r['title'] for r in rows if r['title'].startswith(name+' 제')]
 for sample_key in [(1,0),(2,0)]:
  h=next((h for h in heading_indices if h[1]==sample_key),None)
  if h:
   cand=[r for r in rows if re.match(re.escape(name)+r' 제'+str(sample_key[0])+r'조(?:\(|\s|$)',r['title'])]
   verify.append(dict(document=name,article='제'+str(sample_key[0])+'조',source_version=effect,source_url=url,knowledge_ids='|'.join(x['id'] for x in cand),first_text_excerpt=h[2][:100],check_status='미확인',verified_by='',notes='HWPML 본문과 지식 CSV의 해당 조문 전체 비교 필요'))
 audits.append({
  'document':name,'law_number':no,'effective_date':effect,'kind':'대통령령' if '시행령' in name else '법률',
  'role':group,'source_file':f.name,'source_file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
  'source_size_bytes':f.stat().st_size,'metadata':meta,'start_paragraph_index':start,
  'first_appendix_index':stop,'all_xml_paragraphs':len(ps),'main_text_paragraphs':len(main),
  'article_headings_found':len(heading_indices),'unique_article_numbers_after_filter':len(seen),
  'repeated_or_future_article_exclusions':removed,'knowledge_rows':len(rows),'rows_derived_from':'previous_unchanged' if no in ['21445','36671'] else 'new_ingestion',
  'dropped_short_or_deleted':dropped,'future_effective_marks_in_main':[s[:140] for s in main if '[시행일:' in s][:30],
  'annex_extraction_verified':False,'legal_accuracy_human_verified':False,
 })
 print(name,'rows',len(rows),'articleheads',len(heading_indices),'removed',removed,'dropped',len(dropped),flush=True)

assert len(allrows)==len(alllogs)
assert len({r['id'] for r in allrows})==len(allrows)
assert {r['id'] for r in allrows}=={r['id'] for r in alllogs}
assert all(r['license_checked']=='no' and r['verified_by']=='' for r in alllogs)
assert len(set(r['source_url'] for r in allrows))==10
assert all(r['domain']=='law' for r in allrows)
assert all(r['license']!='unknown' for r in allrows)
assert not any('\n부칙 <' in r['text'] for r in allrows)
assert {r['id']:r for r in prev}=={r['id']:r for r in allrows[:len(prev)]}

def writecsv(path,header,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)
writecsv(OUT/'data/knowledge.csv',cols,allrows)
writecsv(OUT/'data/source_log.csv',SOURCE_LOG_COLUMNS,alllogs)
writecsv(OUT/'docs/research/법령10종_원문대조표.csv',list(verify[0]),verify)
writecsv(OUT/'docs/research/법령10종_판본목록.csv',['document','law_number','effective_date','role','knowledge_rows','source_file','sha256','official_url','human_check'],[
 dict(document=a['document'],law_number=a['law_number'],effective_date=a['effective_date'],role=a['role'],knowledge_rows=a['knowledge_rows'],source_file=a['source_file'],sha256=a['source_file_sha256'],official_url=next(r[4] for r in registry if r[1]==a['law_number']),human_check='pending') for a in audits
])
# Carry forward B research artifacts without altering their reviewed claims.
with zipfile.ZipFile(ROOT/'KORE_B_추가작업_2026-10-30.zip') as z:
 for n in ['docs/research/license_check.md','docs/research/overlap_B.csv','docs/research/domain_candidates.csv','docs/research/paper_review_notes.md','docs/research/domain_license_notes.md']:
  if n in z.namelist():(OUT/n).write_bytes(z.read(n))
# Carry forward earlier transformation reports for audit lineage.
with zipfile.ZipFile(BASEZIP) as z:
 for n in z.namelist():
  if n.startswith('docs/research/') and not n.endswith('/') and not (OUT/n).exists():
   (OUT/n).write_bytes(z.read(n))

cmd=[sys.executable,'-m','kore','validate','--knowledge',str(OUT/'data/knowledge.csv')]
res=subprocess.run(cmd,cwd=CODE,capture_output=True,text=True,timeout=100)
qa={'law_count':10,'csv_rows':len(allrows),'source_log_rows':len(alllogs),'unique_ids':len(set(r['id'] for r in allrows)),
    'source_urls':len(set(r['source_url'] for r in allrows)),'duplicates_removed':sum(len(x['repeated_or_future_article_exclusions']) for x in audits),
    'license_human_verified':False,'human_text_comparison_completed':False,
    'validator_exit_code':res.returncode,'validator_stdout':res.stdout[-12000:],'validator_stderr':res.stderr[-4000:],'documents':audits}
(OUT/'docs/research/법령10종_자동검사.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print('VALIDATOR',res.returncode,res.stdout[-2000:],res.stderr[-1000:]); assert res.returncode==0

lines=['# KORE 지용(B) 법령 10종 수집·변환 보고서','',
      '작성일: 2026-10-30. **사용자 제공 HWP 파일 10종**에서 본문 텍스트를 추출했습니다. 별도의 사주팔자 PDF는 읽거나 분석하거나 결과물에 포함하지 않았습니다.','',
      '## 실제 완료','',
      f'- 업로드된 법령 HWP **10개** 판본의 헤더(시행일·법령번호)와 파일명 일치 검사.',
      f'- 기존 개인정보 보호법·시행령 데이터 **264행을 그대로 보존**하고 나머지 8종을 추가해 `data/knowledge.csv` **{len(allrows)}행**, `data/source_log.csv` **{len(alllogs)}행** 구성.',
      '- 프로젝트 제공 `kore.ingest.ingest_text(mode="article")`으로 조문별 분할·긴 조문 분할. 삭제/짧은 조문은 원문 TXT에 보존했으나 지식 CSV에서는 제외될 수 있습니다.',
      '- HWPML XML 텍스트 추출본(부칙 포함)과 단일 판본 본문을 `raw/`에 별도로 저장.',
      '- `source_log`의 `verified_by`는 전부 공란, `license_checked`는 전부 `no`입니다. 사용자가 직접 확인하지 않은 사실을 확인했다고 표시하지 않았습니다.',
      '- CSV ID는 서로 중복되지 않습니다. 이전 264행 ID는 유지했습니다.',
      f'- 프로젝트 실제 `python -m kore validate --knowledge ...` 실행: 종료 코드 **{res.returncode}**.',
      '', '## 자료별 현황','', '| 법령 | 시행일 | 조문 구분 행 수 | 상태 |','|---|---|---:|---|']
for a in audits:
 lines.append(f"| {a['document']} | {a['effective_date'][:4]}-{a['effective_date'][4:6]}-{a['effective_date'][6:]} | {a['knowledge_rows']} | {'기존 2종 유지' if a['rows_derived_from']=='previous_unchanged' else '신규 변환'} |")
lines+=['','## 최종 실험 전 확인할 항목','',
 '- 법률과 시행령의 **시행일이 모두 동일한 날짜는 아닙니다**. 같은 질문에 서로 다른 시점의 규정이 혼합되어 정답이 달라지는지 확인해 주세요.',
 '- **전자정부법 제43조의2**가 현행과 2026-12-03 시행 예정 본문으로 두 번 실렸습니다. 첫 번째 판본만 CSV에 사용했고, 두 번째 미래 판본은 원문 TXT에서만 보존했습니다. 해당 조문을 평가에 사용하기 전 공식 원문과 비교해야 합니다.',
 '- **개인정보 보호법**의 2027년 시행 예약 조문은 기존 처리 방식을 그대로 보존했습니다. 개인정보 보호법 시행령에는 제42조의2 일부의 2027-02-20 시행 표시가 있습니다. 관련 질문은 검증 전 제외합니다.',
 '- **정보공개법(2023-11-17 판본) + 정보공개법 시행령(2026-01-02 판본)**의 시간차를 우선 대조해야 합니다. 법률이 최신인지 자동으로 보장하지 않습니다.',
 '- PDF/HWP의 별표·서식·표가 전체 추출됐다고 판단하지 않았습니다. 별표를 직접 참조하는 질문은 원문을 검증하고 필요하면 별표 데이터를 추가해야 합니다.',
 '- 법률 조문이 쪼개진 일부 레코드는 개별로는 근거가 불충분할 수 있습니다. 범규(C)가 질문과 `expected_source_ids`를 연결할 때 분할된 부분의 ID를 모두 확인해야 합니다.',
 '- **법령 형식 검증 성공은 정확성·라이선스·실험 적합성 승인이 아닙니다.**',
 '', '## 네가 할 최소 확인','',
 '1. `docs/research/법령10종_원문대조표.csv`를 열고 각 법령의 제1조·제2조 (총 20항목)를 공식 사이트 조문과 비교한 뒤 확인한 항목만 체크.',
 '2. 2026-12-03 예정인 전자정부법 제43조의2와 2027년 시행 예정인 개인정보 보호법 관련 조문의 이용 여부를 서준(A)과 결정.',
 '3. 법률 원문 및 이용 조건을 직접 확인한 뒤 `source_log.csv`의 `license_checked` / `verified_by` 기록.',
 '4. 서준(A)에게 자료 버전과 조문 분할 기준, 2023년 정보공개법 판본 채택 여부 승인 요청.',
 '', '## 생성 파일','',
 '- `data/knowledge.csv` : KORE 조문 지식 데이터 10종',
 '- `data/source_log.csv` : 모든 데이터 행의 출처·확인상태',
 '- `raw/*_추출원문_부칙포함.txt` : HWPML에서 추출한 문자 원문(법령당 1개)',
 '- `raw/*_본문_단일판본.txt` : 주 데이터에 사용된 단일 판본 본문',
 '- `docs/research/법령10종_자동검사.json` : 법령별 원본 해시/중복/검증 결과',
 '- `docs/research/법령10종_원문대조표.csv` : 직접 검증할 표본과 ID',
 '- `docs/research/법령10종_판본목록.csv` : 판본 메타데이터·업로드 출처 기록',
 '- 기타 라이선스·논문 조사 초안: 이전 조사본에서 복사한 자료.',
 '', '## 서준(A) 승인 필요','',
 '- 10개 법령 선정과 혼동용 2개 자료의 채택.',
 '- 법령 시행일을 하나로 통일할지, 제공 판본 그대로 쓸지 결정.',
 '- 연구 사용 전 검증된 법령·조문만 데이터에 포함할지 결정.',
 '- 조문 분할 크기 및 `expected_source_ids` 생성 기준 확정.',
 '']
(OUT/'B_진행보고.md').write_text('\n'.join(lines),encoding='utf-8')
shutil.copy2(__file__,OUT/'tools/build_kore_10_laws.py')
zipout=ROOT/'KORE_B_법령10종_통합_2026-10-30.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
 for path in sorted(OUT.rglob('*')):
  if path.is_file():z.write(path,path.relative_to(OUT))
print('ZIP',zipout,zipout.stat().st_size,'ROWS',len(allrows),'PER LAW',[(a['document'],a['knowledge_rows']) for a in audits])
