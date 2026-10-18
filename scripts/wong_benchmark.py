"""Original-source Wong clinical case parsing. Standard library; no raw text output.

The 340-case XLSX must be supplied locally by the researcher. The repo does
not contain that file. This helper preserves the source-pinned 171/67/102
case-exclusive split used in all prior OpenFARM Wong experiments.
"""
import hashlib,re,zipfile,xml.etree.ElementTree as ET
from pathlib import Path

CATS=['Immediate','15 mins','30-60 mins','120 mins','240 mins']
SHA='4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027'
HEADERS=['Cases','Veterinary triage - median vet','Veterinary triage - vet 1','Veterinary triage - vet 2','Veterinary triage - vet 3','Nurse triage - nurse 1','Nurse triage - nurse 2','ChatGPT-3.5 triage','ChatGPT-4 triage']
T='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
def original_rows(file):
    with zipfile.ZipFile(file) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            r=ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings=[''.join(v.text or '' for v in e.iter(T+'t')) for e in r.findall(T+'si')]
        doc=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows=[]
    for node in doc.findall('.//'+T+'sheetData/'+T+'row'):
        d={}
        for cell in node.findall(T+'c'):
            name=cell.attrib.get('r','')
            mat=re.match(r'([A-Z]+)[0-9]+',name)
            if not mat:raise ValueError('bad cell coordinate')
            ix=0
            for c in mat.group(1):ix=26*ix+ord(c)-64
            ix-=1
            v=cell.find(T+'v')
            if cell.attrib.get('t')=='inlineStr':value=''.join(e.text or '' for e in cell.iter(T+'t'))
            elif v is None:value=''
            elif cell.attrib.get('t')=='s':value=strings[int(v.text)]
            else:value=v.text or ''
            d[ix]=value
        rows.append([d.get(i,'') for i in range(9)])
    return rows
def clean_case(text):
    if not isinstance(text,str) or len(text.strip())<100:raise ValueError('bad clinical presentation')
    state=re.sub(r'(?mi)\bDate\s*:\s*[^\n]*','',text)
    return re.sub(r'(?m)\n{3,}','\n\n',state).strip()
def parse(file):
    file=Path(file)
    sha=hashlib.sha256(file.read_bytes()).hexdigest()
    if sha!=SHA:raise ValueError('original XLSX SHA mismatch; no patient cases processed')
    rows=original_rows(file)
    if len(rows)!=341 or rows[0]!=HEADERS:raise ValueError('original Wong sheet schema mismatch')
    cases=[];ident=set();clean=set()
    for row in rows[1:]:
        text=row[0];rid=hashlib.sha256(text.encode()).hexdigest()[:24]
        if rid in ident:raise ValueError('duplicate case')
        ident.add(rid)
        votes=[str(x).strip() for x in row[2:5]]
        if any(v not in CATS for v in votes):raise ValueError('invalid veterinarian rating')
        median=sorted(votes,key=CATS.index)[1]
        if median!=row[1]:raise ValueError('published vet median differs')
        state=clean_case(text);fingerprint=hashlib.sha256(state.encode()).hexdigest()
        if fingerprint in clean:raise ValueError('duplicate after date scrub')
        clean.add(fingerprint)
        cases.append({'id':rid,'state':state,'median':median,'vet_votes':votes})
    return cases,{'source_sha256':sha,'rows':340,'median_verified':340}
def splits(cases):
    lookup={};counts={}
    for name in CATS:
        members=[x for x in cases if x['median']==name]
        members.sort(key=lambda x:hashlib.sha256(('openfarm-wong-v1:'+x['id']).encode()).hexdigest())
        n=len(members);test=round(n*.30);cal=round(n*.20)
        for i,item in enumerate(members):
            p='test' if i<test else ('cal' if i<test+cal else 'train')
            lookup[item['id']]=p
        counts[name]={'train':n-test-cal,'cal':cal,'test':test}
    if len(lookup)!=len(cases):raise ValueError('split source cardinality mismatch')
    return lookup,counts
