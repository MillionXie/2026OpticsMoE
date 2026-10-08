from pathlib import Path
import json,hashlib,datetime,csv

PROJECT=Path(__file__).resolve().parents[1]
SPLITS={'train':1188,'val':98,'test':298}
MD5='bbd3c5a5576322bc4cdfea780653b1ce'
URL='https://zenodo.org/records/10519652/files/adrenalmnist3d.npz?download=1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_csv(path,rows,fields=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
