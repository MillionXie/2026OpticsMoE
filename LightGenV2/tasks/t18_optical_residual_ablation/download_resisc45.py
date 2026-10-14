"""Single verified acquisition using scoped public DNS answers, unchanged TLS/SNI."""
import argparse,hashlib,json,os,socket,subprocess,time,urllib.parse,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())

def acquire(out,answers,mirror=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    plan=json.loads((HERE/'resisc20_data_plan.json').read_text())
    assert (not answers and mirror) or (set(answers)<= {'huggingface.co','cas-bridge.xethub.hf.co'} and 'huggingface.co' in answers)
    for host,record in answers.items():
        assert record['provider']=='https://dns.google/resolve' and time.time()-record['queried_unix']<300
        assert record['addresses'] and all(socket.inet_aton(ip) for ip in record['addresses'])
    out.mkdir(parents=True,exist_ok=False)
    url=plan['alternate_download_url'] if mirror else plan['download_url']
    save(out/'acquisition_claim.json',dict(pid=os.getpid(),started_unix=time.time(),url=url,
         expected_sha256=plan['archive_sha256'],dns_answers=answers,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         source_sha256=sha(__file__),scope='Separate acquisition after recorded server DNS failure; no automatic retry, no global DNS/env change',automatic_retry=False))
    original=socket.getaddrinfo
    def scoped(host,port,*args,**kwargs):
        if host in answers:
            records=[]
            for ip in answers[host]['addresses']:records.extend(original(ip,port,*args,**kwargs))
            return records
        return original(host,port,*args,**kwargs)
    socket.getaddrinfo=scoped
    part=out/'NWPU-RESISC45.download.part'
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'T18 verified research acquisition'})
        with urllib.request.urlopen(request,timeout=60) as response,part.open('xb') as f:
            host=urllib.parse.urlparse(response.geturl()).hostname
            while True:
                block=response.read(4*1024*1024)
                if not block:break
                f.write(block)
            f.flush();os.fsync(f.fileno())
        assert sha(part)==plan['archive_sha256'],'Known canonical ZIP SHA mismatch'
        archive=out/'NWPU-RESISC45.zip';part.rename(archive)
        receipt=dict(passed=True,archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,resolved_host=host,
                     completed_unix=time.time(),tls_certificate_verification=True,original_hostname_SNI=True,
                     requested_url=url,alternate_transport=mirror,
                     source_scope='Same canonical archive pinned revision and SHA; DNS answers affect only this process, or ordinaryDNS for alternate mirror',automatic_retry=False)
        save(out/'acquisition_receipt.json',receipt);print(json.dumps(receipt),flush=True)
    except BaseException as e:
        save(out/'failure.json',dict(error=repr(e),automatic_retry=False));raise
    finally:socket.getaddrinfo=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resolved-dns-json',default='{}');p.add_argument('--mirror',action='store_true');a=p.parse_args();acquire(a.out,json.loads(a.resolved_dns_json),a.mirror)
