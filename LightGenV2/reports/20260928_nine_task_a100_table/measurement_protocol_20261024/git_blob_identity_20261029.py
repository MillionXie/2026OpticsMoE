"""One bounded Git batch read; the same individual published SHA checks."""
import hashlib
import re
import subprocess

def verify(commit,frozen,repo):
    if not re.fullmatch('[a-f0-9]{40}',commit):raise ValueError('Exact published commit required')
    names=list(frozen)
    if any('\n' in name or '\r' in name for name in names):raise ValueError('Invalid source path')
    query=''.join(commit+':'+name.replace('\\','/')+'\n' for name in names).encode()
    result=subprocess.run(['git','cat-file','--batch'],input=query,capture_output=True,cwd=repo,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf8'))
    check_batch(result.stdout,names,frozen)

def check_batch(data,names,frozen):
    offset=0
    for name in names:
        end=data.index(b'\n',offset);header=data[offset:end].decode('ascii').split()
        if len(header)!=3 or header[1]!='blob':raise ValueError('Published source blob missing: '+name)
        size=int(header[2]);start=end+1;blob=data[start:start+size]
        if len(blob)!=size or data[start+size:start+size+1]!=b'\n':raise ValueError('Incomplete Git batch record')
        if hashlib.sha256(blob).hexdigest()!=frozen[name]:raise ValueError('Published source differs: '+name)
        offset=start+size+1
    if offset!=len(data):raise ValueError('Unexpected additional Git batch bytes')
