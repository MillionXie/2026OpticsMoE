"""Bounded authenticated-SSH DATA framing for actual T12 acquisition ledgers.

This module neither initializes devices nor runs models. Only the three CCD
files or the two prepared-input files belonging to the fixed ledger may move.
"""
import argparse,hashlib,io,json,struct,sys,time,zlib
from pathlib import Path
import numpy as np
from .scene_lab_protocol import Ledger,STAGES,identity,array_sha,encode,put

MAGIC=b'T12DATA1\n';LIMIT=8<<20


def exact(stream,count):
    chunks=[];left=count
    while left:
        value=stream.read(left)
        if not value:raise EOFError('Incomplete data packet')
        chunks.append(value);left-=len(value)
    return b''.join(chunks)


def packet(stream,header,blocks=()):
    data=json.dumps(header,separators=(',',':')).encode()
    if len(data)>65536:raise ValueError('Oversized packet header')
    stream.write(struct.pack('>I',len(data)));stream.write(data)
    for block in blocks:stream.write(block)
    stream.flush()


def read_packet(stream):
    size=struct.unpack('>I',exact(stream,4))[0]
    if not 0<size<=65536:raise ValueError('Invalid data header')
    header=json.loads(exact(stream,size));files={}
    for row in header.get('files',[]):
        if row['name']in files or not 0<=row['bytes']<=LIMIT or not 0<=row['encoded_bytes']<=LIMIT:raise ValueError('Invalid data-file limits')
        value=exact(stream,row['encoded_bytes'])
        if row['codec']=='zlib':
            decoder=zlib.decompressobj();value=decoder.decompress(value,row['bytes']+1)
            if not decoder.eof or decoder.unconsumed_tail:raise ValueError('Invalid lossless data encoding')
        elif row['codec']!='raw':raise ValueError('Unknown data encoding')
        if len(value)!=row['bytes']or hashlib.sha256(value).hexdigest()!=row['SHA256']:raise ValueError('Data checksum mismatch')
        files[row['name']]=value
    return header,files


def validate_inputs(ticket,payload,ledger,request,stage):
    if ticket['ledger_sha256']!=ledger.key or ticket['request_sha256']!=identity(request)or ticket['stage']!=stage or len(ticket['prefix_receipts'])!=STAGES.index(stage):raise ValueError('Wrong input lineage')
    with np.load(io.BytesIO(payload),allow_pickle=False)as z:
        a,p,ag,pg=encode(z['amplitude'],z['total_phase']);items=dict(amplitude=(a,'amplitude_sha256'),total_phase=(p,'total_phase_sha256'),amplitude_code=(ag,'logical_amplitude_code_sha256'),phase_code=(pg,'logical_phase_code_sha256'))
        if set(z.files)!=set(items)or any(not np.array_equal(z[n],v)or array_sha(v)!=ticket[k]for n,(v,k)in items.items()):raise ValueError('Changed native input arrays')


def write_data(path,data,ready):
    """Never change completed data; preserve an interrupted transport prefix."""
    if path.exists():
        old=path.read_bytes()
        if old==data:return
        if ready.exists():raise ValueError(('Completed DATA changed',str(path)))
        # Only an exact byte prefix from an interrupted copy is recoverable here.
        if not data.startswith(old):raise ValueError(('Unknown existing DATA; preserved',str(path)))
        backup=path.with_name(path.name+'.partial_transport_'+hashlib.sha256(old).hexdigest()[:12])
        if backup.exists():raise FileExistsError(backup)
        path.rename(backup)
    with path.open('xb')as f:f.write(data)


def wait_ready(folder,ledger,stage,direction,seconds):
    marker=folder/('input.ready'if direction=='inputs'else'capture.ready');deadline=time.monotonic()+seconds
    while not marker.exists():
        abort=ledger.root.parent/'input_abort.json'
        if abort.exists():raise RuntimeError('Run transfer/capture was stopped')
        if direction=='outputs':
            pointer=ledger.root.parent/(stage+'_active_capture.json');job='capture_'+stage
            if pointer.exists():
                job=json.loads(pointer.read_text())['job']
                if Path(job).name!=job or not job.startswith('capture_'+stage):raise ValueError('Invalid active-capture pointer')
            returned=ledger.root.parent/(job+'_return.json')
            if returned.exists()and json.loads(returned.read_text())['returncode']!=0:raise RuntimeError('Capture stopped before this sample')
        if time.monotonic()>=deadline:raise TimeoutError(('Missing ready data',str(folder)))
        time.sleep(.2)


def completed_prefix(ledger,stage,direction):
    count=0
    for request in ledger.requests:
        folder=ledger.folder(stage,request)
        if not(folder/('input.ready'if direction=='inputs'else'capture.ready')).exists():break
        if direction=='inputs':validate_inputs(json.loads((folder/'ticket.json').read_text()),(folder/'logical_fields.npz').read_bytes(),ledger,request,stage)
        else:ledger.receipt(stage,request)
        count+=1
    return count


def send(ledger,stage,direction,stream,wait_seconds,start_index=0):
    if not 0<=start_index<=len(ledger.requests):raise ValueError('Invalid stream start')
    stream.write(MAGIC);stream.flush();names=('logical_fields.npz','ticket.json')if direction=='inputs'else('raw.npy','calibrated.npy','receipt.json')
    for index,request in enumerate(ledger.requests[start_index:],start_index):
        folder=ledger.folder(stage,request);wait_ready(folder,ledger,stage,direction,wait_seconds)
        if direction=='outputs':ledger.receipt(stage,request)
        rows=[];blocks=[]
        for name in names:
            value=(folder/name).read_bytes();encoded=value if name.endswith('.npz')else zlib.compress(value,1);codec='raw'if name.endswith('.npz')else'zlib'
            if len(encoded)>=len(value):encoded=value;codec='raw'
            rows.append(dict(name=name,bytes=len(value),encoded_bytes=len(encoded),SHA256=hashlib.sha256(value).hexdigest(),codec=codec));blocks.append(encoded)
        packet(stream,dict(kind='sample',index=index,ledger=ledger.key,stage=stage,direction=direction,request=identity(request),files=rows),blocks)
        if(index+1)%32==0:print(json.dumps(dict(sent=index+1,total=len(ledger.requests),stage=stage,direction=direction)),file=sys.stderr,flush=True)
    packet(stream,dict(kind='end',count=len(ledger.requests),ledger=ledger.key,stage=stage,direction=direction))


def receive(ledger,stage,direction,stream,start_index=0):
    if not 0<=start_index<=completed_prefix(ledger,stage,direction):raise ValueError('Resume prefix has not been verified')
    if exact(stream,len(MAGIC))!=MAGIC:raise ValueError('Not a T12 data stream')
    count=start_index;allowed={'logical_fields.npz','ticket.json'}if direction=='inputs'else{'raw.npy','calibrated.npy','receipt.json'}
    while True:
        header,files=read_packet(stream)
        if header['ledger']!=ledger.key or header['stage']!=stage or header['direction']!=direction:raise ValueError('Wrong destination ledger')
        if header['kind']=='end':
            if count!=len(ledger.requests)or header['count']!=count:raise ValueError('Incomplete cohort')
            break
        if count>=len(ledger.requests)or header.get('index')!=count or set(files)!=allowed:raise ValueError('Wrong data order/files')
        request=ledger.requests[count]
        if header['request']!=identity(request):raise ValueError('Wrong request')
        folder=ledger.folder(stage,request);folder.mkdir(parents=True,exist_ok=True);ready=folder/('input.ready'if direction=='inputs'else'capture.ready')
        if direction=='inputs':validate_inputs(json.loads(files['ticket.json']),files['logical_fields.npz'],ledger,request,stage)
        else:
            receipt=json.loads(files['receipt.json'])
            if receipt['ledger_sha256']!=ledger.key or receipt['metadata']['frame_provenance']!='physical_linear_camera':raise ValueError('Not actual CCD data')
            for name,key in(('raw.npy','raw_file_sha256'),('calibrated.npy','calibrated_file_sha256')):
                if hashlib.sha256(files[name]).hexdigest()!=receipt[key]:raise ValueError('CCD receipt mismatch')
        for name,data in files.items():write_data(folder/name,data,ready)
        if direction=='outputs':ledger.receipt(stage,request)
        if not ready.exists():ready.touch(exist_ok=False)
        count+=1
        if count==1 or count%32==0:print(json.dumps(dict(received=count,total=len(ledger.requests),stage=stage,direction=direction)),flush=True)
    path=ledger.root.parent/(stage+'_'+direction+'_data_stream_complete.json');report=dict(complete=True,count=count,stage=stage,direction=direction,ledger_SHA256=ledger.key,SDK=False,GPU=False,TEST_used=False)
    if path.exists():
        if json.loads(path.read_text())!=report:raise ValueError('Changed completion report')
    else:put(path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('send','receive','prefix'),required=True);p.add_argument('--direction',choices=('inputs','outputs'),required=True);p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--wait-seconds',type=float,default=240);p.add_argument('--start-index',type=int,default=0);a=p.parse_args()
    if not 0<a.wait_seconds<=600:raise ValueError('Bounded wait required')
    c=json.loads((a.ledger/'contract.json').read_text());ledger=Ledger(a.ledger,c['session'],c['models'],c['requests']);ledger.ready(a.stage)
    if a.mode=='send':send(ledger,a.stage,a.direction,sys.stdout.buffer,a.wait_seconds,a.start_index)
    elif a.mode=='receive':receive(ledger,a.stage,a.direction,sys.stdin.buffer,a.start_index)
    else:print(json.dumps(dict(next_index=completed_prefix(ledger,a.stage,a.direction),ledger=ledger.key,stage=a.stage,direction=a.direction)),flush=True)


if __name__=='__main__':main()
