"""Read-only camera/GenICam discovery. Does not start acquisition or set features."""
import argparse
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from sdk import Camera, SDKError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.json')
    p.add_argument('--out',required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    report={'operation':'read-only discovery; no capture or setters','levels':{}}
    with Camera(config['camera']) as camera:
        for level,label in [(0,'interface'),(1,'device'),(2,'camera'),(3,'stream')]:
            try:raw=camera.xml(level)
            except SDKError as ex:
                # SDK 1.1.4.22 source falls through to default for interface XML.
                # Do not confuse its stale last-error with an occupied camera.
                report['levels'][label]={'xml_error':str(ex)}
                continue
            if raw.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:raw=z.read(next(n for n in z.namelist() if n.lower().endswith('.xml')))
            raw=raw.rstrip(b'\0')
            (out/(label+'.xml')).write_bytes(raw)
            tree=ET.fromstring(raw);nodes=[];values={}
            for el in tree.iter():
                tag=el.tag.rsplit('}',1)[-1];name=el.attrib.get('Name','')
                if tag in ('Integer','IntSwissKnife','Float','String','Boolean','Enumeration','Command','Category'):
                    nodes.append({'name':name,'kind':tag})
                    if re.search('DeviceModel|DeviceVendor|DeviceSerial|Firmware|Exposure|Gain|FrameRate|Trigger|Acquisition|^Width$|^Height$|Offset|PixelFormat|ScanType',name) and tag not in ('Command','Category'):
                        try:
                            info=camera.info(name,level)
                            if info['access'] not in (3,4):continue
                            values[name]={'value':camera.get(name,level),'info':info}
                        except SDKError as ex:values[name]={'error':str(ex)}
            report['levels'][label]={'nodes':nodes,'features':values}
            print(label,json.dumps(values,ensure_ascii=False),flush=True)
    (out/'probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Discovery saved:',out.resolve(),flush=True)


if __name__=='__main__':main()
