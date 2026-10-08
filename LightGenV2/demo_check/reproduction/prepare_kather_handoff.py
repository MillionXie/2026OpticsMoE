"""Run the unchanged data preparer using the export commit outside a Git clone."""
import json
from pathlib import Path
from unittest.mock import patch
import prepare_kather2016 as p

if __name__=='__main__':
    original=p.subprocess.check_output
    def provenance(command,*args,**kwargs):
        if command==['git','rev-parse','HEAD']:
            value=json.loads((Path(__file__).resolve().parents[1]/'PROVENANCE.json').read_text())['export_commit']+'\n'
            return value if kwargs.get('text') else value.encode()
        return original(command,*args,**kwargs)
    with patch.object(p.subprocess,'check_output',provenance):p.main()
