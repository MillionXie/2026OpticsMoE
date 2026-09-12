import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from run import save_capture


class StorageTests(unittest.TestCase):
    def test_minimal_and_opt_in(self):
        raw=np.arange(24,dtype=np.uint8).reshape(4,6)
        def write(p,v):p.write_text(json.dumps(v),encoding='utf-8')
        for keep in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'frame.png'
                result=save_capture(raw,{},p,{'save_raw_frames':keep},
                                    lambda a,c:a[1:3,2:5],write,'test')
                np.testing.assert_array_equal(np.asarray(Image.open(p)),raw[1:3,2:5])
                np.testing.assert_array_equal(result,raw[1:3,2:5])
                self.assertEqual(p.with_suffix('.raw.png').exists(),keep)
                self.assertEqual(len(list(Path(tmp).glob('*'))),3 if keep else 2)
                self.assertEqual(json.loads(p.with_suffix('.json').read_text())['raw_frame_saved'],keep)


if __name__=='__main__':unittest.main()
