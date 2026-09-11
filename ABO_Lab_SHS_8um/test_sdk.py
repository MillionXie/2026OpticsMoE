import ctypes
import unittest
from sdk import NodeInfo, IntegerInfo, FloatInfo


class ABITests(unittest.TestCase):
    def test_packed_node_abi(self):
        self.assertEqual(ctypes.sizeof(IntegerInfo),32)
        self.assertEqual(ctypes.sizeof(FloatInfo),32)
        self.assertEqual(ctypes.sizeof(NodeInfo),104)
        self.assertEqual(NodeInfo.visibility.offset,40)


if __name__=='__main__':unittest.main()
