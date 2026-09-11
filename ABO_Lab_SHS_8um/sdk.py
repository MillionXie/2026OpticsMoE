"""Explicit x64 ctypes binding to installed Magewell CEasyCapS SDK.

No vendor demo import: it hardcodes C: paths, changes card configuration, and
loads a trigger profile for another camera. No license bypass or GUI capture.
"""
import ctypes as C
import os
from pathlib import Path


class IntegerInfo(C.Structure):
    _pack_ = 1
    _fields_ = [(k, C.c_int64) for k in ('minimum','maximum','increment','value')]

class FloatInfo(C.Structure):
    _pack_ = 1
    _fields_ = [(k, C.c_double) for k in ('minimum','maximum','increment','value')]

class ValueInfo(C.Union):
    _pack_ = 1
    _fields_ = [('integer',IntegerInfo),('floating',FloatInfo),('raw',C.c_uint8*32)]

class NodeInfo(C.Structure):
    _pack_ = 1
    _fields_ = [('node_type',C.c_int32),('access',C.c_int32),('info',ValueInfo),
                ('visibility',C.c_int32),('reserved',C.c_uint8*60)]


class SDKError(RuntimeError): pass


class Camera:
    def __init__(self, config):
        self.config=config
        self.dll=None; self.card=None; self.handle=None
        self.initialized=False; self.streaming=False; self.dll_dirs=[]

    def _bind(self, name, ret, *args):
        fn=getattr(self.dll,name);fn.restype=ret;fn.argtypes=list(args)
        return fn

    def open(self):
        if os.name!='nt' or C.sizeof(C.c_void_p)!=8: raise RuntimeError('Requires Windows x64 Python')
        root=Path(self.config['sdk_root']).resolve()
        dllpath=root/'demo/base_dll/bin/CEasyCapS.dll'
        cti=root/'cti/x86_64'
        if not dllpath.is_file() or not (cti/'cxplink_gentl.cti').is_file(): raise FileNotFoundError('Incomplete Magewell SDK: '+str(root))
        # Only this process is affected; no machine environment/config changes.
        self._old_env={k:os.environ.get(k) for k in ('GENICAM_GENTL64_PATH','GENICAM_GENTL64_PATH_MAGEWELL')}
        for k in self._old_env: os.environ[k]=str(cti)+os.pathsep
        for path in [dllpath.parent,cti,root/'dll',root/'consumer/bin']:
            if path.is_dir(): self.dll_dirs.append(os.add_dll_directory(str(path)))
        self.dll=C.WinDLL(str(dllpath))
        V=C.c_void_p;B=C.c_bool;U=C.c_uint32;Z=C.c_size_t;L=C.c_int32;S=C.c_char_p
        self._bind('scap_init',B)
        self._bind('scap_done',None)
        self._bind('scap_set_language',None,L)
        self._bind('scap_get_card_num',U)
        self._bind('scap_create_card',V,U)
        self._bind('scap_close_card',B,V)
        self._bind('scap_get_device_num',U,V,U)
        self._bind('scap_create_by_camera',V,V,U)
        self._bind('scap_close_by_camera',B,V)
        self._bind('scap_get_string',B,V,L,S,V,C.POINTER(Z))
        self._bind('scap_set_string',B,V,L,S,S)
        self._bind('scap_get_float',B,V,L,S,C.POINTER(C.c_double))
        self._bind('scap_set_float',B,V,L,S,C.c_double)
        self._bind('scap_get_integer',B,V,L,S,C.POINTER(C.c_int64))
        self._bind('scap_set_integer',B,V,L,S,C.c_int64)
        self._bind('scap_set_command',B,V,L,S)
        self._bind('scap_get_xml',B,V,L,V,C.POINTER(Z))
        self._bind('scap_get_node_info',B,V,L,S,C.POINTER(NodeInfo))
        self._bind('scap_get_last_error_info',L,V,C.POINTER(Z))
        self._bind('scap_start',B,V,U)
        self._bind('scap_stop',B,V)
        self._bind('scap_flush_all_buffers',B,V)
        self._bind('scap_get_buf',B,V,C.c_uint64,C.POINTER(V),C.POINTER(V),C.POINTER(Z))
        self._bind('scap_back_buffer',B,V,V)
        self.dll.scap_set_language(0)
        if not self.dll.scap_init(): raise SDKError('scap_init failed; check GenTL driver/runtime paths')
        self.initialized=True
        self.card_count=int(self.dll.scap_get_card_num())
        print('Detected capture cards:',self.card_count,flush=True)
        self.card=self.dll.scap_create_card(self.config.get('card_index',0))
        self.check(bool(self.card),'open capture card (close Viewer if busy)')
        self.handle=self.dll.scap_create_by_camera(self.card,self.config.get('camera_index',0))
        self.check(bool(self.handle),'connect camera')
        return self

    def error(self):
        if not self.initialized:return 'SDK not initialized'
        buf=C.create_string_buffer(4096);size=C.c_size_t(len(buf))
        code=self.dll.scap_get_last_error_info(buf,C.byref(size))
        return str(code)+': '+buf.value.decode('utf-8',errors='replace')

    def check(self, success, message):
        if not success:raise SDKError(message+'; '+self.error())

    def get(self, name, level=2):
        buf=C.create_string_buffer(4096);size=C.c_size_t(len(buf))
        self.check(self.dll.scap_get_string(self.handle,level,name.encode('ascii'),buf,C.byref(size)),'read '+name)
        return buf.value.decode('utf-8',errors='replace')

    def info(self, name, level=2):
        out=NodeInfo()
        self.check(self.dll.scap_get_node_info(self.handle,level,name.encode('ascii'),C.byref(out)),'node '+name)
        result=dict(type=out.node_type,access=out.access,visibility=out.visibility)
        if out.node_type in (2,5):
            obj=out.info.integer if out.node_type==2 else out.info.floating
            result.update({k:getattr(obj,k) for k in ('minimum','maximum','increment','value')})
        return result

    def xml(self, level):
        size=C.c_size_t(0)
        self.check(self.dll.scap_get_xml(self.handle,level,None,C.byref(size)),'query XML size')
        if not 0<size.value<16*1024*1024:raise SDKError('Invalid XML size')
        buf=C.create_string_buffer(size.value)
        self.check(self.dll.scap_get_xml(self.handle,level,buf,C.byref(size)),'read XML')
        return bytes(buf[:size.value]).rstrip(b'\0')

    def close(self):
        if self.streaming:
            self.dll.scap_stop(self.handle);self.streaming=False
        if self.handle:self.dll.scap_close_by_camera(self.handle);self.handle=None
        if self.card:self.dll.scap_close_card(self.card);self.card=None
        if self.initialized:self.dll.scap_done();self.initialized=False
        for handle in self.dll_dirs:handle.close()
        self.dll_dirs=[]
        for k,value in getattr(self,'_old_env',{}).items():
            if value is None:os.environ.pop(k,None)
            else:os.environ[k]=value

    def __enter__(self):
        try:return self.open()
        except BaseException:self.close();raise

    def __exit__(self,*args):self.close()
