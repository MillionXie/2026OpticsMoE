"""Explicit x64 ctypes binding to installed Magewell CEasyCapS SDK.

No vendor demo import: it hardcodes C: paths, changes card configuration, and
loads a trigger profile for another camera. No license bypass or GUI capture.
"""
import ctypes as C
import os
import time
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


class BufferInfo(C.Structure):
    _pack_ = 1
    _fields_ = [(k,C.c_int64) for k in ('offset_x','offset_y','width','height','format')] + [
        ('p_base',C.c_void_p),('p_private',C.c_void_p),('buf_size',C.c_size_t),
        ('timestamp',C.c_uint64),('frame_id',C.c_uint64),('tl_type',C.c_char*32)]


def decode_mono(raw, width, height, pixel_format, *, xpadding=0, image_offset=0):
    """PFNC unpacked Mono only. Never interpret packed samples as uint16."""
    import numpy as np
    formats={0x01080001:('u1',8),0x01100003:('<u2',10),
             0x01100005:('<u2',12),0x01100025:('<u2',14),0x01100007:('<u2',16)}
    if pixel_format not in formats:raise SDKError(f'Unsupported format 0x{pixel_format:08x}; select Mono8 or implement validated unpacking')
    dtype,bits=formats[pixel_format];item=np.dtype(dtype).itemsize
    if min(width,height)<=0 or min(xpadding,image_offset)<0:raise SDKError('Invalid buffer geometry')
    stride=width*item+xpadding
    if len(raw)<image_offset+stride*height:raise SDKError('Short image buffer')
    arr=np.ndarray((height,width),dtype=dtype,buffer=raw,offset=image_offset,strides=(stride,item)).copy()
    if int(arr.max())>(1<<bits)-1:raise SDKError('Samples exceed nominal bit depth; verify alignment')
    return arr,bits


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
        self._bind('scap_get_buf_info',B,V,V,C.POINTER(BufferInfo),C.c_uint8)
        self._bind('scap_get_port',B,V,U,C.POINTER(V))
        # The same loaded GenTL producer, not a second independent acquisition.
        self.cti=C.WinDLL(str(cti/'cxplink_gentl.cti'))
        self.cti.DSGetBufferInfo.restype=L
        self.cti.DSGetBufferInfo.argtypes=[V,V,L,C.POINTER(L),V,C.POINTER(Z)]
        self.dll.scap_set_language(0)
        if not self.dll.scap_init(): raise SDKError('scap_init failed; check GenTL driver/runtime paths')
        self.initialized=True
        self.card_count=int(self.dll.scap_get_card_num())
        print('Detected capture cards:',self.card_count,flush=True)
        self.card=self.dll.scap_create_card(self.config.get('card_index',0))
        self.check(bool(self.card),'open capture card (close Viewer if busy)')
        self.handle=self.dll.scap_create_by_camera(self.card,self.config.get('camera_index',0))
        self.check(bool(self.handle),'connect camera')
        for key,node in [('expected_model','DeviceModelName'),('expected_pixel_format','PixelFormat')]:
            if self.config.get(key) and self.get(node)!=self.config[key]:raise SDKError('Unexpected '+node+': '+self.get(node))
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
        # ZIP end-of-central-directory ends in NUL bytes: never rstrip it.
        return bytes(buf[:size.value])

    def set(self,name,value,level=2):
        info=self.info(name,level)
        if info['access']!=4:raise SDKError('Not writable: '+name)
        if info['type'] in (2,5):
            if not info['minimum']<=float(value)<=info['maximum']:
                raise ValueError(f"{name}={value} outside [{info['minimum']},{info['maximum']}]; frame rate limits exposure")
        self.check(self.dll.scap_set_string(self.handle,level,name.encode('ascii'),str(value).encode('ascii')),'set '+name)
        actual=self.get(name,level)
        if info['type'] in (2,5):
            tolerance=max(abs(float(value))*1e-6,float(info.get('increment',0)),1e-6)
            if abs(float(actual)-float(value))>tolerance:raise SDKError(f'{name} readback mismatch: {value} -> {actual}')
        elif actual!=str(value):raise SDKError(f'{name} readback mismatch: {value} -> {actual}')
        return actual

    def start(self):
        if self.streaming:raise SDKError('Already streaming')
        count=int(self.config.get('buffer_count',4))
        if not 2<=count<=16:raise ValueError('buffer_count must be 2..16')
        self.check(self.dll.scap_start(self.handle,count),'start acquisition')
        self.streaming=True
        self.stream=C.c_void_p()
        self.check(self.dll.scap_get_port(self.handle,3,C.byref(self.stream)),'stream handle')

    def stop(self):
        if self.streaming:
            self.check(self.dll.scap_stop(self.handle),'stop acquisition')
            self.streaming=False

    def buffer_value(self,hbuf,command,ctype,required=True):
        value=ctype();kind=C.c_int32();size=C.c_size_t(C.sizeof(value))
        code=self.cti.DSGetBufferInfo(self.stream,hbuf,command,C.byref(kind),C.byref(value),C.byref(size))
        if code:
            if required:raise SDKError(f'GenTL buffer info {command} unavailable ({code})')
            return None
        return value.value

    def grab(self):
        """Copy exactly one complete frame and return its buffer in finally."""
        if not self.streaming:raise SDKError('Call start before grab')
        hbuf=C.c_void_p();base=C.c_void_p();size=C.c_size_t();t=time.perf_counter()
        self.check(self.dll.scap_get_buf(self.handle,int(self.config.get('timeout_ms',3000)),C.byref(hbuf),C.byref(base),C.byref(size)),'get frame (timeout in ms)')
        try:
            info=BufferInfo()
            self.check(self.dll.scap_get_buf_info(self.handle,hbuf,C.byref(info),0),'buffer metadata')
            incomplete=self.buffer_value(hbuf,7,C.c_uint8)
            filled=self.buffer_value(hbuf,9,C.c_size_t)
            xpad=self.buffer_value(hbuf,14,C.c_size_t)
            offset=self.buffer_value(hbuf,18,C.c_size_t)
            if incomplete or not 0<filled<=size.value or not base.value:raise SDKError('Incomplete/empty image rejected')
            if info.buf_size!=size.value or info.p_base!=base.value:raise SDKError('Buffer pointers/sizes disagree')
            if size.value>512*1024**2:raise SDKError('Unexpected payload >512 MiB')
            endian=self.buffer_value(hbuf,26,C.c_int32,False)
            # GenTL PIXELENDIANNESS_LITTLE=1; Mono8 is endian independent.
            if info.format!=0x01080001 and endian not in (None,1):raise SDKError('Non-little-endian Mono buffer not implemented')
            raw=C.string_at(base,filled)
            arr,bits=decode_mono(raw,info.width,info.height,info.format,xpadding=xpad,image_offset=offset)
            meta={'shape_hw':list(arr.shape),'dtype':str(arr.dtype),'significant_bits':bits,
                  'pixel_format_hex':f'0x{info.format:08x}','frame_id':int(info.frame_id),
                  'timestamp_ticks':int(info.timestamp),'timestamp_ns':self.buffer_value(hbuf,28,C.c_uint64,False),
                  'filled_bytes':filled,'xpadding_bytes':xpad,'image_offset_bytes':offset,
                  'incomplete':bool(incomplete),'copy_wait_decode_ms':(time.perf_counter()-t)*1000,
                  'host_receive_monotonic_ns':time.monotonic_ns()}
            return arr,meta
        finally:self.check(self.dll.scap_back_buffer(self.handle,hbuf),'return frame buffer')

    def fresh(self):
        """Continuous-mode fallback, NOT a hardware-trigger latency guarantee.

        Drains all 4 SDK buffers plus two frames after the caller's SLM wait.
        Frame timestamp/ID are logged; exact trigger sync needs optical testing.
        """
        for _ in range(int(self.config.get('buffer_count',4))+2):self.grab()
        return self.grab()

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
