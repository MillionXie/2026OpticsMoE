"""Reversible DISPLAY CONFIGURATION only; no app/window manipulation.

The observed HDMI panel can be vertically clipped with monitor origin Y != 0.
Opt-in position-only diagnostic: leave primary/resolution/Hz/X untouched and
restore original position even on failure. Never persist to Windows registry.
"""
import time

def choose_panel(rows):
    found=[r for r in rows if any('FNR0002' in s for s in r['ids'])]
    if len(found)!=1:raise RuntimeError('Expected exactly one FNR0002 phase monitor')
    r=found[0]
    if r['flags']&4 or (r['width'],r['height'],r['hz'])!=(1920,1200,60):
        raise RuntimeError('Phase monitor must be non-primary 1920x1200 at 60 Hz')
    return r

class DisplayOrigin:
    def __init__(self,enabled=False):self.enabled=enabled;self.changed=False;self.audit={}
    def inventory(self):
        rows=[]
        for i in range(32):
            try:d=self.api.EnumDisplayDevices(None,i)
            except Exception:break
            if not d.StateFlags&1:continue
            m=self.api.EnumDisplaySettings(d.DeviceName,self.con.ENUM_CURRENT_SETTINGS);ids=[]
            for j in range(8):
                try:ids.append(self.api.EnumDisplayDevices(d.DeviceName,j).DeviceID)
                except Exception:break
            rows.append(dict(device=d.DeviceName,flags=d.StateFlags,ids=ids,width=m.PelsWidth,
                height=m.PelsHeight,hz=m.DisplayFrequency,bits=m.BitsPerPel,x=m.Position_x,y=m.Position_y))
        return rows
    def __enter__(self):
        if not self.enabled:return self
        import win32api,win32con
        self.api=win32api;self.con=win32con
        before=self.inventory();r=choose_panel(before);self.device=r['device']
        self.audit={'before':before,'target_y':0,'registry_persisted':False}
        self.original=self.api.EnumDisplaySettings(self.device,self.con.ENUM_CURRENT_SETTINGS)
        if r['y']==0:return self
        m=self.api.EnumDisplaySettings(self.device,self.con.ENUM_CURRENT_SETTINGS)
        m.Position_y=0;m.Fields=self.con.DM_POSITION
        if self.api.ChangeDisplaySettingsEx(self.device,m,self.con.CDS_TEST)!=0:raise RuntimeError('Phase display position test failed')
        if self.api.ChangeDisplaySettingsEx(self.device,m,0)!=0:raise RuntimeError('Phase display position apply failed')
        self.changed=True
        try:
            time.sleep(1);after=self.inventory();expected=[dict(x,y=0) if x['device']==self.device else x for x in before]
            self.audit['during']=after
            if after!=expected:raise RuntimeError('Unexpected display settings change')
        except BaseException:self.close();raise
        return self
    def close(self):
        if self.changed:
            self.original.Fields=self.con.DM_POSITION
            ret=self.api.ChangeDisplaySettingsEx(self.device,self.original,0)
            self.audit.update(restore_return=ret,after_restore=self.inventory())
            self.audit['restored_exactly']=self.audit['after_restore']==self.audit['before']
            if ret!=0 or not self.audit['restored_exactly']:raise RuntimeError('Could not restore phase display position; inspect Windows display settings')
            self.changed=False
    def __exit__(self,*args):self.close()
