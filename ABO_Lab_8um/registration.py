"""Portable exact 478-grid registration patterns (no SDK/model dependencies).

Same formulas as hardware_sdk/generators/dual_slm_alignment.py and
dual_slm_registration_sweep.py; checked bit-for-bit in test_phase_encoding.py.
"""
import numpy as np


def logical_pairs():
    y,x=np.indices((478,478));vx=((x%8)>=4).astype(np.uint8)*128;vy=((y%8)>=4).astype(np.uint8)*128
    pairs=[]
    for name,cell in [('01_check64',64),('04_check16',16)]:
        a=((x//cell+y//cell)%2*255).astype(np.uint8)
        phase=np.where(((x//cell+y//cell)//2)%2==0,vx,vy)
        phase[(x%cell==0)|(y%cell==0)|(a==0)]=0
        pairs.append((name,a,phase,cell,'alternating x/y in visible white cells'))
    shapes=[[(0,0),(0,1),(1,0),(1,1)],[(r,c) for r in range(2) for c in range(4,7)],
            [(3,0),(4,0),(5,0),(5,1),(5,2)],[(r,c) for r in range(3,6) for c in range(4,7)],
            [(r,c) for r in range(7,9) for c in range(3)],[(7,5),(7,6),(8,5),(8,6)]]
    cells=np.zeros((9,9),np.uint8)
    for shape in shapes:
        for r,c in shape:cells[r,c]=255
    a=np.zeros((478,478),np.uint8);a[23:455,23:455]=np.repeat(np.repeat(cells,48,axis=0),48,axis=1)
    for name,axis,grating in [('02_blocks_x','x',vx),('03_blocks_y','y',vy)]:
        phase=grating.copy();phase[a==0]=0
        pairs.append((name,a,phase,48,axis))
    return sorted(pairs,key=lambda item:item[0])
