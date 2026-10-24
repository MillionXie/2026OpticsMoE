"""Frame counts and hypothetical throughput estimates; never claims measured FPS."""
import argparse
import json


def frame_budget(test_per_task=1000, train_per_current=1000, old_memory=300):
    # Ten cells per architecture: 1+2+3+4. Three/two CCD stages per input.
    evaluation = 10*test_per_task*(3+2)
    moe_train = sum(train_per_current+(s-1)*old_memory for s in range(1,5))*3
    d2nn_train = sum(train_per_current+(s-1)*old_memory for s in range(1,5))*2
    return dict(evaluation_frames=evaluation,
                moe_TRAIN_frames=moe_train,d2nn_TRAIN_frames=d2nn_train,
                total_with_proposed_TRAIN=evaluation+moe_train+d2nn_train,
                final_D_only_evaluation_frames=4*test_per_task*(3+2),
                caveat='Counts assume fixed admitted upstream per phase; changed upstream requires new descendant frames. TRAIN sampling budget is a proposal, not a claim of original full replay.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--seconds-per-frame',type=float)
    a=p.parse_args();r=frame_budget()
    if a.seconds_per_frame is not None:
        if a.seconds_per_frame<=0:p.error('Throughput must be positive')
        r['hypothetical_hours']={k:v*a.seconds_per_frame/3600 for k,v in r.items() if k.endswith('frames') or k=='total_with_proposed_TRAIN'}
        r['throughput_status']='hypothetical input, not measured on this new protocol'
    print(json.dumps(r,indent=2))
