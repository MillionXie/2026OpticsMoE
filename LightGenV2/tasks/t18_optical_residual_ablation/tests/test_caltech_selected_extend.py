"""CPU continuation correctness: exact resumed updates, not a fresh optimizer."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import caltech_selected_extend as e
torch=e.torch


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.raw_phase=torch.nn.Parameter(torch.tensor([.1,-.2,.3]))


def setup():
    model=Tiny();ema=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=8,factor=.5,min_lr=2e-5)
    return model,ema,optimizer,scheduler


def step(objects,index):
    model,ema,optimizer,scheduler=objects;optimizer.zero_grad(set_to_none=True)
    target=torch.tensor([.2,-.1,.4])+index*.00001
    (model.raw_phase-target).square().sum().backward();optimizer.step()
    with torch.no_grad():ema.raw_phase.lerp_(model.raw_phase,.05)
    scheduler.step(2.+(index%3)*.01)


class Continuation(unittest.TestCase):
    def test_resume_matches_uninterrupted_next_update_and_ema(self):
        original=setup()
        for i in range(100):step(original,i)
        checkpoint=copy.deepcopy(dict(model=original[0].state_dict(),ema=original[1].state_dict(),
            optimizer=original[2].state_dict(),scheduler=original[3].state_dict()))
        restored=setup();identity=e.restore_training(*restored,checkpoint)
        self.assertEqual(identity,{k:e.state_digest(v) for k,v in checkpoint.items()})
        step(original,100);step(restored,100)
        for a,b in zip(original,restored):self.assertEqual(e.state_digest(a.state_dict()),e.state_digest(b.state_dict()))

    def test_reset_optimizer_is_detectably_different(self):
        original=setup()
        for i in range(100):step(original,i)
        reset=setup();reset[0].load_state_dict(original[0].state_dict());reset[1].load_state_dict(original[1].state_dict())
        step(original,100);step(reset,100)
        self.assertNotEqual(e.state_digest(original[0].state_dict()),e.state_digest(reset[0].state_dict()))

    def test_config_preserves_regularization_and_all_nonbudget_settings(self):
        parent=e.read(e.parent_tools.PROFILE)
        parent.update(requested_epochs=100,early_stopping=False)
        extended=e.extension_config(parent)
        changed={k for k in parent if parent[k]!=extended[k]}
        self.assertLessEqual(changed,{'maximum_epochs','minimum_epochs','requested_epochs','selection','maximum_seconds','early_stopping'})
        self.assertEqual(parent['mixup_alpha'],extended['mixup_alpha'])
        self.assertEqual(parent['augmentation'],extended['augmentation'])
        self.assertEqual(extended['requested_epochs'],200)

    def test_new_epoch_mixup_is_independent_and_not_replayed(self):
        original=e.parent_tools.expected_mixup_hash(17,100,1372,.2)
        new=e.parent_tools.expected_mixup_hash(17,101,1372,.2)
        self.assertNotEqual(original,new)
        digest=e.hashlib.sha256()
        for batch,start in enumerate(range(0,1372,16)):
            _,_,identity=e.pilot.mixup_schedule(17,101,batch,min(16,1372-start),.2);digest.update(identity.encode())
        self.assertEqual(digest.hexdigest(),new)


if __name__=='__main__':unittest.main()
