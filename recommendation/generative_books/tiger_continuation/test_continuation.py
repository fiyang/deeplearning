import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from model_factory import create_wide_model
from train_continuation import clip_checked, lr_factor


def module_at(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root=Path(__file__).resolve().parents[1]
base=module_at('books_original',root/'models.py')
growth=module_at('books_growth_reference',root/'tiger_growth/growth.py')


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.codes=np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]])
        self.h=torch.tensor([[1,2,0],[2,3,0]])
        self.u,self.y=torch.tensor([0,17]),torch.tensor([2,3])

    def test_strict_parent_load_keeps_predictions_and_width(self):
        parent=growth.expand_tiger(base.create_model('tiger',self.codes)).eval()
        model=create_wide_model(self.codes,base.create_model).eval()
        model.load_state_dict(parent.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()),14281344)
        self.assertEqual(model.net.config.d_model,256)
        self.assertEqual(model.net.config.d_ff,2048)
        self.assertIs(model.net.shared.weight,model.net.lm_head.weight)
        torch.testing.assert_close(model(self.h,self.u,self.y),parent(self.h,self.u,self.y),rtol=0,atol=0)
        seen=torch.zeros(2,4,dtype=torch.bool)
        a,sa=parent.recommend(self.h,self.u,seen,20)
        b,sb=model.recommend(self.h,self.u,seen,20)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(sa,sb,rtol=0,atol=0)

    def test_optimizer_scheduler_and_rng_resume_match_next_update(self):
        def setup():
            model=create_wide_model(self.codes,base.create_model).train()
            opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:lr_factor(n,20,3))
            return model,opt,sched
        def update(model,opt,sched):
            opt.zero_grad()
            model(self.h,self.u,self.y).backward()
            clip_checked(model)
            opt.step(); sched.step()
        model,opt,sched=setup()
        update(model,opt,sched)
        state=copy.deepcopy((model.state_dict(),opt.state_dict(),sched.state_dict()))
        rng=torch.get_rng_state().clone()
        update(model,opt,sched)
        restored,ro,rs=setup()
        restored.load_state_dict(state[0]); ro.load_state_dict(state[1]); rs.load_state_dict(state[2])
        torch.set_rng_state(rng)
        update(restored,ro,rs)
        for a,b in zip(model.parameters(),restored.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(opt.param_groups[0]['lr'],ro.param_groups[0]['lr'])


if __name__=='__main__':
    unittest.main(verbosity=2)
