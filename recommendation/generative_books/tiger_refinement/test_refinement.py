"""Focused checks for recovery scheduling and nonfinite-gradient protection."""
import unittest
import torch
from train_refinement import clip_checked, lr_factor, numerical_report


class Tests(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertAlmostEqual(lr_factor(0, 207630, 1000), .1)
        self.assertAlmostEqual(lr_factor(1000, 207630, 1000), 1.)
        self.assertAlmostEqual(lr_factor(207630, 207630, 1000), .1)
        self.assertAlmostEqual(lr_factor(207631, 207630, 1000), .1)

    def test_optimizer_scheduler_resume_equivalence(self):
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_factor(step, 20, 3))
        x = torch.tensor([[1., 2., 3.]])
        def step(m, o, s):
            o.zero_grad()
            m(x).square().sum().backward()
            clip_checked(m)
            o.step()
            s.step()
        for _ in range(5):
            step(model, opt, scheduler)
        import copy
        state = copy.deepcopy((model.state_dict(), opt.state_dict(), scheduler.state_dict()))
        for _ in range(3):
            step(model, opt, scheduler)
        restored = torch.nn.Linear(3, 1)
        resumed_opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_opt, lambda n: lr_factor(n, 20, 3))
        restored.load_state_dict(state[0])
        resumed_opt.load_state_dict(state[1])
        resumed_scheduler.load_state_dict(state[2])
        for _ in range(3):
            step(restored, resumed_opt, resumed_scheduler)
        for original, recovered in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(original, recovered, rtol=0, atol=0)
        self.assertEqual(opt.param_groups[0]["lr"], resumed_opt.param_groups[0]["lr"])

    def test_nonfinite_is_rejected_before_gradient_mutation(self):
        model = torch.nn.Linear(2, 1)
        model.weight.grad = torch.tensor([[float("inf"), 3.]])
        before = model.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            clip_checked(model)
        self.assertTrue(torch.isinf(model.weight.grad[0, 0]))
        self.assertEqual(model.weight.grad[0, 1], 3.)
        torch.testing.assert_close(model.weight, before)
        report = numerical_report(model)[0]
        self.assertEqual(report["gradient_nonfinite"], 1)

    def test_finite_gradients_are_clipped(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.tensor([[3., 4.]])
        self.assertAlmostEqual(float(clip_checked(model)), 5.)
        self.assertLessEqual(float(model.weight.grad.norm()), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
