import unittest

import torch

from verl.utils.torch_functional import get_constant_schedule_with_warmup


class ConstantWarmupSchedulerTest(unittest.TestCase):
    def test_nonpositive_warmup_updates_on_first_optimizer_step(self):
        for num_warmup_steps in (0, -1):
            with self.subTest(num_warmup_steps=num_warmup_steps):
                parameter = torch.nn.Parameter(torch.tensor([1.0]))
                base_lr = 0.1
                optimizer = torch.optim.AdamW(
                    [parameter], lr=base_lr, weight_decay=0.0
                )
                scheduler = get_constant_schedule_with_warmup(
                    optimizer, num_warmup_steps=num_warmup_steps
                )

                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], base_lr)

                before = parameter.detach().clone()
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()

                self.assertFalse(torch.equal(parameter.detach(), before))
                self.assertGreater(
                    torch.count_nonzero(optimizer.state[parameter]["exp_avg"]).item(),
                    0,
                )

                scheduler.step()
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], base_lr)

    def test_positive_warmup_lr_sequence_is_unchanged(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        base_lr = 0.1
        optimizer = torch.optim.AdamW([parameter], lr=base_lr)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=2
        )

        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(3):
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        expected = [0.0, 0.5 * base_lr, base_lr, base_lr]
        for actual_lr, expected_lr in zip(learning_rates, expected):
            self.assertAlmostEqual(actual_lr, expected_lr)


if __name__ == "__main__":
    unittest.main()
