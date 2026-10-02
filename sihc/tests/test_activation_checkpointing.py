from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from sihc.activation_checkpointing import configure_activation_checkpointing


class TinyBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x) + self.mlp(x)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = TinyBranch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.branch(x)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ActivationCheckpointingTest(unittest.TestCase):
    def _assert_mode_matches_unwrapped(self, mode: str) -> None:
        torch.manual_seed(123)
        reference = TinyModel()
        candidate = copy.deepcopy(reference)
        original_keys = tuple(candidate.state_dict())
        original_param_ids = {id(param) for param in candidate.parameters()}

        wrapped = configure_activation_checkpointing(candidate, mode)
        self.assertEqual(wrapped, 2)
        self.assertEqual(tuple(candidate.state_dict()), original_keys)
        self.assertEqual({id(param) for param in candidate.parameters()}, original_param_ids)

        x_reference = torch.randn(3, 4, requires_grad=True)
        x_candidate = x_reference.detach().clone().requires_grad_(True)
        reference(x_reference).square().mean().backward()
        candidate(x_candidate).square().mean().backward()

        torch.testing.assert_close(x_candidate.grad, x_reference.grad)
        reference_grads = dict(reference.named_parameters())
        candidate_grads = {
            name.replace("_checkpoint_wrapped_module.", ""): parameter
            for name, parameter in candidate.named_parameters()
        }
        self.assertEqual(set(candidate_grads), set(reference_grads))
        for name, parameter in reference_grads.items():
            torch.testing.assert_close(candidate_grads[name].grad, parameter.grad)

    def test_mlp_checkpoint_preserves_state_and_gradients(self) -> None:
        self._assert_mode_matches_unwrapped("mlp")

    def test_branch_checkpoint_preserves_state_and_gradients(self) -> None:
        self._assert_mode_matches_unwrapped("branch")

    def test_none_is_noop(self) -> None:
        model = TinyModel()
        self.assertEqual(configure_activation_checkpointing(model, "none"), 0)


if __name__ == "__main__":
    unittest.main()
