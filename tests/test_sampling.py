import pytest
import torch
from sihc.sampling import cfg_velocity, sample_euler, sample_heun


class ConstantVelocity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, z, t, labels):
        self.calls.append(t.detach().clone())
        return torch.full_like(z, self.value)


class TimeVelocity(torch.nn.Module):
    def forward(self, z, t, labels):
        return t[:, None, None, None].expand_as(z)


class LabelVelocity(torch.nn.Module):
    def forward(self, z, t, labels):
        return labels[:, None, None, None].to(z.dtype).expand_as(z)


def test_cfg_interval_remains_strict_at_both_boundaries():
    z = torch.zeros(3, 3, 2, 2)
    labels = torch.zeros(3, dtype=torch.long)
    t = torch.tensor([0.25, 0.5, 0.75])[:, None, None, None]
    velocity = cfg_velocity(
        LabelVelocity(),
        z,
        t,
        labels,
        num_classes=1,
        cfg_scale=2.0,
        interval=(0.25, 0.75),
        prediction="velocity",
    )
    expected = torch.tensor([0.0, -1.0, 0.0])[:, None, None, None].expand_as(z)
    torch.testing.assert_close(velocity, expected)


def test_euler_constant_velocity_integrates_noise_to_data():
    torch.manual_seed(789)
    actual = sample_euler(
        ConstantVelocity(2.0),
        torch.zeros(1, dtype=torch.long),
        image_size=2,
        steps=7,
        cfg_scale=1.0,
        num_classes=1,
        prediction="velocity",
    )
    torch.manual_seed(789)
    expected = torch.randn(1, 3, 2, 2) + 2.0
    torch.testing.assert_close(actual, expected)


def test_euler_uses_one_fresh_velocity_per_step_on_uniform_grid():
    model = ConstantVelocity(0.0)
    torch.manual_seed(123)
    sample_euler(
        model,
        torch.zeros(1, dtype=torch.long),
        image_size=2,
        steps=9,
        cfg_scale=1.0,
        num_classes=1,
        prediction="velocity",
    )
    # Unit CFG returns after the conditional model call.
    assert len(model.calls) == 9
    evaluated_times = torch.stack([call[0] for call in model.calls])
    reference_times = torch.linspace(0.0, 1.0 - 1.0 / 9, 9)
    torch.testing.assert_close(evaluated_times, reference_times)


def test_euler_rejects_nonpositive_step_count():
    with pytest.raises(ValueError, match="steps must be positive"):
        sample_euler(
            ConstantVelocity(0.0),
            torch.zeros(1, dtype=torch.long),
            image_size=2,
            steps=0,
            prediction="velocity",
        )



@pytest.mark.parametrize('steps', [1, 5])
def test_heun_constant_velocity_and_evaluation_count(steps):
    labels = torch.zeros(1, dtype=torch.long)
    torch.manual_seed(12)
    noise = torch.randn(1, 3, 2, 2)
    torch.manual_seed(12)
    model = ConstantVelocity(2.0)
    result = sample_heun(model, labels, image_size=2, steps=steps, prediction='velocity')
    torch.testing.assert_close(result, noise + 2.0)
    assert len(model.calls) == 2 * steps - 1


def test_heun_rejects_zero_steps():
    with pytest.raises(ValueError, match='positive'):
        sample_heun(ConstantVelocity(1.0), torch.zeros(1, dtype=torch.long), steps=0)
