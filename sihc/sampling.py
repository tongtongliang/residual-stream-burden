import torch


@torch.no_grad()
def _model_to_velocity(model, z, t, labels, prediction, t_eps):
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    # reduce-overhead compile mode may reuse its CUDA Graph output buffer on
    # the next invocation. CFG needs both conditional and unconditional
    # outputs alive at once, so materialize an owning tensor outside compile.
    pred = model(z, t.flatten(), labels).clone()
    if prediction == "clean":
        return (pred - z) / (1.0 - t).clamp_min(t_eps)
    if prediction == "velocity":
        return pred
    raise ValueError(f"unknown prediction target: {prediction}")


@torch.no_grad()
def cfg_velocity(model, z, t, labels, *, num_classes, cfg_scale=1.0, interval=(0.0, 1.0), t_eps=5e-2, prediction="clean"):
    v_cond = _model_to_velocity(model, z, t, labels, prediction, t_eps)
    if cfg_scale == 1.0:
        return v_cond
    null_labels = torch.full_like(labels, num_classes)
    v_uncond = _model_to_velocity(model, z, t, null_labels, prediction, t_eps)
    low, high = interval
    mask = (t < high) & ((low == 0) | (t > low))
    scale = torch.where(mask, torch.full_like(t, cfg_scale), torch.ones_like(t))
    return v_uncond + scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_heun(
    model,
    labels,
    *,
    image_size=256,
    noise_scale=1.0,
    steps=50,
    cfg_scale=1.0,
    interval=(0.0, 1.0),
    num_classes=1000,
    t_eps=5e-2,
    prediction="clean",
):
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    bsz = labels.numel()
    device = labels.device
    z = torch.randn(bsz, 3, image_size, image_size, device=device) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps - 1):
        t = torch.full((bsz, 1, 1, 1), float(ts[i]), device=device)
        t_next = torch.full((bsz, 1, 1, 1), float(ts[i + 1]), device=device)
        v = cfg_velocity(model, z, t, labels, num_classes=num_classes, cfg_scale=cfg_scale, interval=interval, t_eps=t_eps, prediction=prediction)
        z_euler = z + (t_next - t) * v
        v_next = cfg_velocity(model, z_euler, t_next, labels, num_classes=num_classes, cfg_scale=cfg_scale, interval=interval, t_eps=t_eps, prediction=prediction)
        z = z + (t_next - t) * 0.5 * (v + v_next)
    t = torch.full((bsz, 1, 1, 1), float(ts[-2]), device=device)
    t_next = torch.full((bsz, 1, 1, 1), float(ts[-1]), device=device)
    v = cfg_velocity(model, z, t, labels, num_classes=num_classes, cfg_scale=cfg_scale, interval=interval, t_eps=t_eps, prediction=prediction)
    return z + (t_next - t) * v


@torch.no_grad()
def sample_euler(
    model,
    labels,
    *,
    image_size=256,
    noise_scale=1.0,
    steps=100,
    cfg_scale=1.0,
    interval=(0.0, 1.0),
    num_classes=1000,
    t_eps=5e-2,
    prediction="clean",
):
    """Sample the forward flow with uniform-time explicit Euler updates."""
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    bsz = labels.numel()
    device = labels.device
    z = torch.randn(bsz, 3, image_size, image_size, device=device) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)

    for i in range(steps):
        t = ts[i].reshape(1, 1, 1, 1).expand(bsz, -1, -1, -1)
        dt = ts[i + 1] - ts[i]
        velocity = cfg_velocity(
            model,
            z,
            t,
            labels,
            num_classes=num_classes,
            cfg_scale=cfg_scale,
            interval=interval,
            t_eps=t_eps,
            prediction=prediction,
        )
        z = z + dt * velocity

    return z


def to_uint8(x):
    x = (x.detach().float().clamp(-1, 1) + 1.0) / 2.0
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)
