import numpy as np
import torch

_EPS = 1e-12

def empirical_correlation(x, y, tau: float) -> float:
    """Compute the empirical correlation between two lists of numbers.

    Args:
        x (list[float]): First list of numbers.
        y (list[float]): Second list of numbers.
        tau (float): Time lag to apply to the second list.

    Returns:
        float: Empirical correlation between x and y.
    """
    x = np.array(x)
    y = np.array(y)

    if tau > 0:
        x_tau = x[:-tau]
        y_tau = y[tau:]
        x_stand = (x_tau - np.mean(x_tau)) / np.std(x_tau)
        y_stand = (y_tau - np.mean(y_tau)) / np.std(y_tau)
        correlation = np.mean(x_stand * y_stand)
    elif tau < 0:
        tau = -tau
        x_tau = x[tau:]
        y_tau = y[:-tau]
        x_stand = (x_tau - np.mean(x_tau)) / np.std(x_tau)
        y_stand = (y_tau - np.mean(y_tau)) / np.std(y_tau)
        correlation = np.mean(x_stand * y_stand)
    else:
        x_stand = (x - np.mean(x)) / np.std(x)
        y_stand = (y - np.mean(y)) / np.std(y)
        correlation = np.mean(x_stand * y_stand)
    # Notice that the above, E_xy, is equivalent to the correlation because we have unitary variance

    return correlation

def list_lagged_correlation(x, y, max_lag: int) -> list:
    """Compute the lagged correlation between two lists of numbers.

    Args:
        x (list[float]): First list of numbers.
        y (list[float]): Second list of numbers.
        max_lag (int): Maximum lag to compute.

    Returns:
        list[float]: List of lagged correlations.
    """
    correlations = []
    for tau in range(-max_lag,max_lag + 1):
        corr = empirical_correlation(x, y, tau)
        correlations.append(corr)
    return correlations

def grokking_test1(test_acc):
    if test_acc[-1] > 0.95:
        return True
    else:
        return False

def grokking_test2(train_acc, test_acc):
    dummy = False
    for i in range(len(train_acc)):
        if train_acc[i] > 0.9 and test_acc[i] < 0.5:
            dummy = True
    if dummy:
        if test_acc[-1] > 0.9:
            return True
    return False

def grokking_test3(train_acc, test_acc):
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    level = (train_acc - test_acc).mean() # Riemann sum approximation of area between curves
    mask = (test_acc[-1] > 0.9) and (train_acc[-1] > 0.9)
    return level if mask else 0


def _trainable_params(model):
    return [param for param in model.parameters() if param.requires_grad]


def _flatten_tensors(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _flat_grad(loss, params, create_graph=False, retain_graph=False):
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    return _flatten_tensors(grads)


@torch.no_grad()
def _add_vector_(params, vector, scale=1.0):
    offset = 0
    for param in params:
        numel = param.numel()
        param.add_(scale * vector[offset:offset + numel].view_as(param))
        offset += numel


def _sam_sharpness(model, loss_fn, x, y, rho):
    params = _trainable_params(model)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grad = _flat_grad(loss, params)
    grad_norm = grad.norm()

    if grad_norm <= _EPS:
        return {"flatness/sharpness_sam": 0.0}

    epsilon = rho * grad / (grad_norm + _EPS)
    _add_vector_(params, epsilon)
    perturbed_loss = loss_fn(model(x), y)
    _add_vector_(params, epsilon, scale=-1.0)

    return {"flatness/sharpness_sam": (perturbed_loss.detach() - loss.detach()).item()}


def _hvp(loss, params, vector):
    grad = _flat_grad(loss, params, create_graph=True, retain_graph=True)
    directional_grad = torch.dot(grad, vector)
    hvp = torch.autograd.grad(directional_grad, params, retain_graph=True, allow_unused=False)
    return _flatten_tensors(hvp)


def _hutchinson_trace(model, loss_fn, x, y, n_probes):
    params = _trainable_params(model)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)

    num_params = sum(param.numel() for param in params)
    device = params[0].device
    dtype = params[0].dtype
    trace_estimate = torch.zeros((), device=device, dtype=dtype)

    for _ in range(n_probes):
        probe = torch.empty(num_params, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1)
        trace_estimate += torch.dot(probe, _hvp(loss, params, probe))

    return {"flatness/trace_hutchinson": (trace_estimate / n_probes).detach().item()}


def compute_flatness_metrics(model, loss_fn, x, y, sam_rho=1e-3, hutchinson_probes=2):
    params = _trainable_params(model)
    if not params:
        return {}

    was_training = model.training
    model.eval()
    try:
        metrics = {}
        if sam_rho > 0:
            metrics.update(_sam_sharpness(model, loss_fn, x, y, sam_rho))
        if hutchinson_probes > 0:
            metrics.update(_hutchinson_trace(model, loss_fn, x, y, hutchinson_probes))
        return metrics
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
