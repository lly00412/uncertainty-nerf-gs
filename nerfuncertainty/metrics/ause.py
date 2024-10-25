import torch
import numpy as np

# original code from https://github.com/poetrywanderer/CF-NeRF/blob/66918a9748c137e1c0242c12be7aa6efa39ece06/run_nerf_helpers.py#L382


def ause(unc_vec, err_vec, err_type="rmse"):
    ratio_removed = np.linspace(0, 1, 100, endpoint=False)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec)

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        err_slice = err_vec_sorted[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "rmse":
            ause_err.append(torch.sqrt(err_slice.mean()).cpu().numpy())
        elif err_type == "mae" or err_type == "mse":
            ause_err.append(err_slice.mean().cpu().numpy())

    ###########################################

    # Sort by variance
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]
    ause_err_by_var = np.zeros(len(ratio_removed))
    for i, r in enumerate(ratio_removed):
        err_slice = err_vec_sorted_by_var[0 : int((1 - r) * n_valid_pixels)]
        if err_type == "rmse":
            ause_err_by_var[i] = torch.sqrt(err_slice.mean()).cpu().numpy()
        elif err_type == "mae" or err_type == "mse":
            ause_err_by_var[i] = err_slice.mean().cpu().numpy()

    # Normalize and append
    max_val = max(max(ause_err), max(ause_err_by_var))
    ause_err = ause_err / max_val
    ause_err = np.array(ause_err)

    ause_err_by_var = ause_err_by_var / max_val
    ause_err_by_var = np.array(ause_err_by_var)
    ause = np.trapz(ause_err_by_var - ause_err, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause

def auc(unc_vec, err_vec, vc_unc_vec=None, err_type="rmse"):
    intervals = 100
    quants = [1. / intervals * t for t in range(1, intervals + 1)]

    # Sort the error by err
    opt_thres = [torch.quantile(err_vec, q) for q in quants]
    opt_subs = [err_vec <= t for t in opt_thres]
    if err_type == "rmse":
        ause_err = torch.tensor([torch.sqrt(err_vec[s].mean()).item() if s.any() else 0.0 for s in opt_subs])
    elif err_type == "mae" or err_type == "mse":
        ause_err = torch.tensor([err_vec[s].mean().item() if s.any() else 0.0 for s in opt_subs])

    # Sort the error by var
    var_thres = [torch.quantile(unc_vec, q) for q in quants]
    var_subs = [unc_vec <= t for t in var_thres]
    if err_type == "rmse":
        ause_err_by_var = torch.tensor([torch.sqrt(err_vec[s].mean()).item() if s.any() else 0.0 for s in var_subs])
    elif err_type == "mae" or err_type == "mse":
        ause_err_by_var = torch.tensor([err_vec[s].mean().item() if s.any() else 0.0 for s in var_subs])

    # Sort the error by vc var
    if vc_unc_vec is not None:
        vc_var_thres = [torch.quantile(vc_unc_vec, q) for q in quants]
        vc_var_subs = [vc_unc_vec <= t for t in vc_var_thres]
        if err_type == "rmse":
            ause_err_by_vc_var = torch.tensor([torch.sqrt(err_vec[s]).mean().item() if s.any() else 0.0 for s in vc_var_subs])
        elif err_type == "mae" or err_type == "mse":
            ause_err_by_vc_var = torch.tensor([err_vec[s].mean().item() if s.any() else 0.0 for s in vc_var_subs])

    # Normalize and append
    if vc_unc_vec is None:
        max_val = max(max(ause_err), max(ause_err_by_var))
        ause_err = ause_err / max_val
        ause_err_by_var = ause_err_by_var / max_val

        auc_by_var = torch.trapz(ause_err_by_var, dx=1.0 / intervals).cpu().numpy()
        auc_by_opt = torch.trapz(ause_err, dx=1.0 / intervals).cpu().numpy()
        auc_by_vc_var = None
        ause_err_by_vc_var = None
        ause_err = ause_err.cpu().numpy()
        ause_err_by_var = ause_err_by_var.cpu().numpy()
    else:
        max_val = max(max(ause_err), max(ause_err_by_var), max(ause_err_by_vc_var))
        ause_err = ause_err / max_val
        ause_err_by_var = ause_err_by_var / max_val
        ause_err_by_vc_var = ause_err_by_vc_var / max_val

        auc_by_var = torch.trapz(ause_err_by_var, dx=1.0 / intervals).cpu().numpy()
        auc_by_opt = torch.trapz(ause_err, dx=1.0 / intervals).cpu().numpy()
        auc_by_vc_var = torch.trapz(ause_err_by_vc_var,  dx=1.0 / intervals).cpu().numpy()

        ause_err = ause_err.cpu().numpy()
        ause_err_by_var = ause_err_by_var.cpu().numpy()
        ause_err_by_vc_var = ause_err_by_vc_var.cpu().numpy()

    return quants, ause_err, ause_err_by_var, ause_err_by_vc_var, auc_by_opt, auc_by_var,auc_by_vc_var

