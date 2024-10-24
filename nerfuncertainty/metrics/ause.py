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
    ratio_removed = np.linspace(0, 1, 100+1, endpoint=True)
    # Sort the error
    err_vec_sorted, _ = torch.sort(err_vec) # sorted accending

    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ause_err = []
    for r in ratio_removed:
        if r == 0:
            err_slice = torch.zeros(1)
        else:
            err_slice = err_vec_sorted[: int(r * n_valid_pixels)]
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
        if r == 0:
            err_slice = torch.zeros(1)
        else:
            err_slice = err_vec_sorted_by_var[: int(r * n_valid_pixels)]
        if err_type == "rmse":
            ause_err_by_var[i] = torch.sqrt(err_slice.mean()).cpu().numpy()
        elif err_type == "mae" or err_type == "mse":
            ause_err_by_var[i] = err_slice.mean().cpu().numpy()

    if vc_unc_vec is not None:
        # Sorted by vcurf
        _, vc_var_vec_sorted_idxs = torch.sort(vc_unc_vec)
        # Sort error by variance
        err_vec_sorted_by_vc_var = err_vec[vc_var_vec_sorted_idxs]
        ause_err_by_vc_var = np.zeros(len(ratio_removed))
        for i, r in enumerate(ratio_removed):
            if r == 0:
                err_slice = torch.zeros(1)
            else:
                err_slice = err_vec_sorted_by_vc_var[: int(r * n_valid_pixels)]
            if err_type == "rmse":
                ause_err_by_vc_var[i] = torch.sqrt(err_slice.mean()).cpu().numpy()
            elif err_type == "mae" or err_type == "mse":
                ause_err_by_vc_var[i] = err_slice.mean().cpu().numpy()

    # Normalize and append
    if vc_unc_vec is None:
        max_val = max(max(ause_err), max(ause_err_by_var))
        ause_err = ause_err / max_val
        ause_err = np.array(ause_err)

        ause_err_by_var = ause_err_by_var / max_val
        ause_err_by_var = np.array(ause_err_by_var)
        auc_by_var = np.trapz(ause_err_by_var, ratio_removed)
        auc_by_opt = np.trapz(ause_err, ratio_removed)
        auc_by_vc_var = None
        ause_err_by_vc_var = None
    else:
        max_val = max(max(ause_err), max(ause_err_by_var), max(ause_err_by_vc_var))
        ause_err = ause_err / max_val
        ause_err = np.array(ause_err)

        ause_err_by_var = ause_err_by_var / max_val
        ause_err_by_var = np.array(ause_err_by_var)

        ause_err_by_vc_var = ause_err_by_vc_var / max_val
        ause_err_by_vc_var = np.array(ause_err_by_vc_var)

        auc_by_var = np.trapz(ause_err_by_var, ratio_removed)
        auc_by_opt = np.trapz(ause_err, ratio_removed)
        auc_by_vc_var = np.trapz(ause_err_by_vc_var, ratio_removed)
    return ratio_removed, ause_err, ause_err_by_var, ause_err_by_vc_var, auc_by_opt, auc_by_var,auc_by_vc_var

