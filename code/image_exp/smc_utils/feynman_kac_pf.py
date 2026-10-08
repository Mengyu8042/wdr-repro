import torch
import numpy as np 
from .smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights


def smc_FK(M, G, resample_strategy, T, P,  ess_threshold, 
           verbose=False, verbose_notebook=False, log_xt_trace=False, extra_vals=None, vae_path=None, R=1):
    """smc_FK is sequential Monte Carlo in the Feynman-Kac formulation.

    This is mean to exactly mirror the formulation in Chopin's book, except stated in reverse time.

    M & G have an extra input & output, "extra_vals".  This allows the possibility of saving redundant
    computation across M and G at each step.

    Args:
        M: intial/transition distribution.
            x_T, extra_vals = M(T, None, extra_vals, P=P) # Initial sample x_T ~ M_0(dt)
            x_t, extra_vals = M(t, x_{t+1}, extra_vals) if t<T
        G: potential function
            w_T, extra_vals = G(T, None, x_T, extra_vals) # Initial potential
            w_t, extra_vals = G(t, x_{t+1}, x_t, extra_vals) # subsequent potentials
        resample: resampling function
            log_w_t, resample_indices = resample(log_w_t) # performs resampling step (or not), resets weights
            to 0 after resamplings.
        T: final time step
        P: number of particles
        R: expansion factor for multi-descendant propagation (default 1)

    Returns:
        x_ts (shape [T, P, ...]), log_w, resample_indices_trace, ess_trace
    """
    resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold, verbose=False, vae_path=vae_path)
    
    if log_xt_trace:
        xt_trace = [] 
    else: xt_trace = None 
    resample_indices_trace = []
    ess_trace = []
    log_w_trace = [] 
    xtp1 = None
    extra_vals = extra_vals if extra_vals else {} 
    log_wtp1 = 0 
    
    if verbose_notebook:
        from tqdm.notebook import trange
        time_range = trange(T, -1, -1)
    else:
        time_range = range(T, -1, -1)
        if verbose:
            from tqdm.auto import tqdm
            time_range = tqdm(time_range)

    original_P = P

    for t in time_range:
        local_R = 1
        if R > 1 and t < T:
            # Check ESS of previous iteration weights
            if isinstance(log_wtp1, (int, float)) and log_wtp1 == 0:
                prev_ess = P
            elif isinstance(log_wtp1, torch.Tensor) and log_wtp1.ndim == 0:
                prev_ess = P
            else:
                 # log_wtp1 should be a tensor of weights if not reset
                 if isinstance(log_wtp1, torch.Tensor):
                     prev_ess = compute_ess_from_log_w(log_wtp1).item()
                 else:
                     prev_ess = P

            if ess_threshold is not None and prev_ess < P * ess_threshold:
                 local_R = R

        xt_input = xtp1
        extra_vals_input = extra_vals
        log_wtp1_expanded = log_wtp1
        
        if local_R > 1:
            if isinstance(xtp1, torch.Tensor):
                xt_input = xtp1.repeat_interleave(local_R, dim=0)
            
            if isinstance(log_wtp1, torch.Tensor):
                log_wtp1_expanded = log_wtp1.repeat_interleave(local_R, dim=0)
            
            extra_vals_input = {}
            for k, v in extra_vals.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == P:
                     extra_vals_input[k] = v.repeat_interleave(local_R, dim=0)
                else:
                     extra_vals_input[k] = v

        P_eff = P * local_R

        # Transition kernel and reweighting
        xt, extra_vals_out = M(t=t, xtp1=xt_input, extra_vals=extra_vals_input, P=P_eff)
        log_w, x0_pred, sqrt_alphas_cumprod, alphas_cumprod = G(t=t, xtp1=xt_input, xt=xt, extra_vals=extra_vals_out)
        log_w += log_wtp1_expanded  # unnormalized 
        
        # if xt and log_w are tensors detach them.  Otherwise convert to tensors
        if isinstance(xt, torch.Tensor):
            xt = xt.detach()
        else:
            xt = torch.tensor(xt)

        if isinstance(log_w, torch.Tensor):
            log_w = log_w.detach()
        else:
            log_w = torch.tensor(log_w)

        if log_xt_trace:
            xt_trace.append(xt.cpu())
        ess_trace.append(compute_ess_from_log_w(log_w).cpu())
        log_w_trace.append(log_w.cpu())

        if verbose and t % 10 == 0:
            from tqdm.auto import tqdm
            tqdm.write(f"t={t}, ESS={ess_trace[-1].item():.2f}")

        # assert log_w.shape == (P,), log_w.shape
        
        should_contract = (local_R > 1) 

        if t > 0 or should_contract:
            target_P = P # Always contract/resample to original P
            
            if ess_threshold == 0 and not should_contract:
                resample_indices = torch.arange(P)
                log_wtp1 = log_w
            else:
                # resample 
                resample_indices, xt_new, is_resampled, new_log_weights = resample_fn(log_w, xt, x0_pred, target_P=target_P)
                xt = xt_new
                
                if is_resampled:
                    if new_log_weights is not None:
                        log_w = new_log_weights
                        log_wtp1 = new_log_weights
                    else:
                        log_w = torch.zeros(target_P, device=xt.device, dtype=log_w.dtype)
                        log_wtp1 = 0 # not accumulating (uniform weights)
                else: 
                    log_wtp1_normalized = normalize_log_weights(log_w, dim=0)
                    log_wtp1 = log_wtp1_normalized + np.log(P) 
                
                # Update extra_vals based on resample_indices
                current_extra_vals = {}
                for k, v in extra_vals_out.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == P_eff:
                        current_extra_vals[k] = v[resample_indices]
                    else:
                        current_extra_vals[k] = v
                extra_vals = current_extra_vals

            extra_vals[('resample_idcs', t)] = resample_indices
            
            resample_indices_trace.append(resample_indices)

        xtp1 = xt

    # Stack before returning
    if log_xt_trace:
        try:
            xt_trace = torch.stack(xt_trace, dim=0)
        except: pass
    
    try:
        ess_trace = torch.stack(ess_trace, dim=0)
    except: pass
        
    try:
        log_w_trace = torch.stack(log_w_trace, dim=0)
    except: pass
        
    if len(resample_indices_trace) > 0:
        resample_indices_trace = torch.stack(resample_indices_trace, dim=0)

    # Return normalized weights for final step
    normalized_w = normalize_weights(log_w, dim=0) 

    return xt, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace 
