from typing import Dict, Callable

import fast_bss_eval
import torch
import torchmetrics

METRIC_FUNCTIONS = []

def metric_function(function: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]) -> None:
    METRIC_FUNCTIONS.append(function)

@metric_function
def calculate_torchmetrics_metrics(
        reference: torch.Tensor,
        estimate: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate metrics using torchmetrics
    Args:
        reference: true value
        estimate: model estimate

    Returns: Dict containing SNR, SDR, SI-SNR, SI-SDR metrics

    """
    try:
        snr: torch.Tensor = torchmetrics.functional.signal_noise_ratio(estimate, reference)
        sdr: torch.Tensor = torchmetrics.functional.signal_distortion_ratio(estimate, reference)
        si_sdr: torch.Tensor = torchmetrics.functional.scale_invariant_signal_distortion_ratio(estimate, reference)
    except:
        snr = sdr = si_sdr = torch.tensor(0, dtype=torch.float32)
    return {
        "SNR": torch.mean(snr),
        "SDR": torch.mean(sdr),
        "SI-SDR": torch.mean(si_sdr)
    }

@metric_function
def calculate_fast_bsseval_metrics(
        reference: torch.Tensor,
        estimate: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate metrics using museval library
    Args:
        reference: true value
        estimate: model estimate

    Returns: Dict containing SDR, ISR, SIR and SAR metrics
    """
    try:
        _, _, sar = fast_bss_eval.bss_eval_sources(reference, estimate, compute_permutation=False)
        _, _, si_sar = fast_bss_eval.si_bss_eval_sources(reference, estimate, compute_permutation=False)
    except:
        sar = si_sar = torch.tensor(0, dtype=torch.float32)
    return {
        "SAR": torch.mean(sar),
        "SI-SAR": torch.mean(si_sar)
    }

def calculate_metrics(
        reference: torch.Tensor,
        estimate: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Calculate all implemented metrics
    Args:
        reference: true value
        estimate: model estimate

    Returns: Dictionary with all calculated metrics
    """
    return {
        metric_name: metric_value
        for function in METRIC_FUNCTIONS
        for metric_name, metric_value in function(reference, estimate).items()
    }
