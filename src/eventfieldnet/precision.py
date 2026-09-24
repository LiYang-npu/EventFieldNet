"""Explicit R66 arithmetic policy shared by training and evaluation builders.

Call configure_precision(config.precision) in every model factory, including
the official evaluator subprocess. The caller still owns its autocast region:
BF16 uses torch.autocast; FP32 must use no autocast. This module does not assert
that FP32 is deterministic or force a different attention implementation.
"""

import os
import torch


POLICY_VERSION = "r66_no_tf32_explicit_precision_v1"


def snapshot_precision():
    cuda = torch.backends.cuda
    return dict(
        policy_version=POLICY_VERSION,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        matmul_allow_tf32=bool(cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        deterministic_warn_only=bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        flash_sdp_enabled=bool(cuda.flash_sdp_enabled()),
        mem_efficient_sdp_enabled=bool(cuda.mem_efficient_sdp_enabled()),
        math_sdp_enabled=bool(cuda.math_sdp_enabled()),
        bf16_reduced_precision_reduction=bool(
            cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def configure_precision(precision):
    if precision not in ("bf16", "fp32"):
        raise ValueError("R66 precision must be bf16 or fp32")
    # Identical TF32 policy for both arms, preventing the FP32-labelled arm
    # from silently using lower-mantissa TensorFloat32 GEMMs/convolutions.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = snapshot_precision()
    result.update(
        requested_precision=precision,
        autocast_required=precision == "bf16",
        attention_dispatch="existing auto dispatch; actual kernels can depend on dtype",
        deterministic_guarantee=False,
    )
    assert not result["matmul_allow_tf32"] and not result["cudnn_allow_tf32"]
    assert result["float32_matmul_precision"] == "highest"
    return result
