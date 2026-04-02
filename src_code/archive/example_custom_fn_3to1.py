"""
Example: wrapping update_environmentCTs_3to1 as a torch.autograd.Function.

NOT used anywhere — pure illustration of the pattern.

Memory trade-off
────────────────
Without wrapping:
    Every intermediate tensor produced inside  update_environmentCTs_3to1
    (einsum results, reshape views, SVD matrices …) is stored on the autograd
    graph for *every* CTMRG iteration.  For D=4 chi=17 that is ~151 MB of
    (M×M)-class intermediates and ~85 MB from SVD alone, per-update-call,
    accumulated across all iterations.

With this wrapper:
    forward()  runs in no-grad mode (PyTorch disables autograd inside
    Function.forward by default).  Only the 15 original input tensors are
    registered with save_for_backward — zero intermediates touch the graph.

    backward() re-executes the original function on detached-and-re-attached
    copies of the inputs WITH autograd enabled, builds the local computation
    graph for that single update step, calls torch.autograd.grad, then frees
    the local graph immediately.

Cost: one extra forward pass per update step during the backward phase.

When is this large vs. small?
    Large benefit  → many large intermediates (current code, un-fused einsums).
    Shrinking benefit → as you fuse contractions, there are fewer intermediates
                        to store, so the baseline cost drops and the relative
                        gain from this wrapper shrinks proportionally.
    Guideline: do the succinct-contraction rewrite *first*; only add this
    wrapper afterwards if the residual intermediate storage is still significant.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from core_unrestricted import update_environmentCTs_3to1, set_dtype


# ─────────────────────────────────────────────────────────────────────────────
class UpdateEnv3to1Fn(torch.autograd.Function):
    """
    Custom autograd.Function wrapper for update_environmentCTs_3to1.

    Call:
        outputs = UpdateEnv3to1Fn.apply(chi, D_squared,
                                        C21AF, C32CB, C13ED,
                                        T1B, T2E, T2D, T3A, T3F, T1C,
                                        A, B, C, D, E, F)

    Returns the same 9-tuple as the original function:
        (C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E)
    """

    @staticmethod
    def forward(ctx, chi, D_squared,
                C21AF, C32CB, C13ED,
                T1B, T2E, T2D, T3A, T3F, T1C,
                A, B, C, D, E, F):

        # chi and D_squared are plain Python ints — store on ctx directly.
        ctx.chi       = chi
        ctx.D_squared = D_squared

        # Save ONLY the 15 original input tensors.
        # PyTorch disables autograd inside Function.forward, so none of the
        # intermediate computations below create gradient nodes.
        ctx.save_for_backward(
            C21AF, C32CB, C13ED,
            T1B, T2E, T2D, T3A, T3F, T1C,
            A, B, C, D, E, F,
        )

        # Run the full update — no gradient graph is built here.
        return update_environmentCTs_3to1(
            C21AF, C32CB, C13ED,
            T1B, T2E, T2D, T3A, T3F, T1C,
            A, B, C, D, E, F,
            chi, D_squared,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        grad_outputs: 9 gradient tensors (one per output of forward),
                      or None for outputs that did not need grads.
        """
        chi       = ctx.chi
        D_squared = ctx.D_squared

        # ── Unpack saved inputs ────────────────────────────────────────────
        (C21AF, C32CB, C13ED,
         T1B, T2E, T2D, T3A, T3F, T1C,
         A, B, C, D, E, F) = ctx.saved_tensors

        # Detach each saved tensor and re-enable grad only where the original
        # input required it.  This is the standard re-materialisation idiom.
        inputs_re = [
            t.detach().requires_grad_(t.requires_grad)
            for t in ctx.saved_tensors
        ]
        (C21AF_r, C32CB_r, C13ED_r,
         T1B_r, T2E_r, T2D_r, T3A_r, T3F_r, T1C_r,
         A_r, B_r, C_r, D_r, E_r, F_r) = inputs_re

        # ── Re-run the forward WITH autograd to build the local graph ──────
        with torch.enable_grad():
            outputs = update_environmentCTs_3to1(
                C21AF_r, C32CB_r, C13ED_r,
                T1B_r, T2E_r, T2D_r, T3A_r, T3F_r, T1C_r,
                A_r, B_r, C_r, D_r, E_r, F_r,
                chi, D_squared,
            )

            # Only differentiate outputs that actually participated in the graph.
            diff_outputs = []
            diff_grads   = []
            for out, g in zip(outputs, grad_outputs):
                if g is not None and out.requires_grad:
                    diff_outputs.append(out)
                    diff_grads.append(g)

            # torch.autograd.grad requires every tensor in `inputs` to have
            # requires_grad=True.  Filter to only grad-enabled inputs, then
            # map the results back into the full 15-slot tuple.
            grad_inputs_idx  = [i for i, t in enumerate(inputs_re) if t.requires_grad]
            grad_inputs_list = [inputs_re[i] for i in grad_inputs_idx]

            if diff_outputs and grad_inputs_list:
                computed = torch.autograd.grad(
                    outputs      = diff_outputs,
                    inputs       = grad_inputs_list,
                    grad_outputs = diff_grads,
                    allow_unused = True,
                )
            else:
                computed = (None,) * len(grad_inputs_list)

            # Reconstruct the full 15-element gradient tuple (None for
            # non-differentiable inputs such as C corners and T transfers).
            input_grads_list = [None] * len(inputs_re)
            for slot, grad in zip(grad_inputs_idx, computed):
                input_grads_list[slot] = grad

        # Local graph is freed here (goes out of scope).

        # Return one gradient per forward() argument.
        # The first two are chi and D_squared (plain ints) → always None.
        return (None, None) + tuple(input_grads_list)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper matching the original function signature.
# Swap this in place of update_environmentCTs_3to1 where desired.
# ─────────────────────────────────────────────────────────────────────────────
def update_environmentCTs_3to1_custom(
        C21AF, C32CB, C13ED,
        T1B, T2E, T2D, T3A, T3F, T1C,
        A, B, C, D, E, F,
        chi, D_squared):
    return UpdateEnv3to1Fn.apply(
        chi, D_squared,
        C21AF, C32CB, C13ED,
        T1B, T2E, T2D, T3A, T3F, T1C,
        A, B, C, D, E, F,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test: outputs must be numerically identical and gradients must
# flow back to the site tensors (A–F).
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_dtype(use_double=True)   # match production precision
    torch.manual_seed(0)
    chi, D_bond = 10, 3
    D_sq = D_bond ** 2

    def random_C():  return torch.randn(chi, chi, dtype=torch.complex128) * 0.1
    def random_T():  return torch.randn(chi, chi, D_sq, dtype=torch.complex128) * 0.1
    def random_AB(): return torch.randn(D_sq, D_sq, D_sq, dtype=torch.complex128,
                                        requires_grad=True)

    C21AF, C32CB, C13ED   = random_C(), random_C(), random_C()
    T1B, T2E, T2D         = random_T(), random_T(), random_T()
    T3A, T3F, T1C         = random_T(), random_T(), random_T()
    A, B, C_t, D_t, E, F  = (random_AB() for _ in range(6))

    # ── Reference: standard function ─────────────────────────────────────────
    out_ref = update_environmentCTs_3to1(
        C21AF, C32CB, C13ED,
        T1B, T2E, T2D, T3A, T3F, T1C,
        A, B, C_t, D_t, E, F,
        chi, D_sq,
    )
    loss_ref = sum(o.abs().sum() for o in out_ref)
    loss_ref.backward()
    grad_A_ref = A.grad.clone()

    # Reset grads
    for t in (A, B, C_t, D_t, E, F):
        t.grad = None

    # ── Custom Function ───────────────────────────────────────────────────────
    out_custom = update_environmentCTs_3to1_custom(
        C21AF, C32CB, C13ED,
        T1B, T2E, T2D, T3A, T3F, T1C,
        A, B, C_t, D_t, E, F,
        chi, D_sq,
    )
    loss_custom = sum(o.abs().sum() for o in out_custom)
    loss_custom.backward()
    grad_A_custom = A.grad.clone()

    # ── Checks ────────────────────────────────────────────────────────────────
    for o_ref, o_cst in zip(out_ref, out_custom):
        max_diff = (o_ref - o_cst).abs().max().item()
        assert max_diff < 1e-12, f"output mismatch: {max_diff}"

    grad_diff = (grad_A_ref - grad_A_custom).abs().max().item()
    assert grad_diff < 1e-10, f"grad mismatch: {grad_diff}"

    print("✓  outputs numerically identical")
    print(f"✓  grad(A) max diff = {grad_diff:.2e}")
    print("Wrapper is correct.")
