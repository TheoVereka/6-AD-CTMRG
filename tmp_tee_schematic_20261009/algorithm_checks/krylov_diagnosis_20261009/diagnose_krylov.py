"""Read-only diagnosis; instrumented copy, never mutate production code."""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPTS = ROOT / "src_code" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import renyi2_twoc3 as core

TRACE = []
DENSE = None
PROJECTOR = None


def norm(value):
    return float(torch.linalg.vector_norm(value).item())


def orthogonality(q):
    gram = q.T @ q
    gram.diagonal().sub_(1)
    return norm(gram)


def record_basis(event, q, aq, h, m, calls, restarts):
    q = q[:, :m]
    aq = aq[:, :m]
    exact_aq = DENSE @ q
    actual_h = q.T @ aq
    TRACE.append(dict(event=event, m=m, calls=calls, restarts=restarts,
                      orthogonality=orthogonality(q),
                      aq_consistency=norm(exact_aq-aq)/max(norm(exact_aq), 1e-300),
                      h_consistency=norm(h[:m,:m]-actual_h)/max(norm(actual_h),1e-300),
                      max_column_norm=float(torch.linalg.vector_norm(q,dim=0).max()),
                      projected_radius=float(torch.linalg.eigvals(actual_h).abs().max())))


def record_orth(candidate, selected, r, keep, before, q, m, calls, restarts):
    original_q = q[:, :m]
    TRACE.append(dict(event="orthogonalize", m=m, calls=calls, restarts=restarts,
                      before_norm=before, after_norm=norm(candidate),
                      singular_values=torch.linalg.svdvals(candidate).tolist(),
                      r_diagonal=torch.diagonal(r).tolist(),
                      keep=keep.tolist(),
                      old_orthogonality=orthogonality(original_q) if m else 0.,
                      before_qr_cross=norm(original_q.T@candidate) if m else 0.,
                      after_qr_cross=norm(original_q.T@selected) if m else 0.,
                      selected_orthogonality=orthogonality(selected),
                      projected_image_leak=norm(torch.stack([PROJECTOR(selected[:,j]) for j in range(selected.shape[1])],dim=1)-selected) if selected.shape[1] else 0.))


def instrument():
    source=(SCRIPTS/"renyi2_spectral.py").read_text(encoding="utf-8")
    source=source.replace("import torch\n", "import torch\nfrom __main__ import record_basis, record_orth\n")
    source=source.replace("        return candidate_q[:, keep]\n", "        selected = candidate_q[:, keep]\n        record_orth(candidate, selected, candidate_r, keep, before, q_basis, m, calls, restarts)\n        return selected\n")
    source=source.replace("        return width\n", "        record_basis('append', q_basis, aq_basis, projected, m, calls, restarts)\n        return width\n")
    source=source.replace("                transform_basis(u, retained)\n", "                record_basis('before_restart', q_basis, aq_basis, projected, m, calls, restarts)\n                transform_basis(u, retained)\n")
    source=source.replace("                restarts += 1\n", "                restarts += 1\n                record_basis('after_restart', q_basis, aq_basis, projected, m, calls, restarts)\n")
    copy=HERE/"renyi2_spectral_instrumented.py"
    copy.write_text(source,encoding="utf-8")
    spec=importlib.util.spec_from_file_location("instrumented",copy)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    global DENSE, PROJECTOR
    torch.set_num_threads(2)
    module=instrument()
    archive=np.load(HERE.parent/"tiny_checkpoint_pair1_edges.npz")
    edges=core.Edges(*[torch.tensor(archive[name],dtype=torch.float64) for name in "ABCD"]).normalized()
    a,b,c,d=[getattr(edges,name) for name in "ABCD"]
    chi=edges.chi;n=chi**4
    upper=torch.einsum("abip,bcjq,cdkr,dals->pqrsijkl",a,b,a,b).reshape(n,n)
    lower=torch.einsum("abpi,bcqj,cdrk,dasl->ijklpqrs",d,c,d,c).reshape(n,n)
    dense=lower@upper
    report=[]
    for parity in (1,-1):
        TRACE.clear()
        op=core.ReplicaTransfer(edges,batch=8,parity=parity)
        PROJECTOR=op.project
        DENSE=torch.stack([PROJECTOR(dense[:,j]) for j in range(n)],dim=1)
        result=module.solve_block(op,n,8,block_size=4,subspace=80,tol=1e-10,
                                  max_matvec=800,seed=20261009,projector=PROJECTOR)
        (HERE/f"trace_parity_{parity:+d}.json").write_text(json.dumps(TRACE,indent=2),encoding="utf-8")
        first_bad=next((i for i,e in enumerate(TRACE) if e.get("orthogonality",e.get("after_qr_cross",0))>1e-6),None)
        print("parity",parity,"result",result.reason,"ortho",result.orthogonality_error,"radius",abs(result.eigenvalues[0]),"firstbad",first_bad,flush=True)
        if first_bad is not None:
            print(json.dumps(TRACE[max(0,first_bad-2):first_bad+3],indent=2),flush=True)
        report.append(dict(parity=parity,result_reason=result.reason,
                            final_orthogonality=result.orthogonality_error,
                            final_radius=float(abs(result.eigenvalues[0])),first_bad_event=first_bad,
                            exact_radius=float(torch.linalg.eigvals(DENSE).abs().max())))
    (HERE/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":main()
