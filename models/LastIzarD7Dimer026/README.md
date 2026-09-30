# D7 dimer J2=.26 repair bundle

This is a self-contained static Izar bundle.  It uses D=7, chi=91, CTM max
steps 50, double precision, noise 1e-3, QOS `long`, partition `gpu`, Slurm
wall time `167:59:50`, and an internal optimization limit of 70 hours.

It submits six jobs: four independent pure-LBFGS continuations from the
optimized s002 D=7/J2=.265 dimer-plaquette tensor (left branch, insurance 2),
plus a mean-field `main_C3.py` dimer pin at h=.02 whose pure-LBFGS h=0
continuation is submitted with `afterok`.  Exact seed provenance and SHA-256
are recorded under `seeds/provenance.txt`.

From the repository root:

```powershell
scp -r .\models\LastIzarD7Dimer026 chye@izar.hpc.epfl.ch:~/
```

On Izar:

```bash
cd ~/LastIzarD7Dimer026
bash submit_all.sh --dry-run
bash submit_all.sh
```

Download safely while jobs are still running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\models\LastIzarD7Dimer026\download_snapshot.ps1
```

The local destination is
`data/distinVBCsJ2Continuation/D7DimerJ2_0p26`.  Each stage atomically creates
`COMPLETED.stage` only after its best tensor and observation both exist.
Downstream selection requires that marker, so a snapshot taken while jobs are
running cannot ingest a partially written stage.
