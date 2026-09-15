# Izar 31-GiB memory audit

The audit uses only this project's historical run logs. It is deliberately
conservative because the closure peak, not the steady CTMRG allocation,
determines whether optimization survives.

## Direct evidence

- On a GPU reporting 31.74 GiB total memory, a twoC3 `D=9, chi=126` closure
  already had about 28.7 GiB in use and then failed while requesting another
  3.56 GiB. The same failure repeated during line search.
- Project runs at `D=9, chi=108` with the automatic `chi=117` lookahead have no
  recorded OOM on Izar.
- Project runs at `D=8, chi=104` with `chi=112` lookahead also have no recorded
  OOM on Izar.
- The present high-D schedules (`D10 chi160/180`, `D11 chi165`, and nearby)
  were exercised in the August run family on GPUs whose OOM logs report
  93.12 GiB total memory. They are not evidence that those pairs fit Izar.
  Indeed a `D=11, chi=165` closure in that family reached about 81 GiB before
  an additional 11.88-GiB request failed.

## Launcher policy

| D | Izar chi | lookahead | status |
|---:|---:|---:|---|
| 8 | 104 | 112 | allowed; historically safe |
| 9 | 108 | 117 | allowed; historically safe |
| 9 | 126 or higher | 135+ | blocked; documented 31.74-GiB OOM |
| 10 | publication chi 140--180 | 150+ | blocked; not validated on 31 GiB |
| 11 | publication chi 132--165 | 143+ | blocked; expected far above 31 GiB |

Thus Izar can provide a controlled D8/D9 branch test. D10/D11 at the chi used
for the phase-diagram-quality data should be sent to Kuma. A reduced-chi
D10/D11 Izar experiment is possible only as an exploratory calculation; it
would not have the same environmental resolution and should not be mixed into
the final energy-gap scaling.

To force such an experiment, both variables must be explicit:

```bash
ALLOW_UNSAFE_IZAR=1 CHI_OVERRIDE=80 D_VALUES_TEXT="10" \
bash submit_izar_vbc.sh
```

This override means "submit despite lack of validation"; it is not a claim
that the chosen pair fits.

