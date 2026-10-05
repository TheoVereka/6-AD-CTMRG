# Selected NN-correlation data audit

- Discovered candidates: 454
- Selected grid points: 100 / 108
- Global selection objective: 711.80249
- points with a 0713 reference: 100
- unreferenced points retained explicitly: 0
- |dE| <= 0.0002: 100/100 compared points
- relative Delta difference <= 25%: 98/100 compared points
- relative Delta difference > 35% exceptions: 2
- fixed-a diagnostics: 54 rank fits

The hard energy cutoff is 0.0003. Delta <=25% is preferred and <=35% is the normal wide window. A >35% candidate can enter only in the low-J2 restoration regime, only when its absolute splitting is small enough to support D->infinity equality, and only when the grid point has no energy-qualified <=35% option; every such exception is exposed explicitly.

D=5 plaquette data are excluded unconditionally. The old D=7, J2=.26 dimer data are also excluded; that point remains missing until a complete candidate appears under the dedicated D7 repair snapshot root.
D=11 plaquette points at J2=.26, .265, and .27 are excluded because no corresponding 0713 energy/Delta reference exists.

## Missing grid points

- plaquette, D=6, J2=0.29
- plaquette, D=6, J2=0.3
- plaquette, D=6, J2=0.31
- plaquette, D=6, J2=0.32
- plaquette, D=11, J2=0.26
- plaquette, D=11, J2=0.265
- plaquette, D=11, J2=0.27
- plaquette, D=11, J2=0.32

## Delta-window exceptions

- dimer-plaquette, D=7, J2=0.26: 349.8%, d7_repair:pin_h0p02/h0
- dimer-plaquette, D=10, J2=0.28: 64.5%, kuma_targeted_repair:r01:h0p02_to_h0

## Fixed energy-fit length

- J2=0.26: a_g=0.501871646
- J2=0.265: a_g=0.408518444
- J2=0.27: a_g=0.26539243
- J2=0.275: a_g=0.34337477
- J2=0.28: a_g=0.381937591
- J2=0.29: a_g=0.376982062
- J2=0.3: a_g=0.357210863
- J2=0.31: a_g=0.379232092
- J2=0.32: a_g=0.466653397
