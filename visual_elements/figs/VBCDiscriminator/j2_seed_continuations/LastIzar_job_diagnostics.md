# LastIzar job diagnostics

Result states: `{'complete': 45, 'no_output': 13, 'partial': 13}`  
Slurm states: `{'COMPLETED': 45, 'PENDING': 13, 'FAILED': 13}`

| job | task/stage | D | J2 | h | result | Slurm | cause |
|---|---|---:|---:|---:|---|---|---|
| L2a6270z | task2_D6_adam_pin/pin_h0 | 6 | 0.27 | 0 | no_output | PENDING | afterok blocked by L2a6270p: FAILED; IndexError: deque index out of range |
| L2a6270p | task2_D6_adam_pin/pin_h005 | 6 | 0.27 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L2a6275z | task2_D6_adam_pin/pin_h0 | 6 | 0.275 | 0 | no_output | PENDING | afterok blocked by L2a6275p: FAILED; IndexError: deque index out of range |
| L2a6275p | task2_D6_adam_pin/pin_h005 | 6 | 0.275 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L2a6280z | task2_D6_adam_pin/pin_h0 | 6 | 0.28 | 0 | no_output | PENDING | afterok blocked by L2a6280p: FAILED; IndexError: deque index out of range |
| L2a6280p | task2_D6_adam_pin/pin_h005 | 6 | 0.28 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L2a6300z | task2_D6_adam_pin/pin_h0 | 6 | 0.30 | 0 | no_output | PENDING | afterok blocked by L2a6300p: FAILED; IndexError: deque index out of range |
| L2a6300p | task2_D6_adam_pin/pin_h005 | 6 | 0.30 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L2a6320z | task2_D6_adam_pin/pin_h0 | 6 | 0.32 | 0 | no_output | PENDING | afterok blocked by L2a6320p: FAILED; IndexError: deque index out of range |
| L2a6320p | task2_D6_adam_pin/pin_h005 | 6 | 0.32 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5260z | task4_D5_adam_pin/pin_h0 | 5 | 0.26 | 0 | no_output | PENDING | afterok blocked by L4a5260p: FAILED; IndexError: deque index out of range |
| L4a5260p | task4_D5_adam_pin/pin_h005 | 5 | 0.26 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5270z | task4_D5_adam_pin/pin_h0 | 5 | 0.27 | 0 | no_output | PENDING | afterok blocked by L4a5270p: FAILED; IndexError: deque index out of range |
| L4a5270p | task4_D5_adam_pin/pin_h005 | 5 | 0.27 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5275z | task4_D5_adam_pin/pin_h0 | 5 | 0.275 | 0 | no_output | PENDING | afterok blocked by L4a5275p: FAILED; IndexError: deque index out of range |
| L4a5275p | task4_D5_adam_pin/pin_h005 | 5 | 0.275 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5280z | task4_D5_adam_pin/pin_h0 | 5 | 0.28 | 0 | no_output | PENDING | afterok blocked by L4a5280p: FAILED; IndexError: deque index out of range |
| L4a5280p | task4_D5_adam_pin/pin_h005 | 5 | 0.28 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5290z | task4_D5_adam_pin/pin_h0 | 5 | 0.29 | 0 | no_output | PENDING | afterok blocked by L4a5290p: FAILED; IndexError: deque index out of range |
| L4a5290p | task4_D5_adam_pin/pin_h005 | 5 | 0.29 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5300z | task4_D5_adam_pin/pin_h0 | 5 | 0.30 | 0 | no_output | PENDING | afterok blocked by L4a5300p: FAILED; IndexError: deque index out of range |
| L4a5300p | task4_D5_adam_pin/pin_h005 | 5 | 0.30 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5310z | task4_D5_adam_pin/pin_h0 | 5 | 0.31 | 0 | no_output | PENDING | afterok blocked by L4a5310p: FAILED; IndexError: deque index out of range |
| L4a5310p | task4_D5_adam_pin/pin_h005 | 5 | 0.31 | 0.005 | partial | FAILED | IndexError: deque index out of range |
| L4a5320z | task4_D5_adam_pin/pin_h0 | 5 | 0.32 | 0 | no_output | PENDING | afterok blocked by L4a5320p: FAILED; IndexError: deque index out of range |
| L4a5320p | task4_D5_adam_pin/pin_h005 | 5 | 0.32 | 0.005 | partial | FAILED | IndexError: deque index out of range |
