# VRM Twin Epoch Baseline v1 comparison

This is a fair-training-protocol baseline for the unchanged VRM V1.1-shaped architecture. It is not ARC training and not an ARC-solving claim.

|split|condition|direct exact|foreground color acc|candidate score|tie rate|derived relevant-pair output rate|any-support-output match rate|twin same-grid rate|candidate change rate|correct mapping switch rate|pair joint normal acc|mean Hamming|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|train|NORMAL|0.305125|0.337637|0.499844|0.290687|0.305125|0.904687|0.762750|NA|NA|0.029250|3.311812|
|train|TWIN_SWAP_SUPPORTS|0.310000|0.340250|0.500156|0.290687|0.310000|0.904687|0.762750|0.059875|0.029250|0.029250|3.298750|
|train|PAIR_ORDER_SHUFFLED|0.307125|0.336875|0.498344|0.295187|0.307125|0.909375|0.762750|0.198250|NA|0.029250|3.315625|
|dev|NORMAL|0.302000|0.334000|0.505750|0.306500|0.302000|0.906000|0.770000|NA|NA|0.036000|3.330000|
|dev|TWIN_SWAP_SUPPORTS|0.298000|0.326800|0.494250|0.306500|0.298000|0.906000|0.770000|0.064000|0.036000|0.036000|3.366000|
|dev|PAIR_ORDER_SHUFFLED|0.292500|0.323400|0.493000|0.306000|0.292500|0.904500|0.770000|0.198500|NA|0.036000|3.383000|
|sealed|NORMAL|0.304500|0.332600|0.498750|0.307500|0.304500|0.908500|0.756000|NA|NA|0.026000|3.337000|
|sealed|TWIN_SWAP_SUPPORTS|0.298000|0.331200|0.501250|0.307500|0.298000|0.908500|0.756000|0.051000|0.026000|0.026000|3.344000|
|sealed|PAIR_ORDER_SHUFFLED|0.304000|0.335900|0.505000|0.306000|0.304000|0.903500|0.756000|0.187000|NA|0.026000|3.320500|

For `TWIN_SWAP_SUPPORTS`, candidate change and correct mapping switch measure movement toward the opposite twin answer.
For `PAIR_ORDER_SHUFFLED`, candidate change is relative to `NORMAL` and should remain low if predictions are order-stable.

Conclusion: `INCONCLUSIVE`

Runtime seconds: 1906.948
