# Transport-Recolor R5A-0

Verdict: `R5A0_FUNCTION_COLLISION_DETECTED`

The requested primitive split contains exact functional collisions over the required `(x,y,c)->(x_final,y_final,c_final)` domain.
- `F->U` equals `U->F`
- `L->C` equals `C->U`

Per the R5A-0 protocol, dataset generation, model construction, training, checkpointing, and sealed evaluation were not started.
