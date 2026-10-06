# Phase DAGs and input deltas

`Phase(..., delta_scale=...)` optionally sets its input delta's HalfNormal prior
scale. `Order(before, after, anchors=(p, q))` specifies relationships. Branching
and merging are supported; duplicate edges, cycles and inconsistent receiving
quantiles are rejected.

For each non-root phase B, the model derives its location so that

$$
Q_B(q) = \max_{A \to B} Q_A(p_A) + \delta_B, \qquad \delta_B > 0.
$$

The maximum is exact and evaluated on each draw. No smoothing or inequality
potential is introduced. In native negative-BP coordinates the maximum is the
youngest predecessor anchor. Root locations retain their independent priors;
root phases have no input delta. Delta describes anchor separation, not
necessarily a gap or hiatus. Distribution and measurement machinery is unchanged.

`delta` now has a labelled `input_phase` dimension, replacing the old per-edge
`order` dimension. A chain retains the same mathematical prior when settings
are unchanged. `Order.delta_scale` remains a legacy fallback when the receiving
Phase has no explicit setting. Incoming legacy scales must agree at a merge;
otherwise the user must specify the receiving Phase's scale. Auto uses the
largest resolved reference time scale among receiver and predecessors.

DAG resolution uses phase specifications rather than observed label groups.
The current fitter still requires labelled measurements for every phase.
Unobserved phases are a future extension requiring explicit prior resolution;
there is no dummy-phase inference in this MVP.

Checks cover chain compatibility, diamond algebra at fixed parameter points,
cycle/duplicate/anchor validation, and one 12-tune/8-draw single-chain NUTS smoke
run. That short run establishes execution only, not convergence or general
sampling reliability around switching maximum anchors.
