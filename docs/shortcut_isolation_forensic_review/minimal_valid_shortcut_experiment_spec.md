# Minimal Valid Shortcut-Isolation Experiment Spec

This is a design only. It must not be implemented as part of this forensic review.

## Candidate shortcut family S

Use one finite, predeclared low-cost family, for example an output-bag/color-frequency shortcut: predict the query output by copying a support-output color histogram or selecting the nearest support output by a nuisance tag. Do not define S as all possible shortcuts.

## Legitimate relation R

R is the true support input-to-output transformation that remains available in every condition. For a tiny model, use synthetic ARC-like tasks where R is exactly known, such as recolor object A by relation f(input shape/color) while preserving coordinates.

## Nuisance variable N

N is a semantically irrelevant cue made correlated with the answer in C_preserve and counterbalanced in C_break, such as support-output histogram identity, support order, or a dummy tag that is explicitly independent of the true rule.

## Matched conditions

C_preserve: R and N are aligned, so both the legitimate rule and the shortcut family predict the same query answer.

C_break: R is unchanged and still visible, but N is broken or counterbalanced so the shortcut family predicts the wrong or uninformative answer.

## Behavioral outcome

Predeclare held-out query exact grid accuracy and query NLL as primary outcomes. Record support reconstruction and optimization trajectory as secondary outcomes if the model is trained/adapted.

## Criteria

Shortcut dependence is supported only if C_break harms query behavior while preservation tests confirm R is intact and a rule-based checker still solves both conditions. Legitimate rule dependence is supported if behavior remains stable when N is broken. Numerical/trace artifact is assigned when only internal hashes or thresholds move. No measurable effect is assigned if behavior stays within a predeclared equivalence margin.

## Necessity test

Remove or counterbalance N while preserving R. The candidate shortcut family is necessary only if the behavioral effect disappears when N is unavailable or inverted.

## Preservation test

An independent rule checker must solve C_preserve and C_break identically using R, and task manifests must prove support inputs, outputs, target rule, and query target semantics remain valid.

## Substitution test

After blocking S, evaluate whether the model migrates to an alternate declared nuisance family S2. If it does, the result is shortcut-family substitution, not solved reasoning.

## Why this is more informative

Support-order equivariance asks whether a representation respects pair ordering. Byte equality and Cummax tracing locate internal mismatches. Layer locking changes architecture without proving a nuisance route. The minimal experiment instead tests the causal triad: preserved R, selectively destroyed N, and changed end-to-end query behavior.
