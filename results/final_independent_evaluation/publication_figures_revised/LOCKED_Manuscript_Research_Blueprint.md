# Final Research Positioning and Scientific Contributions

Based on our frozen implementation, independent synthetic evaluation, verified results, and finalized package of 9 figures and 3 tables, I recommend the following scientific positioning for our IEEE Access manuscript.

Our research should be presented as an AI-enabled, motion-driven, causal closed-loop regulation framework evaluated under uncertainty and explicit control-resource constraints.

It is not primarily a motion-classification paper, a physiological-state prediction paper, or a demonstration of physically deployed VR technology.

## 1. Core research objective

To investigate whether incomplete, motion-derived information can support effective and resource-constrained closed-loop regulation of an unobservable simulated state, while explicitly quantifying the benefits, limitations, and adverse consequences of imperfect AI-driven feedback.

This objective captures the three fundamental dimensions of our research:

- State observability: The AI must infer a three-class state from motion rather than directly accessing the simulated latent state.
- Closed-loop effectiveness: The inferred state must produce meaningful improvements in recovery and stability.
- Constrained reliability: The controller must respect intervention limits, return to neutral, and expose failures caused by incorrect feedback.

Importantly, our objective is not to demonstrate that the AI reconstructs a physiological state with clinical accuracy.

## 2. Central research statement

> Motion-derived state estimates can support beneficial causal closed-loop adaptation in a simulated VR environment, even without direct access to the underlying latent state. However, these benefits are conditional on the quality of inferred feedback and the available control resources, creating measurable differences between AI-driven regulation, neutral control, and privileged-state regulation.

Our independent evaluation supports this statement through improved average recovery and stability, an oracle performance gap, heterogeneous individual outcomes, and documented incorrect interventions.

The claim is intentionally restricted to the investigated synthetic environment.

## 3. Core proposed work

Our framework integrates five essential methodological components.

1\. Synthetic state–motion environment

Parameterized synthetic episodes model the evolution of a hidden scalar state under Low and Elevated initial conditions. Three-dimensional motion observations are generated from the simulated process.

The latent state is available for simulation, matched-oracle control, and evaluation, but is excluded from the AI estimator's inputs.

2\. Causal motion-based state inference

Motion is observed at 90 Hz. Completed one-second windows provide 14 motion features, supplemented by elapsed time.

A frozen `StandardScaler + LogisticRegression` pipeline predicts Low, Regulated, or Elevated, without future observations or direct latent-state measurements.

3\. Neutral-return, resource-constrained feedback

A 10 Hz controller uses the inferred state to adjust a single normalized stimulus variable.

Control is restricted by a maximum per-update change of 0.05 and a total-variation budget of 0.6, including the reserved capacity needed to return the stimulus to neutral (0.5).

4\. Independent, resource-aware comparative evaluation

The frozen controller is evaluated against constant-neutral feedback and a privileged-state matched oracle.

Across 200 independently parameterized synthetic test profiles, 400 Low/Elevated episodes use matched random disturbances for paired comparisons.

5\. Control-performance and failure auditing

Evaluation jointly examines recovery, regulated-state stability, stimulus variation, profile-specific effects, opposite-direction interventions, adverse recovery outcomes, and reserve-boundary utilization.

This establishes both the achievable benefit and the limitations of feedback driven by imperfect observations.

## 4. Core research contributions

I recommend organizing the manuscript's contribution statement around four substantive contributions, rather than presenting each implementation component as a separate novelty.

### Contribution 1 — A causal motion-driven regulation framework

We develop an integrated synthetic closed-loop framework connecting observable motion, causal state inference, bounded stimulus adaptation, and continuously evolving hidden-state dynamics.

Scientific significance: It evaluates AI predictions through their downstream control consequences rather than treating prediction accuracy as the ultimate objective.

### Contribution 2 — Resource-constrained feedback with neutral-return reservation

We implement a state-dependent feedback policy that enforces limited intervention magnitude, a strict cumulative variation budget, and sufficient reserve for returning the applied stimulus to neutral.

Scientific significance: This makes control effectiveness assessable alongside intervention burden and terminal neutralization.

We should describe this as an implemented control mechanism, not claim mathematical optimality or a formal safety guarantee that has not been established.

### Contribution 3 — Independent and matched closed-loop benchmarking

We evaluate the frozen policy on independently parameterized synthetic profiles, comparing it with constant-neutral control and a privileged-state oracle subject to matched control constraints.

Mean recovery time

# 2.036 s

AI

Versus 2.262 s under neutral control

Regulated-state stability

# 83.03%

AI

Versus 81.15% under neutral control

The AI improves mean recovery by 0.2256 seconds (approximately 9.97%), while the oracle remains better at 1.8945 seconds.

The paired profile-cluster bootstrap confidence interval for the AI–neutral recovery difference is approximately −0.2608 to −0.1923 seconds.

Scientific significance: The comparative design distinguishes the value of estimated-state feedback from both the neutral baseline and the performance achievable with privileged state information.

### Contribution 4 — Explicit characterization of inference-driven feedback failures

Rather than reporting only population-average improvements, we characterize when and how erroneous control actions compromise individual outcomes.

# 65/400

Adverse recovery episodes

# 167/400

Episodes with opposite-direction actions

# 373/400

Episodes reaching the reserve boundary

All 65 adverse episodes contained opposite-direction interventions, but many episodes with such interventions still benefited overall.

Scientific significance: The work demonstrates that incorrect feedback is not equivalent to adverse recovery, and that favorable average performance can coexist with meaningful episode-level failures and resource saturation.

This is one of the strongest aspects of our research narrative.

## 5. What is genuinely novel?

We should be precise about the distinction between methodological novelty and experimental significance.

| Research element                                                                       | Defensible novelty/significance                                                 |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Motion-based state estimation                                                          | Established general technique; not independently novel                          |
| Logistic regression classification                                                     | Standard estimator; not an algorithmic novelty                                  |
| Causal windowing                                                                       | Established principle; essential to experimental validity                       |
| Neutral-return variation reservation                                                   | Specific implemented control design; potentially distinctive in its integration |
| Matched neutral/privileged-oracle comparison                                           | Strong comparative evaluation strategy                                          |
| Independent profile-disjoint evaluation                                                | Important methodological rigor                                                  |
| Joint analysis of benefit, incorrect action, adverse recovery, and reserve utilization | Distinctive contribution of the integrated evaluation                           |

Our strongest originality claim is therefore the integrated causal, resource-bounded closed-loop framework and its systematically audited evaluation under imperfect state information, rather than a new classifier or an entirely new control-theoretic algorithm.

We should not assert that this combination is the first of its kind without a dedicated comparison with relevant prior literature.

## 6. Principal scientific finding

The paper's central result can be stated as follows:

> Under the investigated synthetic state–motion dynamics, a frozen motion-derived three-class estimator enables resource-constrained feedback to improve average recovery time and regulated-state stability relative to neutral control. Nevertheless, the controller remains inferior to a privileged-state oracle and exhibits heterogeneous outcomes, including adverse recovery associated with opposite-direction interventions and frequent exhaustion of its available variation reserve.

This formulation captures the actual experimental findings without implying physiological validation, guaranteed safety, or control optimality.

## 7. Research questions guiding the manuscript

I recommend three tightly connected questions:

RQ1 — Closed-loop effectiveness: Can causal motion-derived state estimates improve recovery and stability over neutral feedback across independent synthetic profiles?

RQ2 — Information and resource constraints: How does the performance of resource-constrained AI feedback compare with a matched privileged-state oracle, and what do development information-source ablations reveal about the contribution of motion and elapsed time?

RQ3 — Reliability and failure: How do opposite-direction interventions, heterogeneous recovery outcomes, and reserve-boundary utilization characterize the limitations of the proposed feedback policy?

These align directly with our nine finalized figures and three proposed tables.

## 8. Boundaries of our conclusions

Four limitations must remain explicit throughout the manuscript:

- Synthetic validation: The study does not establish human physiological-state inference or effectiveness in a physical VR environment.
- Model specificity: The findings apply to the implemented motion-generation assumptions, frozen classifier, and investigated control policies.
- Control-resource interpretation: Stimulus variation is a normalized mathematical intervention-cost measure, not measured physical energy consumption.
- Reliability: Observed compliance with intervention limits does not establish safety under all possible environments, disturbances, or users.

## Final recommended positioning

Research objective: Establish whether motion-inferred, imperfect state information can support effective causal regulation under explicit intervention limits.

Proposed method: A frozen motion-based three-class estimator integrated with a neutral-return, variation-constrained closed-loop controller in a parameterized synthetic environment.

Main contribution: Independent, paired evidence of beneficial average control, quantified information limitations, and transparent analysis of adverse feedback behavior.

Scientific novelty: The integration and systematic evaluation of causal state inference, constrained intervention, privileged-oracle benchmarking, and feedback-failure auditing.

Significance: The study provides a reproducible framework for examining the practical relationship between imperfect AI state inference, closed-loop effectiveness, resource use, and reliability in simulated adaptive environments.

My final recommendation: This should be the locked scientific narrative for our manuscript. Every section—Introduction, Methods, Results, Discussion, Abstract, and Conclusions—should reinforce these claims and remain strictly consistent with the frozen implementation and verified results.

The research is best presented as a rigorous synthetic closed-loop systems study, not as an experimentally validated physiological VR system.