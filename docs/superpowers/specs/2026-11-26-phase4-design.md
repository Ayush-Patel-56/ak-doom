# Phase 4: learned hierarchy, helper rewards, curiosity, and RGB

## Requested outcome

Implement the four requested Phase 4 features on top of Phase 3: a learned five-goal manager operating every eight decisions, a goal-conditioned worker, goal helper rewards, optional RND curiosity, and a learned RGB encoder with exit-visibility supervision. Deliver a tested feature branch and a described PR targeting Ayush-Patel-56/ak-doom:main. The fork hosts the source branch only; neither main branch is pushed directly.

## Approach and alternatives

Extend the existing custom recurrent PPO rather than introducing a second RL framework. Train the worker at the decision rate and the manager at goal boundaries, with stored sampled goals and boundary masks so PPO replays exactly the behavior distribution that generated each rollout. This preserves the project's recurrent and spatial memory architecture.

A scripted manager would violate the requirement that goal selection is learned. A flat policy with a five-way auxiliary classifier would not implement temporal manager/worker control. A separate external hierarchical-RL framework would duplicate the current checkpoint, rollout, and evaluation paths. These alternatives are outside this implementation.

## Learned hierarchy

- Manager goals follow System Design §3.4: Explore, Fight, Collect, Go to landmark, Go to exit. Goal count and an eight-decision cadence are architecture requirements, not hardcoded action choices.
- Sample a goal from neural manager logits at episode start and each subsequent eight-decision boundary. Hold that sampled goal between boundaries. Reset cadence and recurrent/goal state per episode.
- Condition the worker's six action-group distributions on an embedding of the sampled goal. No hand-written goal-selection conditions or movement rules.
- Preserve active goal and cadence across rollout boundaries and replay the stored goal rather than resampling during PPO optimization. Manager policy/entropy losses apply only at decision boundaries.
- Use discounted rewards over the actual manager interval, including shortened intervals at episode end, for its return/advantage calculation. Keep manager and worker objectives and metrics explicit.
- Keep training and inference state transitions consistent, including partial intervals, asynchronous episode resets, and checkpoint reload.

## Goal helper rewards

- Training-only, configurable small bonuses attach observed progress to the selected goal: new scorer areas for Explore, kills for Fight, actual pickups for Collect, door-opening events for Go to landmark, and successful exits for Go to exit.
- Separate extrinsic scorer-aligned reward, helper reward, and curiosity in logs. Manager learning uses extrinsic task outcomes; worker learning can use the configured helpers and curiosity.
- Reward weights are documented hyperparameters, not invented observations or fixed actions. Avoid rewarding mere attack/use button presses or counting ammo expenditure as collection.
- Helpers can be disabled for controlled comparisons. No privileged reward state is passed into the actor or exported inference code.

## Optional RND

- A frozen randomly initialized target and a trainable predictor operate on legal observation features. Prediction error supplies the optional novelty reward; target initialization is fixed for a run and checkpointed.
- Normalize observations/reward statistics without using future rollout samples; cap/weight intrinsic reward and decay its contribution explicitly. Episode boundaries must not attach novelty to an automatically reset replacement observation.
- Train only the predictor against detached target outputs. Store predictor, target, optimizer, and normalization state for resumable training. RND is training-only and unnecessary for evaluation inference.
- Default the feature off for legacy runs. Tests demonstrate a stationary real-observation batch becomes less novel as the predictor learns.

## RGB and exit supervision

- Encode real RGB observations using a small CNN on the planned 60x80 image input; normalize uint8 pixels in the network. Enable actual render buffers when RGB is on.
- Feed learned visual features into policy/recurrent processing and add an auxiliary binary exit-visibility head with a masked training loss.
- The official observation lacks exit identity. Build training-only visibility labels from supported generated-map exit surfaces plus actual camera geometry/occlusion. Include explicit unknown masks for map formats or visibility cases the labeler cannot establish reliably. Do not substitute distance-to-exit, episode success, a texture heuristic, or all-zero labels for visibility ground truth.
- Test supported-map visibility against real engine views with visible, behind-camera, and occluded exits. Keep map metadata and supervision entirely outside submission inputs.
- No detector output or historical result is presented as a measured label or trained success rate.

## Compatibility and scope

- New architecture switches default off when absent from older checkpoints. Existing Phase 2/3 checkpoints retain their established parameter shapes and behavior.
- Any warm start into the expanded architecture is explicit, checks shared parameter shapes, and reports newly initialized heads. Do not silently present random new heads as trained parameters.
- Export only inference-required weights/configuration. Extend existing validation, evaluation, and benchmark paths to exercise the new actor.
- No depth encoder, Stage 4 full-level curriculum, or dashboard UI in this four-feature PR. Expose the active goal and exit score for future dashboard use.
- Preserve unrelated local setup-document and .dockerignore edits outside the feature commits.

## Verification and delivery

Test neural goal conditioning and gradients, exactly eight-decision goal holding, boundary-only manager losses, manager interval returns, rollout replay, reset, legacy checkpoint loading, resumed RND training, RGB effects, label masking, and real-engine visibility. Run real PPO smoke training with all requested features, reload its checkpoint into DoomAgent, play engine episodes, and measure CPU latency. Run regression suites and the official validator; report platform exclusions and warnings.

Record actual run configurations, losses, and measured outcomes. Smoke training proves the pipeline learns and executes; it does not establish improved held-out exit rate. Do not replace or promote the champion without independent evaluation evidence. Request an independent whole-branch review before delivery.

Create the PR with a concrete description, actual verification evidence, and limitations. If the original repository still rejects connector access, use a normal authenticated GitHub flow or report that specific access blocker; never substitute another fork-targeted PR or extract stored credentials.
