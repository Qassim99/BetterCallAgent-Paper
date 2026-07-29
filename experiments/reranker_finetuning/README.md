# Reranker fine-tuning (exploratory)

The audited SFT/GRPO work is not part of the default paper pipeline:

- the measured SFT gain was approximately `0.008` on a ten-query validation split;
- GRPO did not provide a reliable improvement;
- earlier rewards implemented an obsolete citation rule; and
- the available training/evaluation language distributions were mismatched.

For those reasons, generated training files and predecessor training scripts are excluded
from the clean release. A future reproduction should first define a held-out split,
align the reward with `bettercallagent.citations.policy`, pin the complete training
environment, and publish model/data cards. Keeping this placeholder is more honest
than presenting unvalidated experimental code as a paper component.
