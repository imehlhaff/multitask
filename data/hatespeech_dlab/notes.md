# Hate Speech (DLab)

Ordinal features ["sentiment", "respect", "insult", "humiliate", "status", "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"]. Started with 5 ordinal labels each but items with higher intersubjectivity/disagreement were recoded into fewer options.

Hate speech score is computed by MLE on the following:
$$\log\frac{P(\text{rating } k)}{P(\text{rating } k-1)} = \theta_n - \delta_i - \alpha_j - \tau_k$$
$$\log\frac{p_{nijk}}{p_{nijk-1}} = \theta_n - \delta_i - \alpha_j - \tau_k$$

$p_{nijk}$ is the probability that rater $j$ assigns comment $n$ a rating $k$ on survey item $i$. $p_{nijk-1}$ is the same but for rating $k-1$.

