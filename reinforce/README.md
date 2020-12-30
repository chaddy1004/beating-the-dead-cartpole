# REINFORCE

REINFORCE is one of the simplest algorithm for policy gradient methods.

Policy Gradient (PG) methods all root from the following main equation:

<img src="https://render.githubusercontent.com/render/math?math=\nabla_{theta} = Q^{\pi](s,a)\pi_{\theta}(a|s)">


where different PG methods are (usually) distinguished by how the value, <img src="https://render.githubusercontent.com/render/math?math=$Q^{\pi](s,a)$ "> is calculated.

REINFORCE approximates that value using Monte-Carlo approach, where the return value, G_t is estimated through the 
trajectory obtained in each episode.

## Approximating Return value, G
Recall:

$G_1 = R_2 + \gammaR_3 + \gamma^2R_4 + \gamma^3R_5 + \gamma^4R_6$
...
$G_4 = R_4 + \gammaR_5 + \gamma^2R_6$
$G_4 = R_5 + \gammaR_6$
$G_5 = R_6$

This can be summarized by the following equation:

$G_t = R_{t+1} + \gammaG_{t+1}]$