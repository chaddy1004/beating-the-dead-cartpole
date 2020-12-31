# REINFORCE

REINFORCE is one of the simplest algorithm for policy gradient methods.

Policy Gradient (PG) methods all root from the following main equation:
https://latex.codecogs.com/gif.latex?%5Cnabla_%7Btheta%7D%20%3D%20Q%28s%2Ca%29%5Cpi_%7B%5Ctheta%7D%28a%7Cs%29


where different PG methods are (usually) distinguished by how the value, https://latex.codecogs.com/gif.latex?Q%28s%2Ca%29 is calculated.

REINFORCE approximates that value using Monte-Carlo approach, where the return value, https://latex.codecogs.com/gif.latex?G_t is estimated through the 
trajectory obtained in each episode.



## Approximating Return value, G
Recall:
https://latex.codecogs.com/gif.latex?G_t%20%3D%20R_%7Bt&plus;1%7D%20&plus;%20%5Cgamma%20G_%7Bt&plus;1%7D