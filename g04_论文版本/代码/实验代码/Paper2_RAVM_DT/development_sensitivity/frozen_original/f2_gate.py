"""The unique new gate. Exactly three current Verification scalars; no provider capability."""
from math import isfinite
def joint_gate(d,parent_mean,p,weight,margin):
 if not all(map(isfinite,(d,parent_mean,p,weight,margin))) or parent_mean<0 or weight<0 or margin<0 or not 0<=p<=1:raise ValueError('Invalid joint-gate input')
 j=d+weight*max(parent_mean,1e-12)*(p-.2)
 return bool(d<0 and j<=-margin),j
