""" Generate ma1 process on sparse pattern and reconstruction with ABD """
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from convtools import cconv, flip
from gradbt import gradbt

# Parameters
n, p0, theta = int(5e5), 2000, 1e-3
p = 3*p0-2

dependency_dists = [500, 900, 1500, 2500, 4000, 6000, None] 
nexps, ndists = 30, len(dependency_dists)


# Define the ABD function
#	(f(Rt(a+t*h)) - f(a)) / t ~= gf(a).dot(h) for all unit vectors h
lda = 0.8/np.sqrt(p0*theta)
def soft(x,L): return np.sign(x) * np.maximum(abs(x)-L, 0)
def f(y,a,L):  return -0.5*la.norm(soft(cconv(flip(y), a), L))**2
def df(y,a,L): return  -cconv(y, soft(cconv(flip(y), a), L))[:p]
def gf(y,a,L): dfa = df(y,a,L); return dfa - dfa.dot(a) * a
def Rt(a): return a / np.sqrt(a.dot(a))

# Run all experiments
results = [[dict() for j in range(nexps)] for i in range(ndists)] 
for i in range(ndists):
	d = dependency_dists[i]
	for j in range(nexps):
		print("\n:::::: Dependency Distance:", d,\
		      "Exp. No.", j+1, "::::::")

		# a0
		a0 = np.random.randn(p0)
		a0 /= la.norm(a0)

		# x0
		if d is None:
			m = n-p0+1
			x0 = (np.random.rand(m) < theta) * np.random.randn(m)
		else:
			m = n-p0+1+d
			x0 = (np.random.rand(m) < theta/2) * np.random.randn(m)	
			x0 = (x0[:-d] + x0[d:])
		x0 = np.concatenate([x0, np.zeros(p0-1)])

		# y, a_init
		y = cconv(a0,x0)
		a_init = y[p0:2*p0]
		a_init /= la.norm(a_init)
		a_init = np.concatenate([np.zeros(p0-1), a_init, np.zeros(p0-1)])

		# run experiment
		a_out = gradbt(lambda a: f(y,a,lda), lambda a: gf(y,a,lda),\
		               lambda a: Rt(a), a_init)
		results[i][j]['a0'] = a0
		results[i][j]['x0'] = x0
		results[i][j]['a_init'] = a_init
		results[i][j]['a_out'] = a_out

np.save('results_abd_on_dpdtx0.npy', results)

# Calculate average correlation
avg_corr = [0 for i in range(ndists)]
var_corr = [0 for i in range(ndists)]
for i in range(ndists):
	maxcorrs = []
	for j in range(nexps):
		a_out = results[i][j]['a_out']
		a0 = results[i][j]['a0']
		maxcorrs.append(max(abs(np.correlate(a_out,a0,'full'))))
	maxcorrs = np.array(sorted(maxcorrs)[2:-2])
	avg_corr[i] = sum(maxcorrs) / len(maxcorrs)
	var_corr[i] = np.sqrt(sum((maxcorrs - avg_corr[i])**2))

xdata = list(dependency_dists[:-1]) + [10000]
xdatalabel = list(dependency_dists[:-1]) + ['iid.']
plt.errorbar(xdata, avg_corr, var_corr); 
plt.xscale("log")
plt.xticks(xdata, xdatalabel)
plt.xlabel("Dependent entries distance")
plt.ylabel("Largest shict-coherence")
plt.title("Effect of support distance in sparse pattern")
plt.draw(); plt.pause(1e-3)
plt.savefig('results_abd_on_dpdtx0.pdf', transparent=True)



