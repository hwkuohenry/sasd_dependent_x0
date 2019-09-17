""" Gradient descent with backtracking """
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def gradbt(f, gf, Rt, x, niter=500, isplot=False):
	"""
	Parameters
	---------- 
	f :  func: x -> float - objective
	gf : func: x -> array - riemannian gradient 
	Rt : func: x -> array - retraction map
	x: array     - initial vector
	niter: int   - max iteration number
	isplot: bool - plot on/off 

	Returns
	------- 
	x_out: array - local minimizer of f
	"""

	t, fx, gfx = 1e-4, f(x), gf(x)
	gfxnorm = la.norm(gfx)
	iiter = 0

	print("----- Begin iteration -----")
	if isplot:
		fig, axes = plt.subplots(2,1)
		fxs, gfxnorms, iiters = [], [], []
		lines = [None]*2
		axes[0].set_title("f(x)");
		lines[0], = axes[0].plot(iiters,fxs)
		axes[1].set_title("||gf(x)||")
		lines[1], = axes[1].plot(iiters,gfxnorms)
		
	while gfxnorm > 1e-4 and iiter < niter:
		t *= 2
		x_n = Rt(x - t * gfx)
		fx_n = f(x_n) 
		while fx_n > fx - 0.5 * t * gfxnorm**2:
			t /= 2
			x_n = Rt(x - t * gfx)
			fx_n = f(x_n) 
		x, fx, gfx = x_n, fx_n, gf(x_n)
		gfxnorm = la.norm(gfx)
		iiter += 1

		if iiter % 5 == 0:
			print("iter.", '%4d:' % iiter, "f(x) =",'%.2e,' % fx,\
			      "||gf(x)|| =", '%.2e,' % gfxnorm, 't =', '%.2e' % t)
			if isplot:
				fxs.append(fx)
				gfxnorms.append(gfxnorm)
				iiters.append(iiter)
				lines[0].set_data(iiters, fxs)
				lines[1].set_data(iiters, gfxnorms)
				for i in range(len(axes)): 
					axes[i].relim(); axes[i].autoscale_view()	
				plt.tight_layout(); plt.plot(); plt.pause(1e-17);	

	print("iter.", '%4d:' % iiter, "f(x) =",'%.2e,' % fx,\
			      "||gf(x)|| =", '%.2e,' % gfxnorm, 't =', '%.2e' % t)
	print("------ End iteration ------")

	return x
