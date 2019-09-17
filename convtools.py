"""
Tools for circular convolution, using fft, adopted to different signal length.
"""
import numpy as np

def cconv(a,b):
	"""Circular convolution of 1D np.arrays, zero filling for short array"""
	n = max(len(a),len(b))
	if len(a) < n: 
		a = np.concatenate((a,np.zeros(n-len(a))))
	elif len(b) < n: 
		b = np.concatenate((b,np.zeros(n-len(b))))
	return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

def flip(y):
	return np.concatenate( (y[0:1], y[:0:-1]) )