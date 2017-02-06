import numpy as np

class ContinuumArmedBandit:
	def __init__(self, var_noise=1.0):
		self.var_noise = var_noise
		self.action_space = np.array([-4.0, 5.0])
	
	def fx(self, x):
		fx = np.sin(np.pi/2.0 - 5.0 * x) + np.sin(0.5 * x) + 2.0 * np.sin(2 * x)
		return fx
	
	def action(self, x):
		fx_w_noise = None

		if x <= self.action_space[1] and x >= self.action_space[0]:
			fx = self.fx(x)
			y = np.random.normal(loc=fx, scale=self.var_noise)
		else:
			print('Error: Please choose an action within the action-space')

		return y	
