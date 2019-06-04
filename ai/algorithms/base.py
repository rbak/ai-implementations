LINE_LENGTH = 20

class BaseAlgorithm(object):
	hyper_parameter_data = {
		'alpha': {
			'default': .05,
			'description': 'The learning rate.',
			'range': '0-1'
			},
		'gamma': {
			'default': 1,
			'description': 'The discount rate.',
			'range': '0-1'
			}
	}

	def __init__(self):
		pass

	def run(self, **kwargs):
		self._load_hyper_parameters(**kwargs)
		self._print_run()

	def finish_run(self, **kwargs):
		self.env.visualize()
		print('='*LINE_LENGTH)

	def _load_hyper_parameters(self, **kwargs):
		for hp in self.hyper_parameters:
			if hp in kwargs:
				setattr(self, hp, kwargs[hp])
			else:
				setattr(self, hp, self.hyper_parameter_data[hp]['default'])

	def _print_run(self):
		print('='*LINE_LENGTH)
		print('Algorithm: {}'.format(type(self).__name__))
		print('Hyper Parameters:')
		for p in self.hyper_parameters:
			print('  {}: {}'.format(p, getattr(self, p)))
		print('-'*LINE_LENGTH)
		print('Environment: {}'.format(type(self.env).__name__))
		print('-'*LINE_LENGTH)
