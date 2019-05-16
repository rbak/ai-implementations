import ai.algorithms as algorithms
import ai.environments as envs

import cmd
import sys

class AIShell(cmd.Cmd):
    intro = 'Type help or ? to list commands.\n'
    prompt = '(ai) '
    file = None

    def do_run(self, arg):
        """Run an algorithm: RUN MonteCarloPrediction Blackjack

        If the algorithm and environment are not specified, you will be given some to choose from.
        """
        algorithm = None
        environment = None
        args = arg.split()
        if len(args) >= 1:
            if not hasattr(algorithms, args[0]):
                print('ERROR: The algorithm %s is not supported'.format(args[0]))
            else:
                algorithm = getattr(algorithms, args[0])
        if len(args) >= 2:
            if not hasattr(envs, args[1]):
                print('ERROR: The environment %s is not supported'.format(args[1]))
            else:
                environment = getattr(envs, args[1])

        self.run(algorithm, environment)

    def do_list_algorithms(self, arg):
        'List all available algorithms'
        for module in algorithms.__all__:
            print(module)

    def do_list_environments(self, arg):
        'List all available environments'
        for module in envs.__all__:
            print(module)

    def do_describe_algorithms(self, arg):
        'List all available algorithms:  RIGHT 20'
        pass

    def do_describe_environments(self, arg):
        'Turn all available environments:  RIGHT 20'
        # right(*parse(arg))
        pass

    def do_exit(self, arg):
        'Exit the shell'
        return True

# ------------------------------

    def run(self, alg_type, env_type):
        with env_type() as env:
            alg_instance = alg_type(env)
            alg_instance.run()

    def list_algorithms(self, filters):
        return []

    def list_environments(self, filters):
        return []

if __name__ == '__main__':
    AIShell().cmdloop()


