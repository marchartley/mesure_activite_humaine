import os
import sys

class HiddenPrints:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self._original_stdout = sys.stdout
        self._silent_stdout = os.devnull

    def __enter__(self):
        pass
        if self.verbose:
            # if sys.stdout is self._silent_stdout:
            #     sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            sys.stdout = open(self._silent_stdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        # try:
        #     sys.stdout.close()
        # except:
        #     pass
        sys.stdout = self._original_stdout
