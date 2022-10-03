class Runtime():
    
    pass

class PythonRuntime(Runtime):
    shapely = True

class NumbaRuntime(Runtime):
    pass

class CythonRuntime(Runtime):
    pass

class NumpyRuntime(Runtime):
    pass

class JAXRuntime(Runtime):
    pass

class GlobalRuntime():
    
    runtime = PythonRuntime

    @classmethod
    def set_mode(cls, runtime):
        if type(runtime) is str:
            if runtime == 'python':
                cls.runtime = PythonRuntime
            elif runtime == 'numba':
                cls.runtime = NumbaRuntime
            elif runtime == 'cython':
                cls.runtime = CythonRuntime
            elif runtime == 'numpy':
                cls.runtime = NumpyRuntime
            elif runtime == 'jax':
                cls.runtime = JAXRuntime
        else:
            cls.runtime = runtime

    @classmethod
    def is_python_runtime(cls):
        return cls.runtime is PythonRuntime
    
    @classmethod
    def is_numba_runtime(cls):
        return cls.runtime is NumbaRuntime
    
    @classmethod
    def is_cython_runtime(cls):
        return cls.runtime is CythonRuntime
    
    @classmethod
    def is_numpy_runtime(cls):
        return cls.runtime is NumpyRuntime

    @classmethod
    def is_jax_runtime(cls):
        return cls.runtime is JAXRuntime