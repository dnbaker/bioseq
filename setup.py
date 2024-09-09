from setuptools import setup, Extension, find_packages, distutils
from sys import platform
from setuptools.command.build_ext import build_ext
from glob import glob
import multiprocessing
import subprocess


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """

    for flag in ("-std=c++%s" % x for x in ("2a", 17, 14, 11, "03")):
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')

full_gomp_path = subprocess.check_output("realpath `$CXX --print-file-name=libgomp.a`", shell=True).decode('utf-8').strip()

extra_compile_args = ['-march=native',
                      '-Wno-char-subscripts', '-Wno-unused-function', '-Wno-ignored-qualifiers',
                      '-Wno-strict-aliasing', '-Wno-ignored-attributes', '-fno-wrapv',
                      '-Wall', '-Wextra', '-Wformat',
                      '-lz', '-fopenmp',
                      "-pipe", '-O0', '-DNDEBUG']

extra_link_opts = ["-fopenmp", "-lz"]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if platform == 'darwin':
        darwin_opts = ['-mmacosx-version-min=10.7']# , '-libstd=libc++']
        # darwin_opts = []
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts


    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_compile_args += extra_compile_args
            ext.extra_link_args = link_opts + extra_link_opts
            ext.extra_objects = [full_gomp_path, "spoa/lib/libspoa.a"]
        build_ext.build_extensions(self)


def build_spoa():
    import subprocess
    import os
    os.makedirs("spoa/build", exist_ok=True)
    subprocess.check_call("cd spoa/build && cmake .. && cd .. && make", shell=True)

if __name__ == "__main__":
    __version__ = "0.1.8"
    build_spoa()
    include_dirs = [get_pybind_include(), get_pybind_include(True), "./", "spoa/include"]
    ext_modules = [Extension("cbioseq", ["src/bioseq.cpp", "src/poa.cpp", "src/tokenize.cpp", "src/omp.cpp", 'src/fxstats.cpp'], include_dirs=include_dirs, language='c++')]
    setup(
        name='bioseq',
        version=__version__,
        author='Daniel Baker',
        author_email='dnb@cs.jhu.edu',
        url='https://github.com/dnbaker/bioseq',
        description='A python module for tokenizing biological sequences',
        long_description='',
        ext_modules=ext_modules,
        install_requires=['pybind11', 'numpy>=0.19', 'einops', 'torch', 'fast_transformer_pytorch', 'x-transformers'],
        setup_requires=['pybind11'],
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        packages=find_packages(),
        scripts=['scripts/flatten_swiss', 'scripts/makeflatfile']
    )
