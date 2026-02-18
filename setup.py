import sys
import os
from setuptools import setup
from setuptools.extension import Extension
from pathlib import Path
import importlib

version = '1.2.4'

from setuptools.command.build_ext import build_ext

ext = Extension(
    "_pyvoronoi",
    sources=["src/bindings.cpp", "src/voronoi.cpp"],
    include_dirs=["src"],
    language="c++",
    optional=os.environ.get('CIBUILDWHEEL', '0') != '1',
)


# This command has been borrowed from
# http://www.pydanny.com/python-dot-py-tricks.html

if sys.argv[-1] == 'tag':
    tag_command = "git tag -a v%s -m 'v%s'" % (version, version)
    print(tag_command)
    os.system(tag_command)
    os.system("git push --tags")
    sys.exit()

class build_ext_subclass( build_ext ):
    def build_extensions(self):
        pybind11 = importlib.import_module("pybind11")
        pybind11_include = pybind11.get_include()
        for e in self.extensions:
            if pybind11_include not in e.include_dirs:
                e.include_dirs.append(pybind11_include)
        print(f'Compiler type: {self.compiler.compiler_type} - Version: {sys.version_info.major}.{sys.version_info.minor}')
        # Starting from 3.11, the file longintrepr.h has moved. It is no longer under Python@3.XX\include but Python@3.XX\include\cpython        
        if sys.version_info.major == 3 and sys.version_info.minor >= 11:
            c = self.compiler.compiler_type
            if c == 'msvc':
                for e in self.extensions:
                    install_dir = os.path.dirname(sys.executable)
                    cpython_directory = os.path.join(install_dir, 'include', 'cpython')         
                    e.extra_compile_args = [cpython_directory]
        build_ext.build_extensions(self)

      

this_directory = Path(__file__).parent
setup(
    name='pyvoronoi',
    python_requires='>=3.8',
    version=version,
    description='pybind11 wrapper for the Boost Voronoi library (version 1.59.0)',
    long_description=(this_directory / "README.md").read_text(),
    long_description_content_type='text/markdown',
    author='Fabien Ancelin / Andrii Sydorchuk, Voxel8',
    author_email='',
    url='https://github.com/fabanc/pyvoronoi',
    keywords=['voronoi','Boost','polygon'],
    packages=['pyvoronoi'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: C++",
        "Environment :: Other Environment",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=[ext],
    cmdclass={
        'build_ext': build_ext_subclass
    },   
)
