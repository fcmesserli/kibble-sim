from setuptools import setup

setup(
    name="kibble-sim",
    url="https://github.com/fcmesserli/kibble-sim",
    author="John Ladan",
    author_email="jladan@uwaterloo.ca",
    # Needed to actually package something
    packages=["moving_mode"],
    # Needed for dependencies
    install_requires=["numpy", "scipy"]
    version="0.1",
    license="MIT",
    description="An example of a python package from pre-existing code",
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
