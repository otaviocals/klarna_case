from distutils.core import setup

setup(
    name="credit-model",
    version="1.0",
    description="Klarna Credit Model",
    author="Otavio Cals",
    packages=["credit_model"],
    install_requires=["catboost", "kserve"],
)
