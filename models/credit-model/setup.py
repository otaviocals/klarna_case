from distutils.core import setup

setup(
    name="credit_model",
    version="1.0",
    description="Klarna Credit Model",
    author="Otavio Cals",
    packages=["credit-model"],
    install_requires=["catboost", "kserve"],
)
