from distutils.core import setup

setup(
    name="credit_model",
    version="1.0",
    description="Klarna Credit Model",
    author="Otavio Cals",
    packages=["credit_model"],
    include_package_data=True,
    install_requires=["catboost", "kserve==0.7.0", "xgboost", "awswrangler"],
)
