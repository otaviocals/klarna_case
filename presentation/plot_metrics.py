import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings

warnings.filterwarnings("ignore")


def ABS_SHAP(df_shap, df):
    # import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop("index", axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(
        0
    )
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["Variable", "Corr"]
    corr_df["Sign"] = np.where(corr_df["Corr"] > 0, "deeppink", "royalblue")

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["Variable", "SHAP_abs"]
    k2 = k.merge(corr_df, left_on="Variable", right_on="Variable", how="inner")
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    colorlist = k2["Sign"]
    ax = k2.plot.barh(
        x="Variable", y="SHAP_abs", color=colorlist, figsize=(5, 9), legend=False
    )
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.savefig("summary_plot2.png", bbox_inches="tight", dpi=600)
    plt.close()


# Load Train Data
train_data = pd.read_csv("train_data.csv")

# Load Metrics
metrics = pd.read_csv("metrics.csv")

# Load model
model = joblib.load("model.joblib")

# Get Operators
preproc = model.get_params()["preproc"]
final_model = model.get_params()["model__model"]

# Trandform Data
fitted_train_data = preproc.transform(train_data)
fitted_train_data = fitted_train_data[model.get_params()["featselect__select_features"]]

metrics = metrics.sort_values(by="roc_auc")
#final_model = final_model.get_params()["base_estimator"]
# Get SHAP Values
shap_values = shap.TreeExplainer(final_model).shap_values(fitted_train_data)

# Plot Shap
f = plt.figure()
shap.summary_plot(shap_values, fitted_train_data, show=False)
f.savefig("summary_plot1.png", bbox_inches="tight", dpi=600)
plt.close()

ABS_SHAP(shap_values, fitted_train_data)

# Plot Metrics
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(6, 7))
f.suptitle("Metrics")
ax1.bar(metrics["model"], metrics["roc_auc"])
ax1.set_ylabel("ROC AUC")
ax1.set_ylim(0.5, 1.0)
ax2.bar(metrics["model"], metrics["recall"])
ax2.set_ylabel("RECALL")
ax2.set_ylim(0.0, 1.0)
ax3.bar(metrics["model"], metrics["f1"])
ax3.set_ylabel("F1")
ax3.set_ylim(0.0, 1.0)
ax4.bar(metrics["model"], metrics["precision"])
ax4.set_ylabel("Precision")
ax4.set_ylim(0.0, 1.0)
f.savefig("metrics.png", bbox_inches="tight", dpi=600)
plt.close()
