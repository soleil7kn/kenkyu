import numpy as np
import matplotlib.pyplot as plt

setting = "ETTh1_96_96_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0"

pred = np.load(f"./results/{setting}/pred.npy")
true = np.load(f"./results/{setting}/true.npy")

print(pred.shape)
print(true.shape)

feature_idx = 0
sample_idx = 0

plt.figure(figsize=(12,6))

plt.plot(
    pred[sample_idx, :, feature_idx],
    label="Prediction"
)

plt.plot(
    true[sample_idx, :, feature_idx],
    label="Ground Truth"
)

plt.title(f"ETTh1 Forecast (Feature {feature_idx})")

plt.xlabel("Time Step")
plt.ylabel("Value")

plt.legend(loc="upper right")
plt.grid()

plt.savefig(
    "prediction_result.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()