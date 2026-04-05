import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("fig1_silver_trend.csv")

plt.rcParams["font.family"] = "serif"

# Match the paper-like visual style in the reference image
fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor="#e7e7e7")
ax.set_facecolor("#e7e7e7")

ax.plot(
	df["Year"],
	df["Web_of_Science"],
	color="#0a8a6a",
	marker="s",
	markersize=7,
	linewidth=2,
	label="WOS"
)

ax.plot(
	df["Year"],
	df["Scopus"],
	color="#e35a00",
	marker="o",
	markersize=7,
	linewidth=2,
	label="Scopus"
)

ax.set_xlabel("Year", fontsize=18)
ax.set_ylabel("No. of publications", fontsize=18)

# Show every 2 years to mimic the reference figure axis density
xticks = df["Year"][::2]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks, fontsize=11)
ax.tick_params(axis="y", labelsize=11)

legend = ax.legend(
	loc="upper left",
	frameon=True,
	fontsize=12,
	fancybox=False,
	borderaxespad=1,
	handlelength=2.8,
	handletextpad=0.5,
)
legend.get_frame().set_facecolor("#f1f1f1")
legend.get_frame().set_edgecolor("#444444")
legend.get_frame().set_linewidth(0.8)

for spine in ax.spines.values():
	spine.set_linewidth(1.1)
	spine.set_color("#222222")

ax.grid(False)
fig.tight_layout()

# Save
fig.savefig("results/figures/fig1_publication_trend_clean.png", dpi=300)
plt.show()
