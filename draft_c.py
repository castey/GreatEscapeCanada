import csv, re
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

# --- Load Canadian migration data ---
cleaned_data = []
with open("data.csv", newline="") as f:
    rows = csv.reader(f)
    for row in rows:
        if row[1] == "Canada" and row[3] in ("Immigrants", "Emigrants"):
            row[0] = re.sub(r"-\d{2}", "", row[0])  # drop quarter suffix
            if "2025" not in row[0]:
                cleaned_data.append([row[0], row[1], row[3], row[10]])

totals = defaultdict(lambda: {"Immigrants": 0, "Emigrants": 0})
for row in cleaned_data:
    year, _, comp, value = row
    totals[year][comp] += int(value)

years = sorted(totals.keys())
years_int = [int(y) for y in years]
immigration = [totals[y]["Immigrants"] for y in years]
emigration = [totals[y]["Emigrants"] for y in years]
net_migration = [totals[y]["Immigrants"] - totals[y]["Emigrants"] for y in years]

can_df = pd.DataFrame({
    "Year": years_int,
    "Immigrants": immigration,
    "Emigrants": emigration,
    "NetMigration": net_migration
})

# --- Extended U.S. draft data (1940–1973) ---
draft_data = {
    1940: 50208, 1941: 923842, 1942: 3033361, 1943: 1915845,
    1944: 1572277, 1945: 945862, 1946: 183383, 1947: 0,
    1948: 202517, 1949: 96208, 1950: 219771, 1951: 551806,
    1952: 474263, 1953: 249555, 1954: 242, 1955: 8758,
    1956: 1378, 1957: 6664, 1958: 6341, 1959: 8716,
    1960: 8175, 1961: 118586, 1962: 82252, 1963: 119373,
    1964:112386, 1965:230991, 1966:382010, 1967:228263,
    1968:296406, 1969:283586, 1970:162746, 1971:94092,
    1972:49514, 1973:646
}
draft_df = pd.DataFrame(list(draft_data.items()), columns=["Year","Inductions"])

# --- Merge datasets ---
merged = pd.merge(can_df, draft_df, on="Year", how="inner")

# Keep only years >= 1951
merged = merged[merged["Year"] >= 1951]
can_df = can_df[can_df["Year"] >= 1951]
draft_df = draft_df[draft_df["Year"] >= 1951]

# --- Subsets (Immigration only for correlation) ---
viet = merged[(merged["Year"]>=1964)&(merged["Year"]<=1973)]
korea = merged[(merged["Year"]>=1951)&(merged["Year"]<=1953)]
combined = pd.concat([viet, korea])

# --- Immigration correlations ---
corr_texts = []
for label, df in [("Vietnam", viet), ("Korea", korea), ("Combined", combined)]:
    if len(df) > 1:
        r, p = pearsonr(df["Immigrants"], df["Inductions"])
        corr_texts.append(f"{label} (Immigration): r={r:.3f}, p={p:.3g}, n={len(df)})")
        print(f"{label} (Immigration): r={r:.3f}, p={p:.3g}, n={len(df)})")
    else:
        corr_texts.append(f"{label}: not enough data (n={len(df)})")
        print(f"{label}: not enough data (n={len(df)})")

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10,6))

# Left axis: Canadian migration
ax1.plot(can_df["Year"], can_df["NetMigration"], marker="o", label="Net Migration")
ax1.plot(can_df["Year"], can_df["Immigrants"], marker="s", label="Immigrants")
ax1.plot(can_df["Year"], can_df["Emigrants"], marker="^", label="Emigrants")
ax1.set_xlabel("Year")
ax1.set_ylabel("Population Change (Canada)")
ax1.tick_params(axis="y")

# Right axis: U.S. Draft inductions (purple line)
ax2 = ax1.twinx()
ax2.plot(draft_df["Year"], draft_df["Inductions"], marker="d", color="purple", label="US Draft Inductions")
ax2.set_ylabel("US Draft Inductions", color="purple")
ax2.tick_params(axis="y", labelcolor="purple")

# Title, grid, ticks
plt.title("Canada Migration vs. U.S. Draft Inductions (Korea & Vietnam) — Immigration correlation")
ax1.set_xticks(range(1951, max(can_df["Year"])+1, 5))
ax1.grid(True, linestyle="--", alpha=0.6)

# Legends (combine both axes)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Bottom text
plt.figtext(
    0.01, 0.02,
    "Source: Statistics Canada (Table 17-10-0040-01) & U.S. Selective Service",
    ha="left", va="bottom", fontsize=8
)
plt.figtext(
    0.99, 0.02,
    "Graph by David Castro",
    ha="right", va="bottom", fontsize=8
)

# --- Add correlation results on graph ---
plt.figtext(
    0.5, 0.15, "\n".join(corr_texts),
    ha="center", va="bottom", fontsize=9,
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
)

# Save
plt.tight_layout()
plt.savefig("canada_draft_immigration_overlay.png", dpi=300)
plt.close()
