import pandas as pd

# Input data: raw event counts
data_counts = pd.DataFrame({
    "Final state": [
        r"$B^+ \to K^0_{S(LL)}K^+K^-\pi^+$",
        r"$B^+ \to K^0_{S(DD)}K^+K^-\pi^+$",
        r"$B^+ \to K^0_{S(LL)}K^+K^+\pi^-$",
        r"$B^+ \to K^0_{S(DD)}K^+K^+\pi^-$"
    ],
    "2015": [2172108, 8442991, 1076139, 4072609],
    "2016": [15941679, 57675750, 7834911, 27501781],
    "2017": [14114114, 57802730, 6915663, 27345670],
    "2018": [17095051, 69720912, 8416140, 32948949],
    "Total": [49322952, 193642383, 24242853, 91869009]
})

mc_counts = pd.DataFrame({
    "Final state": [
        r"$B^+ \to K^0_{S(LL)}K^+K^-\pi^+$",
        r"$B^+ \to K^0_{S(DD)}K^+K^-\pi^+$",
        r"$B^+ \to K^0_{S(LL)}K^+K^+\pi^-$",
        r"$B^+ \to K^0_{S(DD)}K^+K^+\pi^-$"
    ],
    "2015": [1299, 4507, 41, 154],
    "2016": [12299, 39286, 460, 1659],
    "2017": [13260, 42733, 497, 1798],
    "2018": [11381, 38204, 437, 1757],
    "Total": [38239, 124730, 1435, 5368]
})

def compute_year_percent(df):
    df_percent = df.copy()
    for year in ["2015", "2016", "2017", "2018"]:
        df_percent[year] = (df[year] / df["Total"] * 100).round(2).astype(str) + "%"
    return df_percent.drop(columns=["Total"])

def make_latex(df, caption, label):
    cols = df.columns
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        " & ".join(cols) + r" \\",
        r"\hline"
    ]
    for _, row in df.iterrows():
        line = " & ".join(row) + r" \\"
        lines.append(line)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def make_markdown(df):
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)

# Compute percentages
data_percent = compute_year_percent(data_counts)
mc_percent = compute_year_percent(mc_counts)

# Generate LaTeX output
latex_data = make_latex(data_percent, "Year-to-total percentages for reconstructed data statistics.", "tab:data_percent")
latex_mc = make_latex(mc_percent, "Year-to-total percentages for reconstructed MC statistics.", "tab:mc_percent")

with open("data_percentages.tex", "w") as f:
    f.write(latex_data)

with open("mc_percentages.tex", "w") as f:
    f.write(latex_mc)

# Print Markdown output
print("## Data Year-to-Total Percentages\n")
print(make_markdown(data_percent))
print("\n## MC Year-to-Total Percentages\n")
print(make_markdown(mc_percent))
