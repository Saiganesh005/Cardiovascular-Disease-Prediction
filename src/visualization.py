import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df):

    plt.figure(figsize=(12,8))

    sns.heatmap(df.corr(), annot=True)

    plt.savefig("output/images/correlation_heatmap.png")
