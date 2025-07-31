import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compare_distributions_by_class(
    df: pd.DataFrame,
    class_col: str = 'class',
    dataset_name: str = 'Dataset',
    output_dir: str = None
):
    """
    Plot boxplots and KDEs of all numeric features split by class.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    class_col : str
        Name of the target/class column.
    dataset_name : str
        Label to prefix titles and filenames.
    output_dir : str, optional
        If provided, saves each plot to this directory.
    """
    # Prepare output folder
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Select numeric columns (excluding the class column)
    numeric_cols = (
        df.select_dtypes(include='number')
          .drop(columns=[class_col], errors='ignore')
          .columns
    )

    print(f"\n🔎 EDA on {dataset_name} — comparing {len(numeric_cols)} numeric features by '{class_col}'\n")
    for feature in numeric_cols:
        print(f"• Feature: {feature}")

        # Boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(x=class_col, y=feature, data=df, palette='Set3')
        plt.title(f"{dataset_name}: {feature} by {class_col} (Boxplot)")
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{feature}_boxplot.png"))
        plt.show()

        # KDE
        plt.figure(figsize=(6,4))
        sns.kdeplot(data=df, x=feature, hue=class_col, common_norm=False)
        plt.title(f"{dataset_name}: {feature} by {class_col} (KDE)")
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{feature}_kde.png"))
        plt.show()