import pandas as pd

def validate_data(df, name="Dataset"):
    print(f"\n🔍 Auditing: {name}")
    print("=" * 50)

    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Data types
    print("\nData Types:")
    print(df.dtypes)

    # Duplicate rows
    dupes = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dupes}")

    # Descriptive stats for numeric columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not num_cols.empty:
        print("\nNumeric Summary:")
        print(df[num_cols].describe())

    # Unique values for categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        print("\nCategorical Summary:")
        for col in cat_cols:
            print(f"{col}: {df[col].nunique()} unique values")
            print(df[col].value_counts(dropna=False).head(5))
            print("-" * 20)

    # Check for negative values in time-based features
    time_cols = [col for col in df.columns if 'time' in col.lower()]
    for col in time_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"\n⚠️ Warning: {neg_count} negative values in '{col}'")

    print("=" * 50)
    print("✅ Audit complete.\n")