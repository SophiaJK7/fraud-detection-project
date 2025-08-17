import pandas as pd

def load_raw(raw_dir='data/raw'):
    fraud  = pd.read_csv(f"{raw_dir}/Fraud_Data.csv",
                         parse_dates=["signup_time","purchase_time"])
    ip_map = pd.read_csv(f"{raw_dir}/IpAddress_to_Country.csv")
    credit = pd.read_csv(f"{raw_dir}/creditcard.csv")
    return fraud, ip_map, credit

def clean_fraud(fraud_df, ip_df):
    df = fraud_df.copy()
    # IP→Country merge
    df['ip_int'] = df['ip_address'].astype(int)
    ip_map = ip_df.copy()
    ip_map['lower'] = ip_map['lower_bound_ip_address'].astype(int)
    ip_map = ip_map.sort_values('lower')
    df = pd.merge_asof(
        df.sort_values('ip_int'),
        ip_map[['lower','country']].sort_values('lower'),
        left_on='ip_int', right_on='lower', direction='backward'
    ).drop(columns=['lower','ip_address'])
    return df

def clean_credit(credit_df):
    df = credit_df.drop_duplicates().reset_index(drop=True)
    df.rename(columns={'Class':'class'}, inplace=True)
    return df

def unify_classes(df):
    # Ensure both dataframes use 'class' target
    return df.rename(columns={'Class':'class'}) if 'Class' in df.columns else df

def combine(fraud_df, credit_df):
    return pd.concat([
        unify_classes(fraud_df),
        unify_classes(credit_df)
    ], ignore_index=True)

if __name__ == "__main__":
    fraud_raw, ip_raw, credit_raw = load_raw()
    fraud_clean = clean_fraud(fraud_raw, ip_raw)
    credit_clean= clean_credit(credit_raw)
    fraud_clean.to_csv('data/processed/fraud_clean.csv', index=False)
    credit_clean.to_csv('data/processed/credit_clean.csv', index=False)
    combined = combine(fraud_clean, credit_clean)
    combined.to_csv('data/processed/combined.csv', index=False)
    print("Preprocessing done. Files in data/processed/")