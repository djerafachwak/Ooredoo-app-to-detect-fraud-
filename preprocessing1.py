import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def full_preprocess(df, df1_mapping):
    # ===========================
    # 1. Jointure avec la table CELL-LAC-WILAYA
    # ===========================
    df = df.copy()
    df1_mapping = df1_mapping.copy()

    # ✅ Assurer la compatibilité des types pour le merge
    df['OPTIONAL_FIELD_2'] = df['OPTIONAL_FIELD_2'].astype(str)
    df1_mapping['Cell'] = df1_mapping['Cell'].astype(str)

    df = df.merge(df1_mapping[['Cell', 'Wilaya']], left_on='OPTIONAL_FIELD_2', right_on='Cell', how='left')
    df.rename(columns={"Wilaya": "Wilaya_code"}, inplace=True)
    df.drop(columns=["Cell"], inplace=True)

    # ===========================
    # 2. Création des colonnes temporelles et historiques
    # ===========================
    df['Wilaya_code'] = df['Wilaya_code'].astype(str).str.zfill(2)
    df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])
    df.sort_values(["PHONE_NUMBER", "TIME_STAMP"], inplace=True)

    df["PREV_PHONE_NUMBER"] = df.groupby("PHONE_NUMBER")["PHONE_NUMBER"].shift(1)
    df["PREV_WILAYA"] = df.groupby("PHONE_NUMBER")["Wilaya_code"].shift(1)
    df["PREV_TIME"] = df.groupby("PHONE_NUMBER")["TIME_STAMP"].shift(1)
    df["PREV_CELL"] = df.groupby("PHONE_NUMBER")["OPTIONAL_FIELD_2"].shift(1)
    df["TIME_DIFF"] = (df["TIME_STAMP"] - df["PREV_TIME"]).dt.total_seconds() / 60

    df["Fraud"] = 0

    # ===========================
    # 3. Détection fraude Type 3 (changement wilaya rapide)
    # ===========================
    cell_wilaya_map = df[['OPTIONAL_FIELD_2', 'Wilaya_code']].drop_duplicates()
    cell_wilaya_count = cell_wilaya_map.groupby('OPTIONAL_FIELD_2')['Wilaya_code'].nunique()
    shared_cells = cell_wilaya_count[cell_wilaya_count > 1].index.astype(str).tolist()

    neighboring_wilayas = {
        '02': ['48', '27', '38', '44', '42'], '03': ['17', '32', '14', '47'], '04': ['25', '43', '24', '41', '05', '40', '12'],
        '05': ['19', '04', '28', '43', '07', '40'], '06': ['15', '10', '34', '19', '18'], '07': ['28', '17', '39', '40', '05'],
        '09': ['16', '10', '35', '26', '44', '42'], '10': ['09', '15', '35', '06', '34', '28', '26'], '12': ['39', '40', '04', '41'],
        '13': ['22', '46', '45'], '14': ['38', '48', '20', '29', '32', '03', '17'], '15': ['06', '10', '35'], '16': ['42', '09', '35', '10'],
        '17': ['28', '07', '26', '47', '03', '14', '38'], '18': ['21', '25', '43', '06', '19'], '19': ['34', '43', '05', '18', '06', '28'],
        '20': ['29', '14', '32', '22'], '21': ['18', '43', '25', '24', '23'], '22': ['29', '20', '32', '45', '13', '26', '31'],
        '23': ['24', '21', '36'], '24': ['21', '23', '36', '41', '04', '25'], '25': ['04', '24', '21', '18', '43'],
        '26': ['09', '44', '38', '17', '28', '10'], '27': ['31', '29', '48', '02'], '28': ['34', '19', '05', '07', '17', '10', '26'],
        '29': ['48', '14', '20', '22', '31', '27'], '31': ['27', '22', '29', '46'], '34': ['19', '06', '10', '28'],
        '35': ['16', '09', '10', '15'], '36': ['24', '41', '23'], '38': ['44', '02', '48', '26', '14', '17'],
        '40': ['12', '07', '05', '04', '39'], '41': ['12', '04', '24', '36'], '42': ['16', '02', '44', '09'],
        '43': ['19', '18', '21', '25', '04', '05'], '44': ['42', '02', '38', '26', '09'], '45': ['32', '20', '22', '13', '08'],
        '46': ['22', '31', '13'], '48': ['02', '27', '29', '14', '38']
    }

    def is_fraud_type_3(row):
        if pd.isna(row["PREV_WILAYA"]) or pd.isna(row["TIME_DIFF"]): return row["Fraud"]
        if row["TIME_DIFF"] > 30 or row["TIME_DIFF"] <= 0: return row["Fraud"]
        if row["PHONE_NUMBER"] != row["PREV_PHONE_NUMBER"]: return row["Fraud"]
        if row["OPTIONAL_FIELD_2"] == row["PREV_CELL"]: return row["Fraud"]
        if str(row["OPTIONAL_FIELD_2"]) in shared_cells: return row["Fraud"]
        if row["Wilaya_code"] != row["PREV_WILAYA"] and row["PREV_WILAYA"] not in neighboring_wilayas.get(row["Wilaya_code"], []):
            return 3
        return row["Fraud"]

    df["Fraud"] = df.apply(is_fraud_type_3, axis=1)

    # ===========================
    # 4. Détection fraude type 5 (spam SMS)
    # ===========================
    sms_out = df[df["CDR_SOURCE"] == "MSS SMSO"]
    unique_dest_per_caller = sms_out.groupby("CALLER_NUMBER")["CALLED_NUMBER"].nunique().reset_index()
    unique_dest_per_caller.columns = ["CALLER_NUMBER", "UNIQUE_SMS_DEST"]
    df = df.merge(unique_dest_per_caller, on="CALLER_NUMBER", how="left").fillna({"UNIQUE_SMS_DEST": 0})

    df.loc[
        (df["CDR_SOURCE"] == "MSS SMSO") & (df["UNIQUE_SMS_DEST"] > 50),
        "Fraud"
    ] = 5

    # ===========================
    # 5. Nettoyage & encodage final
    # ===========================
    df.replace("[NULL]", np.nan, inplace=True)
    df['PREV_WILAYA'] = df['PREV_WILAYA'].fillna('00')
    df['EQUIPMENT_ID'].fillna('Other EQUI', inplace=True)

    # 🔧 Convertir DURATION en float avant de calculer la médiane
    df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
    df["DURATION"].fillna(df["DURATION"].median(), inplace=True)

    df["TIME_DIFF"].fillna(df["TIME_DIFF"].median(), inplace=True)
    df.drop_duplicates(inplace=True)

    # ✅ Ajouter DS_NAME si absent
    if 'DS_NAME' not in df.columns:
        df['DS_NAME'] = 'unknown'

    # ✅ Encodage catégoriel
    df = pd.get_dummies(df, columns=['VPMN', 'DS_NAME', 'CDR_SOURCE', 'PREV_WILAYA', 'Wilaya_code'], drop_first=True)

    df['EQUIPMENT_ID'] = df['EQUIPMENT_ID'].astype(str)
    le = LabelEncoder()
    df['EQUIPMENT_ID'] = le.fit_transform(df['EQUIPMENT_ID'])

    scaler = StandardScaler()
    num_cols = ['DURATION', 'TIME_DIFF', 'UNIQUE_SMS_DEST', 'EQUIPMENT_ID']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df['OPTIONAL_FIELD_2'] = df['OPTIONAL_FIELD_2'].astype(str)
    df = pd.get_dummies(df, columns=['OPTIONAL_FIELD_2'], drop_first=False)

    return df