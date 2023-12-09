import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("data/raw/cancer_diagnosis.csv")

# Drop columns with too much missingness or that are not useful
df.drop(
    columns=[
        "Patient Number",
        "Urine epitheilum (UL)",
        "A/G Ratio",
        "Urine Ketone",
        "Urine Glucose",
        "Strip WBC",
    ],
    inplace=True,
)

# Rename known lab tests with MIMIC-IV itemid's (and correct typos from the raw dataset)
df = df.rename(
    columns={
        "Albumin": "50862",
        "Alk": "50863",
        "ALT (GPT)": "50861",
        "AST (GOT)": "50878",
        "BUN": "51006",
        "Calcium": "50893",
        "Chloride": "50902",
        "Creatinine": "50912",
        "Direct Bilirubin": "50883",
        "Glucose AC": "50931",
        "pH": "51491",
        "Potassium": "50971",
        "Sodium": "50983",
        "Specific Gravity": "51498",
        "Total Bilirubin": "50885",
        "Total Cholesterol": "50907",
        "Total Protein": "50976",
        "Triglyceride": "51000",
        "Urine epithelium count": "51476",
        "Uric acid": "51007",
        "Hyper1en1ion": "Hypertension",
        "Diabe1es": "Diabetes",
        "Bee1leNu1": "BetelNut",
        "FamilyHis1ory": "FamilyHistory",
    }
)

# Encode categorical feature variables
categorical_variables = [
    "FamilyHistory",
    "BetelNut",
    "Drinking",
    "Smoking",
    "Diabetes",
    "Hypertension",
    "gender",
    "Nitrite",
    "Urine Bilirubin",
]
enc = preprocessing.OrdinalEncoder(encoded_missing_value=-1)
df[categorical_variables] = enc.fit_transform(df[categorical_variables])

# Encode the label
le = preprocessing.LabelEncoder()
le.fit(df["Disease"])
df["Disease"] = le.transform(df["Disease"])

# Impute the remaining miss values with the column mean
df = df.fillna(df.mean())

# Save the cleaned dataset
df.to_csv("data/evaluations/cancer_diagnosis_baselines_data.csv", index=False)
