import json, copy

src_path = r'Lab04.COMPAS_exercises.ipynb'
dst_path = r'Lab04.COMPAS_exercises_completed.ipynb'

with open(src_path, 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)

# ── helpers ────────────────────────────────────────────────────────────────
def code_cell(lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines if isinstance(lines, list) else lines.splitlines(keepends=True)
    }

def set_source(cell, text):
    lines = text.splitlines(keepends=True)
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None

# ── solutions ──────────────────────────────────────────────────────────────

cell2 = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("Libraries imported successfully!")
"""

cell4 = """\
df = pd.read_csv('../Data/compas-scores-two-years.csv')

print(f"Shape: {df.shape}")
df.head()
"""

cell6 = """\
print(f"Starting with {len(df)} records")

df = df[(df['days_b_screening_arrest'] >= -30) & (df['days_b_screening_arrest'] <= 30)]
print(f"After days_b_screening_arrest filter: {len(df)} records")

df = df[df['is_recid'] != -1]
print(f"After removing is_recid == -1: {len(df)} records")

df = df[df['c_charge_degree'] != 'O']
print(f"After removing traffic offenses: {len(df)} records")

df = df[df['score_text'] != 'N/A']
print(f"Final dataset: {len(df)} records")
"""

cell8 = """\
print("Distribution of two_year_recid:")
print(df['two_year_recid'].value_counts())
print("\\nPercentages:")
print(df['two_year_recid'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

print("\\nRace distribution:")
print(df['race'].value_counts())
print("\\nPercentages:")
print(df['race'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
"""

cell10 = """\
features = ["juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
X = df[features]
y = df["two_year_recid"]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("\\nBasic statistics of X:")
print(X.describe())
"""

cell12 = """\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

print(f"Training set size: {len(X_train)}")
print(f"Test set size:     {len(X_test)}")
print(f"\\nRecidivism rate in training set: {y_train.mean():.3f}")
print(f"Recidivism rate in test set:     {y_test.mean():.3f}")
"""

cell14 = """\
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

print("Model trained successfully!")
"""

cell16 = """\
y_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {dt_accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Recid', 'Recid'],
            yticklabels=['No Recid', 'Recid'])
plt.title('Confusion Matrix — Decision Tree (depth=3)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print("\\nDecision Tree Structure:")
print(export_text(dt_model, feature_names=features))
"""

cell18 = """\
df['compas_binary'] = (df['decile_score'] > 5).astype(int)

test_indices = y_test.index
compas_test = df.loc[test_indices, 'compas_binary']

agreement = (y_pred == compas_test.values).mean()
print(f"Agreement between Decision Tree and COMPAS: {agreement:.4f}")

compas_accuracy = accuracy_score(y_test, compas_test)
print(f"\\nDecision Tree accuracy: {dt_accuracy:.4f}")
print(f"COMPAS accuracy:         {compas_accuracy:.4f}")
"""

cell20 = """\
from fairlearn.metrics import demographic_parity_difference

race_test = df.loc[y_test.index, 'race']

dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=race_test)
print(f"Demographic Parity Difference: {dpd:.4f}")
print("\\nInterpretation: 0 = perfect parity across groups.")
print(f"A value of {dpd:.4f} means prediction rates differ by that amount across racial groups.")
"""

cell22 = """\
from fairlearn.metrics import false_positive_rate

black_mask = race_test == 'African-American'

fpr_black = false_positive_rate(y_test[black_mask], y_pred[black_mask.values])
fpr_overall = false_positive_rate(y_test, y_pred)

print(f"FPR — Black/African-American: {fpr_black:.4f}")
print(f"FPR — Overall:                {fpr_overall:.4f}")
print(f"Difference:                   {fpr_black - fpr_overall:+.4f}")
print("\\nA higher FPR for Black individuals means they are more often")
print("incorrectly flagged as likely to reoffend when they would not.")
"""

cell24 = """\
from fairlearn.metrics import equalized_odds_difference

eod = equalized_odds_difference(y_test, y_pred, sensitive_features=race_test)
print(f"Equalized Odds Difference: {eod:.4f}")
print("\\nInterpretation: equalized odds requires both TPR and FPR to be equal across groups.")
print(f"A value of {eod:.4f} shows the model does not achieve equalized odds across racial groups.")
"""

cell26 = """\
for depth in [2, 3, 5]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    dpd = demographic_parity_difference(y_test, preds, sensitive_features=race_test)

    print(f"\\nMax Depth = {depth}:")
    print(f"  Accuracy:                      {acc:.4f}")
    print(f"  Demographic Parity Difference: {dpd:.4f}")
"""

cell28 = """\
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_dpd = demographic_parity_difference(y_test, y_pred_rf, sensitive_features=race_test)

print("Random Forest Results:")
print(f"  Accuracy:                      {rf_accuracy:.4f}")
print(f"  Demographic Parity Difference: {rf_dpd:.4f}")
"""

cell30 = """\
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_dpd = demographic_parity_difference(y_test, y_pred_lr, sensitive_features=race_test)

print("Logistic Regression Results:")
print(f"  Accuracy:                      {lr_accuracy:.4f}")
print(f"  Demographic Parity Difference: {lr_dpd:.4f}")
"""

cell32 = """\
results_df = pd.DataFrame({
    'Model': ['Decision Tree (d=3)', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [dt_accuracy, rf_accuracy, lr_accuracy],
    'Dem. Parity Diff': [
        demographic_parity_difference(y_test, y_pred, sensitive_features=race_test),
        rf_dpd,
        lr_dpd,
    ]
})
print(results_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['steelblue', 'forestgreen', 'coral']

ax1.bar(results_df['Model'], results_df['Accuracy'], color=colors)
ax1.set_title('Accuracy Comparison')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.5, 0.8)
ax1.tick_params(axis='x', rotation=15)

ax2.bar(results_df['Model'], results_df['Dem. Parity Diff'].abs(), color=colors)
ax2.set_title('|Demographic Parity Difference|')
ax2.set_ylabel('|DPD|')
ax2.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

print("\\nRecommendation: Logistic Regression tends to offer competitive accuracy with")
print("lower demographic parity difference, making it a good balance for this dataset.")
"""

# Map cell index → solution text
solutions = {
    2: cell2,
    4: cell4,
    6: cell6,
    8: cell8,
    10: cell10,
    12: cell12,
    14: cell14,
    16: cell16,
    18: cell18,
    20: cell20,
    22: cell22,
    24: cell24,
    26: cell26,
    28: cell28,
    30: cell30,
    32: cell32,
}

new_nb = copy.deepcopy(nb)
for idx, text in solutions.items():
    set_source(new_nb['cells'][idx], text)

with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print(f"Written: {dst_path}")
