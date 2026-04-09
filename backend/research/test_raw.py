import warnings
warnings.filterwarnings('ignore')
from data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

X_all, y = load_data('../../heart_disease_uci.csv', task='binary')
X_train_raw, _, y_train, _ = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(k_neighbors=2, random_state=42)
model = LogisticRegression(random_state=42, max_iter=1000)
pipeline = make_pipeline(StandardScaler(), smote, model)

try:
    pipeline.fit(X_train_raw, y_train)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
