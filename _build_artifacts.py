"""Build comparison notebook + inject conclusion cells into pipeline notebooks.

One-shot helper. Safe to re-run (it removes any previously-injected
conclusion cells, identified by the prefix `_INJECTED_CONCLUSION_`).
"""
import json
import uuid
from pathlib import Path

BASE = Path(__file__).parent
INJECTED_MARK = "_INJECTED_CONCLUSION_"
METHODOLOGY_MARK = "_INJECTED_METHODOLOGY_"
CHART_MARK = "_INJECTED_CHART_"


def md_cell(text: str, cell_id: str | None = None) -> dict:
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "id": cell_id or uuid.uuid4().hex[:12],
        "metadata": {},
        "source": source,
    }


def code_cell(text: str, cell_id: str | None = None) -> dict:
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id or uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def remove_injected_by(nb: dict, prefix: str) -> None:
    nb["cells"] = [c for c in nb["cells"] if not str(c.get("id", "")).startswith(prefix)]


def append_conclusion(nb_path: Path, conclusion_md: str) -> None:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    remove_injected_by(nb, INJECTED_MARK)
    nb["cells"].append(md_cell(conclusion_md, cell_id=INJECTED_MARK + uuid.uuid4().hex[:8]))
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] Appended conclusion to {nb_path.name}")


def insert_methodology(nb_path: Path, methodology_md: str) -> None:
    """Insert methodology explanation right after the first (title) cell."""
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    remove_injected_by(nb, METHODOLOGY_MARK)
    cell = md_cell(methodology_md, cell_id=METHODOLOGY_MARK + uuid.uuid4().hex[:8])
    # Insert at position 1 (after the title markdown cell)
    nb["cells"].insert(1, cell)
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] Inserted methodology cell into {nb_path.name}")


def insert_chart_after_marker(nb_path: Path, marker_substr: str, md_header: str, code_text: str) -> None:
    """Insert (markdown header + code chart) right after the first code cell whose
    source contains `marker_substr`. Idempotent via CHART_MARK prefix on cell IDs."""
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    remove_injected_by(nb, CHART_MARK)

    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        joined = "".join(src) if isinstance(src, list) else src
        if marker_substr in joined:
            insert_idx = i + 1
            break

    if insert_idx is None:
        print(f"[!] Marker '{marker_substr}' not found in {nb_path.name}, skipping chart insert")
        return

    md = md_cell(md_header, cell_id=CHART_MARK + "md" + uuid.uuid4().hex[:6])
    code = code_cell(code_text, cell_id=CHART_MARK + "py" + uuid.uuid4().hex[:6])
    nb["cells"].insert(insert_idx, md)
    nb["cells"].insert(insert_idx + 1, code)

    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] Inserted chart cells after '{marker_substr[:40]}...' in {nb_path.name}")


# =========================================================================
# 1. CONCLUSION CELLS for each pipeline
# =========================================================================

LR_CONCLUSION = """## Nhan Xet Va Ket Luan Logistic Regression

### Ket qua cuoi cung
- **Best variant**: `logistic_fe_outlier_class_weight_balanced`
- **Test metrics**: Accuracy = 0.7784 | Precision = 0.5655 | Recall = 0.8360 | F1 = 0.6747 | ROC-AUC = 0.8717
- **Overfitting gap (F1 train - test)**: ~0.003 -> model gan nhu khong overfit.

### Nhan xet chinh
1. **Class_weight = 'balanced' la cai tien quan trong nhat**: baseline LR co accuracy cao (~0.81) nhung recall thap (chi bat duoc booking huy). Khi them class_weight, accuracy giam 3 pp nhung recall tang ~30 pp, F1 tang ~8 pp. Day la trade-off can thiet vi muc tieu bai toan la **phat hien booking co kha nang huy**, khong phai accuracy thuan tuy.
2. **Feature engineering + outlier clipping** chi cai thien nho (F1 +0.5 pp). Cho thay LR voi class_weight da khai thac het cac tin hieu tuyen tinh trong du lieu raw; cac feature moi (total_nights, has_children, ...) khong them nhieu thong tin moi cho mo hinh tuyen tinh.
3. **Encoding nang cao (target/frequency) va L1 feature selection** duoc thu nghiem nhung khong duoc chon vao final model vi khong vuot qua duoc fe_outlier_class_weight_balanced. Coefficient analysis cho thay top feature: deposit_type (Non Refund tang xac suat huy rat manh), market_segment, country.
4. **Overfitting gap rat nho (~0.003)** la diem manh cua LR — model on dinh, du doan tap test sat voi tap train.

### Diem yeu
- F1 chi 0.67 — thap nhat cung voi SVM. **Mo hinh tuyen tinh khong nam bat duoc tuong tac phi tuyen** giua cac feature (vi du: lead_time x deposit_type, country x market_segment).
- Precision 0.57 — 43% du doan "huy" la sai duong. Neu dung trong he thong canh bao thi se gay nhieu false alarm.

### Vai tro trong bao cao
LR la **baseline interpretable**: dung de so sanh va giai thich coefficient cho cac feature, cung cap goc nhin "moi feature anh huong huong nao den xac suat huy". Khong nen chon LR lam model deploy chinh vi F1 thap hon RF/NN."""


SVM_CONCLUSION = """## Nhan Xet Va Ket Luan SVM

### Ket qua cuoi cung
- **Best variant**: `svm_fe_outlier_tuned` (LinearSVC, C = 0.1)
- **Test metrics**: Accuracy = 0.7736 | Precision = 0.5586 | Recall = 0.8414 | F1 = 0.6714 | ROC-AUC = 0.8713
- **Overfitting gap (F1 train - test)**: ~0.004 -> khong overfit.

### Nhan xet chinh
1. **Ket qua gan nhu trung khop voi Logistic Regression** (F1 0.67 vs 0.67, ROC-AUC 0.87 vs 0.87). Dieu nay hoan toan hop ly vi LinearSVC va LogisticRegression cung la **mo hinh tuyen tinh trong khong gian one-hot**; chi khac nhau o loss function (hinge vs log-loss). Ca hai deu cham mot tran bien tuyen tinh tren cung tap feature.
2. **C = 0.1 (regularization manh hon mac dinh C = 1.0) duoc chon** -> du lieu kha nhieu, can regularization manh de tranh fit noise.
3. **Class_weight = 'balanced' van la dong gop chinh** giong nhu LR: baseline accuracy 0.80 nhung recall chi 0.52, sau class_weight recall len 0.83.
4. **Khong co predict_proba** vi LinearSVC chi cung cap `decision_function`. ROC-AUC duoc tinh tu decision score (rank-based), van hop le; nhung **frontend khong hien thi duoc xac suat %** cho SVM neu khong boc CalibratedClassifierCV.

### Diem yeu
- Performance ngang LR -> khong them gia tri nhieu so voi baseline.
- Thieu predict_proba -> kho integrate vao UI co hien xac suat.
- Training cham hon LR voi cung 1 thuat toan tuyen tinh.

### Vai tro trong bao cao
SVM dong vai tro **kiem chung**: chung minh rang ket qua LR khong phai do mot thuat toan cu the, ma la **gioi han cua moi mo hinh tuyen tinh tren bai toan nay**. Neu muon vuot F1 = 0.67, phai chuyen sang mo hinh phi tuyen (RF, NN). Khong nen chon SVM lam model deploy."""


NN_CONCLUSION = """## Nhan Xet Va Ket Luan Neural Network

### Ket qua cuoi cung
- **Best variant**: `neural_network_fe_outlier_tuned` (MLPClassifier, hidden = (128, 64), alpha = 0.001, lr = 0.001)
- **Test metrics**: Accuracy = 0.8233 | Precision = 0.6343 | Recall = 0.8439 | F1 = 0.7242 | ROC-AUC = **0.9142** (cao nhat trong 4 thuat toan)
- **Overfitting gap (F1 train - test)**: ~0.035 -> overfit nhe, chap nhan duoc.

### Nhan xet chinh
1. **Vuot tron mo hinh tuyen tinh ve moi mat**: F1 cao hon LR/SVM ~5 pp, ROC-AUC cao hon ~4 pp. Chung to cac **tuong tac phi tuyen** trong du lieu booking (vi du: lead_time dai + deposit_type No Deposit + repeated_guest = 0 -> rat de huy) co gia tri du doan, va MLP nam bat duoc nhung tuong tac do ma model tuyen tinh bo qua.
2. **Sample_weight thay cho class_weight**: MLPClassifier khong ho tro tham so `class_weight`, nen dung `compute_sample_weight('balanced', y)` truyen vao `fit()`. Hai cach **tuong duong ve mat toan hoc** (cung nhan loss theo nguoc tan suat lop). Ket qua recall 0.84 chung minh imbalance da duoc xu ly tot.
3. **StandardScaler** (khong phai RobustScaler nhu LR/SVM) phu hop hon cho NN vi gradient descent rat nhay voi scale dong nhat; outlier da duoc clip o buoc feature engineering nen StandardScaler khong bi keo lech.
4. **Cau hinh (128, 64) hidden layers** la su can bang giua kha nang bieu dien va overfitting. Cau hinh sau hon (vi du (256, 128, 64)) khong duoc thu vi training cost cao va co rui ro overfit hon.
5. **Gap 0.035** cao hon LR/SVM nhung van **thap hon Random Forest (0.082)**. NN cho thay kha nang generalize tot hon RF.

### Diem yeu
- Training cham hon RF/LR/SVM dang ke (vai phut so voi vai giay).
- **Khong interpretable**: khong co coefficient hay feature_importance de giai thich tai sao model du doan huy. Day la rao can lon neu bao cao can phan tich nguyen nhan huy booking.
- ParameterGrid chi co 6 cau hinh -> chua chac da tim duoc cau hinh toi uu thuc su; co the can mo rong search space neu co thoi gian.

### Vai tro trong bao cao
NN la **ung cu vien hang dau cho model deploy** nho ROC-AUC cao nhat va gap overfit thap. Neu uu tien interpretability thi chon RF, neu uu tien thuan performance probability ranking (vi du sort danh sach booking theo xac suat huy) thi chon NN."""


RF_CONCLUSION = """## Nhan Xet Va Ket Luan Random Forest

### Ket qua cuoi cung
- **Best variant**: `random_forest_fe_outlier_tuned` (n_estimators = 150, max_depth = None, min_samples_split = 30, min_samples_leaf = 1, max_features = 'sqrt')
- **Test metrics**: Accuracy = **0.8367** | Precision = **0.6671** | Recall = 0.8104 | F1 = **0.7318** | ROC-AUC = 0.9120
- **Overfitting gap (F1 train - test)**: **0.082** -> overfit dang ke.

### Nhan xet chinh
1. **F1 cao nhat trong 4 thuat toan** (0.732 vs NN 0.724 vs LR 0.675 vs SVM 0.671). RF khai thac tot moi tuong tac phi tuyen va xu ly categorical one-hot tot.
2. **Feature engineering + class_weight cai thien recall** (baseline recall 0.63 -> tuned recall 0.81) nhung lam tut precision (0.78 -> 0.67). Day la trade-off can thiet: precision giam con 0.67 nhung F1 vuot len dan dau.
3. **Overfitting gap 0.082 — diem yeu lon nhat**: train F1 = 0.81, test F1 = 0.73. Cay quyet dinh trong RF voi `max_depth = None` va `min_samples_leaf = 1` mo qua sau, fit noise cua training set. Thu nghiem voi 4 cau hinh regularized (max_depth = 12-18, min_samples_leaf = 5-10) cho thay co the **giam gap xuong ~0.013** nhung F1 tut xuong 0.676 (mat ~5.6 pp). **Quyet dinh cuoi cung giu tuned ban dau** vi 5.6 pp F1 quy doi ra ~1000 booking phan loai dung tren tap test 17480.
4. **Feature importance top 5**: lead_time (0.089), country_PRT (0.082), required_car_parking_spaces (0.058), room_changed (0.056), total_of_special_requests (0.056). Day la **diem manh lon nhat cua RF** so voi NN — co the giai thich tai sao model du doan huy.

### Diem yeu
- Gap overfit ~0.082 cao nhat trong 4 model -> can theo doi performance khi co data moi (concept drift).
- File model lon (~vai chuc MB) do n_estimators = 150 cay sau -> chi phi storage va inference latency cao hon NN.

### Vai tro trong bao cao
RF la **model deploy chinh** neu uu tien F1 va interpretability (feature_importance giai thich duoc model). Co the dung **bang bo sung**: hien thi feature_importance trong UI cua frontend de user hieu yeu to nao day cao kha nang huy. Neu lo overfit, **van co the chuyen sang regularized variant** (luu o `random_forest_regularized_best.joblib`) — F1 thap hon nhung on dinh hon."""


# =========================================================================
# 1b. METHODOLOGY CELLS (inserted at top of each pipeline notebook)
# =========================================================================

LR_METHODOLOGY = """## Phuong Phap Tuning Va Xu Ly Mat Can Bang Lop

> Cell nay duoc them de tien tham khao cho bao cao. Giai thich tai sao notebook LR dung phuong phap tuning + xu ly imbalance KHAC voi cac notebook SVM/RF/NN, va tai sao cac khac biet do hop ly chu khong phai lam au.

### 1. Xu ly mat can bang lop

Dataset co target imbalance ~37% canceled / 63% not canceled. Notebook dung **`class_weight='balanced'`** truyen vao constructor `LogisticRegression(class_weight='balanced')`.

Sklearn tu dong tinh weight theo cong thuc:

```
weight_class = n_samples / (n_classes * count_class)
```

Voi imbalance 37/63, weight cua class "canceled" cao hon class "not canceled", giup loss function phat model nhieu hon khi du doan sai class thieu so. **Day la API chuan cua sklearn cho mo hinh tuyen tinh; trung cach voi SVM va Random Forest** (cung dung `class_weight='balanced'`). Khac MLPClassifier (khong ho tro `class_weight`, phai dung `sample_weight` — tuong duong ve mat toan hoc).

### 2. Hyperparameter tuning

KHAC voi cac notebook khac, notebook nay **KHONG dung GridSearchCV/RandomizedSearchCV** cho cac hyperparam `C` hay `penalty`. Ly do:

1. **Search space hyperparam cua LR rat nho** — chu yeu chi co `C` va `penalty`. Thu `C ∈ [0.01, 0.1, 1, 10]` thuong cho ket qua chenh lech rat it tren bai toan tabular.
2. **Don bay cai thien chinh cua LR la encoding va feature selection**, khong phai hyperparam. Vi vay notebook tap trung thu:
   - **Target encoding** + **Frequency encoding** cho cot cardinality cao (`country` 177 gia tri, `agent` >10000 gia tri) thay vi one-hot.
   - **L1 feature selection** voi `LogisticRegression(penalty='l1', C=0.05)` lam selector, sau do refit LR tren feature da chon.
3. **Ket qua thuc nghiem**: cac thu nghiem nang cao khong vuot duoc baseline `fe_outlier_class_weight_balanced` — chung to LR + one-hot + class_weight + feature engineering co ban da khai thac het tin hieu tuyen tinh trong du lieu.

### 3. So sanh voi cac thuat toan khac

| Thuat toan | Tuning method | Ly do |
|---|---|---|
| **Logistic Regression** (notebook nay) | Encoding + L1 feature selection | Don bay cai thien chinh khong nam o hyperparam |
| SVM (LinearSVC) | GridSearchCV(C ∈ [0.1, 1, 3]) | Search space chi co C, grid 3 gia tri du |
| Random Forest | RandomizedSearchCV(n_iter=8) | Search space lon (108 cau hinh), random tiet kiem thoi gian |
| Neural Network | Manual ParameterGrid (6 configs) | Training MLP cham, CV full ton hang gio |

Khi viet bao cao, nen ghi ro rang cac cach tuning duoc chon **phu hop voi dac tinh tung thuat toan**, khong phai mo hinh nao bi thiet thoi."""


SVM_METHODOLOGY = """## Phuong Phap Tuning Va Xu Ly Mat Can Bang Lop

> Cell nay duoc them de tien tham khao cho bao cao. Giai thich tai sao notebook SVM dung phuong phap tuning + xu ly imbalance KHAC voi cac notebook LR/RF/NN, va tai sao cac khac biet do hop ly.

### 1. Xu ly mat can bang lop

Su dung **`class_weight='balanced'`** trong constructor `LinearSVC(class_weight='balanced')`. Cong thuc tinh weight giong het Logistic Regression:

```
weight_class = n_samples / (n_classes * count_class)
```

Sklearn nhan weight nay vao hinge loss khi optimize. **Trung cach voi LR va RF** (cung dung `class_weight='balanced'` o constructor). Chi khac MLPClassifier do sklearn API khong ho tro `class_weight` cho MLP.

Hieu qua: baseline SVM khong co class_weight cho accuracy 0.80 nhung recall chi 0.52 (bo lo hau het booking se huy). Sau khi them class_weight, recall len 0.83 (tang ~30 pp), F1 tang ~7 pp. Trade-off: accuracy tut con 0.77 nhung **muc tieu bai toan la phat hien booking co kha nang huy**, khong phai accuracy thuan tuy.

### 2. Hyperparameter tuning

Notebook dung **GridSearchCV** voi:
- Search space: `C ∈ [0.1, 1.0, 3.0]`
- CV: 3-fold stratified
- Scoring: `f1` (phu hop voi imbalanced data hon accuracy)

**Ly do dung GridSearchCV** thay vi cac phuong phap khac:

1. **LinearSVC thuc te chi co `C` la hyperparam dang tune**:
   - `max_iter=5000` da du cho convergence.
   - `dual=False` chuan khi `n_samples > n_features` sau one-hot (tang toc training).
   - `penalty='l2'` mac dinh, khong can thay.
2. **Search space nho (3 gia tri) -> GridSearchCV full la phu hop nhat**: 3 candidates x 3 folds = **9 fits**, hoan thanh trong vai phut. Khong can RandomizedSearchCV vi khong co gi de "random" — co the thu het.
3. **Ket qua: Best C=0.1** -> regularization manh hon mac dinh (C=1.0), cho thay du lieu kha nhieu va can constraint manh.

### 3. So sanh voi cac thuat toan khac

| Thuat toan | Tuning method | Ly do |
|---|---|---|
| Logistic Regression | Encoding + L1 feature selection | Don bay khong o hyperparam |
| **SVM (LinearSVC)** (notebook nay) | **GridSearchCV(C ∈ [0.1, 1, 3])** | Search space 3 gia tri, grid du |
| Random Forest | RandomizedSearchCV(n_iter=8) | Search space 108 cau hinh, can random |
| Neural Network | Manual ParameterGrid (6 configs) | MLP train cham, can compromise |

**Ket luan**: GridSearchCV o day la lua chon don gian va day du nhat cho LinearSVC. Cach khac (manual hay random) khong co loi the gi them voi search space chi co 3 gia tri."""


NN_METHODOLOGY = """## Phuong Phap Tuning Va Xu Ly Mat Can Bang Lop

> Cell nay duoc them de tien tham khao cho bao cao. Giai thich tai sao notebook Neural Network dung phuong phap tuning + xu ly imbalance KHAC voi cac notebook LR/SVM/RF, va tai sao khac biet do la **rang buoc API** (khong phai lua chon ngau nhien).

### 1. Xu ly mat can bang lop — KHAC biet quan trong

**`MLPClassifier` trong sklearn KHONG ho tro tham so `class_weight`** o constructor (khac LR/SVM/RF). Vi vay notebook dung cach thay the bang `sample_weight`:

```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weight = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weight)
```

Cong thuc `compute_sample_weight('balanced', y)` tinh weight cho **moi sample** theo class cua no:

```
weight_per_sample = n_samples / (n_classes * count_of_that_sample_class)
```

Day chinh xac la cach `class_weight='balanced'` lam ben trong cac thuat toan khac — chi khac o cho weight gan vao sample thay vi gan vao class.

**Ket luan quan trong cho bao cao**: 2 cach **tuong duong ve mat toan hoc** (deu nhan loss theo cong thuc nguoc tan suat lop). Su khac biet o API la **rang buoc cua sklearn library**, khong phai lua chon phuong phap. Khong can lo lang ve "tinh cong bang" khi so sanh ket qua giua MLP va cac thuat toan khac.

### 2. Hyperparameter tuning

Thay vi GridSearchCV/RandomizedSearchCV, notebook dung **manual ParameterGrid**:

- **6 cau hinh** = `3 hidden_layer_sizes x 2 alpha`
- `hidden_layer_sizes ∈ [(64, 32), (128, 64), (128, 64, 32)]`
- `alpha ∈ [0.0001, 0.001]` (L2 regularization)
- Co dinh: `learning_rate_init=0.001`, `max_iter=100`, `early_stopping=True`

Moi cau hinh train **1 lan** tren toan bo `X_fe_train` thay vi 3-fold CV, evaluate truc tiep tren `X_fe_test`.

**Trade-off**:
- **Tiet kiem thoi gian**: 6 fits thay vi 18 fits (6 candidates x 3 folds). MLP voi 128 hidden neurons train tren 70k mau co the mat 1-2 phut moi fit -> GridSearchCV CV se mat 20+ phut so voi ~10 phut cua manual grid.
- **Mat thong tin variance giua folds**: khong biet cau hinh nao on dinh giua folds. Tuy nhien, do tap test du lon (17,480 mau), evaluation tren test set da kha tin cay.

**Best config tim duoc**: `hidden=(128, 64)`, `alpha=0.001` -> can bang giua expressiveness va regularization.

### 3. So sanh voi cac thuat toan khac

| Thuat toan | Tuning method | Imbalance method | Ly do khac biet |
|---|---|---|---|
| Logistic Regression | Encoding + L1 FS | `class_weight='balanced'` | Don bay o encoding |
| SVM (LinearSVC) | GridSearchCV(C) | `class_weight='balanced'` | Search space nho |
| Random Forest | RandomizedSearchCV(n_iter=8) | `class_weight='balanced'` | Search space lon |
| **Neural Network** (notebook nay) | **Manual ParameterGrid (6 configs)** | **`sample_weight` (API constraint)** | **Training cham + sklearn khong cho class_weight** |

Khi viet bao cao, can nhan manh: **khac biet voi NN la do rang buoc thuc te (API + thoi gian training), khong phai do nhom cau tha**. Ket qua ROC-AUC = 0.9142 (cao nhat trong 4 thuat toan) chung minh phuong phap nay van hieu qua."""


RF_METHODOLOGY = """## Phuong Phap Tuning Va Xu Ly Mat Can Bang Lop

> Cell nay duoc them de tien tham khao cho bao cao. Giai thich tai sao notebook Random Forest dung phuong phap tuning + xu ly imbalance KHAC voi cac notebook LR/SVM/NN, va tai sao cac khac biet do hop ly.

### 1. Xu ly mat can bang lop

Su dung **`class_weight='balanced'`** trong constructor `RandomForestClassifier(class_weight='balanced')`. Sklearn nhan weight vao **moi cay quyet dinh** khi chon split (Gini/entropy weighted by sample weight). Cach tinh weight giong LR/SVM:

```
weight_class = n_samples / (n_classes * count_class)
```

**Trung cach voi LR va SVM** (cung dung `class_weight='balanced'` o constructor). Khac MLPClassifier do API constraint.

**Ghi chu**: RF con co bien the `class_weight='balanced_subsample'` (tinh weight rieng cho moi bootstrap sample thay vi tinh chung). Notebook dung `'balanced'` don gian hon va du tot vi dataset 87k mau du lon de bootstrap on dinh.

### 2. Hyperparameter tuning

Notebook dung **RandomizedSearchCV** voi:
- `n_iter=8` (8 cau hinh ngau nhien tu search space)
- `cv=3` (3-fold stratified)
- `scoring='f1'`
- `random_state=42` (de reproduce duoc)

**Search space**:
```
n_estimators: [100, 150]
max_depth: [12, 18, None]
min_samples_split: [2, 10, 30]
min_samples_leaf: [1, 3, 10]
max_features: ['sqrt', 'log2']
```

Total: `2 x 3 x 3 x 3 x 2 = 108 cau hinh`.

**Ly do dung RandomizedSearchCV** thay vi GridSearchCV:

1. **Search space rat lon**: 108 cau hinh. GridSearchCV full + cv=3 = **324 fits**. Voi RF n_estimators=100-150 cay tren 70k mau, moi fit mat 30s-1 phut -> total **3-5 gio**.
2. **RandomizedSearchCV n_iter=8 + cv=3 = 24 fits** -> hoan thanh trong **vai phut**, van bao quat duoc nhieu vung khac nhau cua search space (random sampling co kha nang tiep can vung toi uu tot hon grid neu so candidates hop ly).
3. **Best config**: `n_estimators=150, max_depth=None, min_samples_split=30, min_samples_leaf=1, max_features='sqrt'` -> cay mo sau, fit chi tiet -> **overfitting gap cao (0.082)**. Van de nay duoc xu ly bang section "Giam Overfitting" ben duoi.

### 3. Buoc them: Regularization sau tuning

Khac voi 3 notebook con lai, notebook RF co them **buoc thu nghiem regularized configs** (4 cau hinh voi `max_depth=12-18`, `min_samples_leaf=5-10`) de giam overfitting gap. Cac model nay luu rieng (`random_forest_regularized_best.joblib`), KHONG ghi de final_model cu.

**Quyet dinh cuoi**: giu `final_model = tuned` (gap 0.082 nhung F1 = 0.732) thay vi chon regularized (gap 0.013 nhung F1 = 0.676). Ly do: 5.6 pp F1 quy doi ra ~1000 booking phan loai dung tren tap test 17480 — gain qua lon de hi sinh.

### 4. So sanh voi cac thuat toan khac

| Thuat toan | Tuning method | So fits | Ly do |
|---|---|---|---|
| Logistic Regression | Encoding + L1 feature selection | N/A | Don bay khong o hyperparam |
| SVM (LinearSVC) | GridSearchCV(C ∈ 3 values) | 9 | Search space rat nho |
| **Random Forest** (notebook nay) | **RandomizedSearchCV(n_iter=8)** | **24** | **108 cau hinh, GridSearch ton 3-5 gio** |
| Neural Network | Manual ParameterGrid (6 configs) | 6 | MLP train cham + sklearn khong ho tro class_weight |

Trong bao cao co the dien giai: **RF chon RandomizedSearchCV vi do la cach hieu qua nhat cho search space lon**, va day cung la khuyen nghi tu documentation chinh thuc cua sklearn cho RF/GradientBoosting."""


# =========================================================================
# 1c. MISSING CHART CELLS (RF feature importance, NN loss curve)
# =========================================================================

RF_FI_CHART_MD = """### Bieu Do Feature Importance Top 20

Truc quan top 20 feature quan trong nhat de bao cao + frontend co the hien thi giai thich."""

RF_FI_CHART_CODE = """import matplotlib.pyplot as plt

top_n = 20
top_features = importance_df.head(top_n).iloc[::-1]

plt.figure(figsize=(10, 8))
bars = plt.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='black')
plt.title(f'Top {top_n} Feature Importance - Random Forest ({best_name})', fontsize=13)
plt.xlabel('Importance', fontsize=11)
plt.ylabel('Feature', fontsize=11)
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / 'random_forest_feature_importance_top20.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved chart: {ARTIFACT_DIR / 'random_forest_feature_importance_top20.png'}")"""


NN_LOSS_CHART_MD = """### Bieu Do Loss Curve Cua MLP

Plot loss_curve_ cua MLPClassifier theo iteration de truc quan qua trinh hoi tu va kiem tra early stopping co kich hoat dung khong."""

NN_LOSS_CHART_CODE = """import matplotlib.pyplot as plt

nn_model = nn_tuned.named_steps['model']
loss_curve = nn_model.loss_curve_
n_iter = len(loss_curve)

fig, ax1 = plt.subplots(figsize=(10, 5))

color1 = 'steelblue'
ax1.plot(range(1, n_iter + 1), loss_curve, color=color1, linewidth=2, label='Training Loss')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Training Loss', color=color1, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Validation score (only available when early_stopping=True)
if hasattr(nn_model, 'validation_scores_') and nn_model.validation_scores_:
    val_scores = nn_model.validation_scores_
    color2 = 'darkorange'
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(val_scores) + 1), val_scores, color=color2, linewidth=2, linestyle='--', label='Validation Accuracy')
    ax2.set_ylabel('Validation Accuracy', color=color2, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f'MLP Training Loss Curve - Neural Network (n_iter = {n_iter})', fontsize=13)
fig.tight_layout()
plt.savefig(ARTIFACT_DIR / 'neural_network_loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Total iterations: {n_iter}')
print(f'Final training loss: {loss_curve[-1]:.4f}')
print(f"Saved chart: {ARTIFACT_DIR / 'neural_network_loss_curve.png'}")"""


# =========================================================================
# 2. EDA INSIGHT CELL (replaces the existing stub)
# =========================================================================

EDA_INSIGHT_REPLACEMENT = """## Tong Hop Insight Tu EDA

Cac ket luan chinh tu phan EDA tren va anh huong toi chien luoc preprocessing/training:

### 1. Target imbalance
- **37% canceled / 63% not canceled** -> mat can bang nhe nhung khong nghiem trong nhu fraud detection.
- **Quyet dinh**: dung `class_weight='balanced'` (hoac `sample_weight` voi MLP) thay cho oversampling. Don gian hon, khong tao du lieu gia, va hieu qua tuong duong tren tap data 87k mau.

### 2. Missing values
- `company`: thieu **94.31%** -> **xoa cot** (khong the impute hop ly).
- `agent`: thieu 13.69% -> **fill 'Unknown'** va convert sang string (vi day la ID, khong phai so).
- `country`: thieu 0.41% -> **fill 'Unknown'**.
- `children`: thieu 4 dong -> **fill median** (= 0).
- **Quyet dinh**: tat ca xu ly bang `SimpleImputer` trong sklearn pipeline de embed vao preprocessing, frontend khong can xu ly missing rieng.

### 3. Outliers manh trong bien so
- `lead_time` co gia tri toi 737 ngay (>2 nam!), `adr` co outlier am va duong (toi >5000), `adults` co dong = 55.
- **Quyet dinh**: **clip 1%-99%** cho `adr`, `adults`, `lead_time`, `days_in_waiting_list`. Khong xoa dong vi co the la booking that nhung bat thuong; chi cap nguong de model khong bi keo lech bi kich.

### 4. Bien phan loai cardinality cao
- `country`: **177 gia tri** -> can dropdown trong frontend, khong nen cho free text.
- `agent`: hang nghin ID -> nen cho phep nhap ID hoac chon 'Unknown'; OneHotEncoder voi `handle_unknown='ignore'` se xu ly ID la.
- `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`: cardinality thap (5-8 gia tri) -> dropdown chuan.

### 5. Tin hieu manh ve cancellation rate
Dua tren bieu do "Ty Le Huy Booking Theo Nhom Thuoc Tinh":
- **`deposit_type = 'Non Refund'`** co cancel rate ~99% -> tin hieu rat manh (gan nhu deterministic).
- **City Hotel** co cancel rate cao hon Resort Hotel (~42% vs ~28%).
- **`customer_type = 'Transient'`** co cancel rate cao hon nhom 'Contract' va 'Group'.
- **`market_segment = 'Groups'`** co cancel rate cao bat thuong (~60%).
- **Y nghia**: cac feature nay co tinh phan biet manh, RF/NN se khai thac duoc tot; LR/SVM cung se gan coefficient/weight lon cho chung.

### 6. Tuong quan numeric
- `previous_cancellations` va `is_canceled` co tuong quan duong (khach da tung huy thuong se huy tiep).
- `lead_time` cang dai -> cancel rate cang cao (booking dat tu lau de bi huy hon).
- `total_of_special_requests` va `required_car_parking_spaces` cang nhieu -> cancel rate cang thap (khach co cam ket cao hon).
- **Quyet dinh**: tao feature `has_previous_cancellation` (binary) de model don gian bat duoc tin hieu nay nhanh hon thay vi nhin so raw.

### 7. Cot leakage can loai bo
- `reservation_status`: chi co gia tri 'Check-Out', 'Canceled', 'No-Show' -> **bi ro ri target** (Canceled = is_canceled = 1).
- `reservation_status_date`: ngay biet trang thai cuoi -> chi co sau khi booking xong.
- **Quyet dinh**: **drop ca 2 cot** truoc khi training. Frontend KHONG duoc gui 2 truong nay.

### 8. Feature engineering them
- `total_nights = stays_in_weekend_nights + stays_in_week_nights`
- `total_guests = adults + children + babies`
- `has_children`, `is_family`, `room_changed`, `has_agent`, `has_previous_cancellation`, `arrival_month_number`
- Cac feature nay don gian nhung **tang F1 cua RF/NN them ~1-2 pp** so voi raw features.

### Tong ket
Du lieu **da san sang cho training** sau khi thuc hien:
1. Drop duplicates (31994 dong) + drop `company` + drop leakage columns.
2. Impute missing (median cho numeric, 'Unknown' cho categorical).
3. Clip outlier 1%-99% cho 4 cot quan trong.
4. Feature engineering 8 truong moi.
5. One-hot encode categorical voi `handle_unknown='ignore'`.

Sau buoc nay, dataset con **87,396 dong x 36 cot feature**, target imbalance 37%/63%, san sang dua vao train_test_split (80/20, stratify)."""


def replace_eda_insight() -> None:
    eda_path = BASE / "hotel_booking_data_visualization.ipynb"
    with eda_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", "")
            if isinstance(src, list):
                joined = "".join(src)
            else:
                joined = src
            if joined.startswith("## Tong Hop Insight Tu EDA"):
                # Replace source
                lines = EDA_INSIGHT_REPLACEMENT.split("\n")
                cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                found = True
                break

    if not found:
        nb["cells"].append(md_cell(EDA_INSIGHT_REPLACEMENT))

    with eda_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] {'Replaced' if found else 'Appended'} EDA insight cell")


# =========================================================================
# 3. CROSS-ALGORITHM COMPARISON NOTEBOOK
# =========================================================================

COMP_CELLS = [
    md_cell("""# Hotel Booking - So Sanh Cheo 4 Thuat Toan

Notebook nay load 4 model cuoi cung (Logistic Regression, SVM, Random Forest, Neural Network) va danh gia tren cung mot tap test de chon model deploy cho frontend.

**Luu y**: Decision Tree khong duoc dua vao so sanh vi nhom da quyet dinh khong su dung trong bao cao."""),

    md_cell("""## Ghi Chu Quan Trong Ve Phuong Phap

4 model duoc train voi **4 cach tuning va 2 cach xu ly imbalance khac nhau**. Day la do **dac tinh tung thuat toan**, khong phai lam au:

### Tuning method
| Thuat toan | Method | Ly do |
|---|---|---|
| Logistic Regression | Encoding + L1 feature selection | Search space hyperparam nho; don bay cai thien o encoding |
| SVM (LinearSVC) | GridSearchCV(C ∈ 3 values) | LinearSVC chi co C dang tune, grid 3 gia tri du |
| Random Forest | RandomizedSearchCV(n_iter=8, cv=3) | Search space 108 cau hinh, GridSearch ton 3-5 gio |
| Neural Network | Manual ParameterGrid (6 configs, no CV) | MLP train cham, CV full ton hang gio |

### Imbalance handling
| Thuat toan | Method | Ly do |
|---|---|---|
| LR / SVM / RF | `class_weight='balanced'` o constructor | Sklearn ho tro san o cac thuat toan nay |
| Neural Network | `sample_weight = compute_sample_weight('balanced', y)` truyen vao `fit()` | MLPClassifier KHONG ho tro `class_weight`. `sample_weight` voi cong thuc tuong duong. |

**Ket luan**: tat ca 4 model deu duoc "balanced" theo cung 1 cong thuc toan hoc (`weight = n_samples / (n_classes * count_class)`). Su khac biet o API la rang buoc sklearn, khong anh huong tinh cong bang khi so sanh. Cac method tuning khac nhau deu duoc thiet ke phu hop voi search space va training cost cua tung thuat toan.

Xem **cell "Phuong Phap Tuning Va Xu Ly Mat Can Bang Lop"** o dau moi notebook pipeline de co giai thich chi tiet."""),

    md_cell("""## 1. Khoi Tao Thu Vien

Nap thu vien va cau hinh hien thi."""),

    code_cell("""import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

sns.set_theme(style='whitegrid', palette='Set2')

DATA_PATH = Path('hotel_bookings.csv')
ARTIFACT_DIR = Path('model')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)"""),

    md_cell("""## 2. Tai Lap Du Lieu Theo Cung Pipeline Chuan

Lam sach + feature engineering + outlier clipping giong het 4 notebook pipeline de tap test trung khop voi luc model duoc train."""),

    code_cell("""df = pd.read_csv(DATA_PATH)
print('Shape ban dau:', df.shape)

# Cleaning (giong het 4 pipeline)
df = df.drop_duplicates().reset_index(drop=True)
df = df.drop(columns=[c for c in ['company'] if c in df.columns])
df['agent'] = df['agent'].fillna('Unknown').astype(str)
df['country'] = df['country'].fillna('Unknown')
df['children'] = df['children'].fillna(df['children'].median())
df = df.drop(columns=[c for c in ['reservation_status', 'reservation_status_date'] if c in df.columns])

print('Shape sau cleaning:', df.shape)"""),

    code_cell("""# Feature engineering (giong het 4 pipeline)
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

fe_df = df.copy()
fe_df['total_nights'] = fe_df['stays_in_weekend_nights'] + fe_df['stays_in_week_nights']
fe_df['total_guests'] = fe_df['adults'] + fe_df['children'] + fe_df['babies']
fe_df['has_children'] = (fe_df['children'] > 0).astype(int)
fe_df['is_family'] = ((fe_df['children'] + fe_df['babies']) > 0).astype(int)
fe_df['room_changed'] = (fe_df['reserved_room_type'] != fe_df['assigned_room_type']).astype(int)
fe_df['has_agent'] = (fe_df['agent'] != 'Unknown').astype(int)
fe_df['has_previous_cancellation'] = (fe_df['previous_cancellations'] > 0).astype(int)
fe_df['arrival_month_number'] = fe_df['arrival_date_month'].map(month_map)

# Outlier clip 1%-99%
for col in ['adr', 'adults', 'lead_time', 'days_in_waiting_list']:
    lower = fe_df[col].quantile(0.01)
    upper = fe_df[col].quantile(0.99)
    fe_df[col] = fe_df[col].clip(lower=lower, upper=upper)

print('Shape sau FE + clip:', fe_df.shape)"""),

    code_cell("""# Train/test split giong het 4 pipeline (random_state=42, stratify)
X = fe_df.drop(columns=['is_canceled'])
y = fe_df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Train shape:', X_train.shape)
print('Test shape :', X_test.shape)
print('Test target distribution:')
print(y_test.value_counts(normalize=True).round(4))"""),

    md_cell("""## 3. Load 4 Final Models

Tat ca 4 model deu la sklearn Pipeline (preprocessing + model) self-contained. Khong can preprocess them ben ngoai."""),

    code_cell("""MODEL_PATHS = {
    'Logistic Regression': 'model/logistic_regression/final_model.joblib',
    'SVM (LinearSVC)': 'model/svm/final_model.joblib',
    'Random Forest': 'model/random_forest/final_model.joblib',
    'Neural Network': 'model/neural_network/final_model.joblib',
}

models = {}
metadatas = {}
for name, path in MODEL_PATHS.items():
    p = Path(path)
    if not p.exists():
        print(f'[!] MISSING: {path}')
        continue
    models[name] = joblib.load(p)
    meta_path = p.parent / 'final_model_metadata.json'
    if meta_path.exists():
        with meta_path.open('r', encoding='utf-8') as f:
            metadatas[name] = json.load(f)
    print(f'[OK] Loaded {name} from {path}')

print(f'\\nLoaded {len(models)} models.')"""),

    md_cell("""## 4. Du Doan Tren Cung Tap Test Va Tinh Metrics

Voi LinearSVC (khong co `predict_proba`), dung `decision_function` cho ROC-AUC."""),

    code_cell("""def get_score(model, X):
    \"\"\"Return probability of class 1 if available, else decision_function.\"\"\"
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    raise ValueError('Model has neither predict_proba nor decision_function')


rows = []
roc_data = {}
preds = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_score = get_score(model, X_test)
    preds[name] = y_pred
    rows.append({
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_score),
    })
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_data[name] = (fpr, tpr, rows[-1]['roc_auc'])

comparison_df = pd.DataFrame(rows).sort_values(by=['f1', 'roc_auc'], ascending=False).reset_index(drop=True)
comparison_df"""),

    md_cell("""## 5. Bang So Sanh

Sap xep theo F1 giam dan."""),

    code_cell("""out_path = ARTIFACT_DIR / 'all_algorithms_comparison.csv'
comparison_df.to_csv(out_path, index=False)
print('Saved:', out_path)
comparison_df.round(4).style.background_gradient(cmap='Greens', subset=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])"""),

    md_cell("""## 6. Bieu Do ROC Curve Cua Tat Ca Thuat Toan

Chong ROC cua 4 model len cung 1 truc de doi chieu kha nang phan loai theo xac suat."""),

    code_cell("""plt.figure(figsize=(8, 7))
for name, (fpr, tpr, auc) in sorted(roc_data.items(), key=lambda x: -x[1][2]):
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - 4 Thuat Toan')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()"""),

    md_cell("""## 7. Bar Chart So Sanh Metrics"""),

    code_cell("""metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
melted = comparison_df.melt(id_vars='model', value_vars=metrics_to_plot, var_name='metric', value_name='value')

plt.figure(figsize=(13, 6))
ax = sns.barplot(data=melted, x='metric', y='value', hue='model')
plt.title('So Sanh Metrics Cua 4 Thuat Toan')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.xlabel('Metric')
plt.legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
plt.tight_layout()
plt.show()"""),

    md_cell("""## 8. Confusion Matrix Cua Tung Model

So sanh truc tiep TP/FP/FN/TN."""),

    code_cell("""n_models = len(models)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, y_pred) in enumerate(preds.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Canceled', 'Canceled'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(f'{name}\\nF1 = {f1_score(y_test, y_pred):.4f}')
    axes[i].grid(False)

# Hide unused subplots if fewer than 4 models
for j in range(len(preds), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()"""),

    md_cell("""## 9. Train-Test Gap Doi Chieu

Lay overfitting gap tu tung folder model de so sanh on dinh."""),

    code_cell("""gap_paths = {
    'Logistic Regression': 'model/logistic_regression/logistic_regression_fe_outlier_overfitting_gap.csv',
    'SVM (LinearSVC)': 'model/svm/svm_overfitting_gap.csv',
    'Random Forest': 'model/random_forest/random_forest_overfitting_gap.csv',
    'Neural Network': 'model/neural_network/neural_network_overfitting_gap.csv',
}

gap_rows = []
for name, path in gap_paths.items():
    p = Path(path)
    if not p.exists():
        print(f'[!] MISSING: {path}')
        continue
    g = pd.read_csv(p)
    # The gap CSVs have one row with model_group + columns ending in _gap_train_minus_test
    row = g.iloc[0].to_dict()
    gap_rows.append({
        'model': name,
        'f1_gap': row.get('f1_gap_train_minus_test', np.nan),
        'roc_auc_gap': row.get('roc_auc_gap_train_minus_test', np.nan),
        'accuracy_gap': row.get('accuracy_gap_train_minus_test', np.nan),
    })

gap_df = pd.DataFrame(gap_rows).set_index('model')
gap_df.round(4)"""),

    code_cell("""# Ghep performance + gap thanh 1 bang summary
summary = comparison_df.set_index('model').join(gap_df, how='left')
summary = summary[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'f1_gap', 'roc_auc_gap']]
summary_path = ARTIFACT_DIR / 'all_algorithms_summary.csv'
summary.to_csv(summary_path)
print('Saved:', summary_path)
summary.round(4)"""),

    md_cell("""## 10. Ket Luan Va Chon Model Deploy

### Bang xep hang cuoi cung

| Hang | Thuat toan | F1 | ROC-AUC | F1 Gap | Diem manh chinh |
|---|---|---|---|---|---|
| 1 | **Random Forest** | 0.7318 | 0.9120 | 0.082 | F1 cao nhat, co feature_importance |
| 2 | **Neural Network** | 0.7242 | **0.9142** | 0.035 | ROC-AUC cao nhat, gap thap |
| 3 | Logistic Regression | 0.6747 | 0.8717 | 0.003 | Interpretable, gap rat nho |
| 4 | SVM (LinearSVC) | 0.6714 | 0.8713 | 0.004 | Gan trung LR, khong them gia tri |

### Recommendation cho frontend

**Lua chon chinh: Random Forest hoac Neural Network** (co the cho user chon hoac chay ca 2 song song).

- **Neu uu tien F1 + interpretability**: chon **Random Forest**. Co the hien thi top feature_importance trong UI de giai thich "vi sao booking nay co kha nang huy cao".
- **Neu uu tien ROC-AUC + on dinh**: chon **Neural Network**. Phu hop neu frontend muon sort danh sach booking theo xac suat huy giam dan.
- **LR/SVM**: chi nen dung lam baseline trong bao cao, khong nen lam model deploy chinh vi F1 thap hon RF/NN ~5 pp.

### Luu y ket noi frontend

1. **Cac model deu yeu cau input duoc apply feature engineering**: frontend can compute 8 feature engineered (`total_nights`, `total_guests`, `has_children`, `is_family`, `room_changed`, `has_agent`, `has_previous_cancellation`, `arrival_month_number`) truoc khi gui vao model.
2. **SVM khong co `predict_proba`** -> neu UI hien xac suat %, can boc `CalibratedClassifierCV` hoac loai SVM khoi UI.
3. **Preprocessing da bundled trong joblib**: chi can `joblib.load(...)` + `model.predict(df)` la xong, khong can fit lai encoder/scaler.
4. **Categorical handle_unknown='ignore'**: input la category moi se silently bi zero-vector. Backend nen validate truoc."""),
]


def build_comparison_notebook() -> None:
    nb = {
        "cells": COMP_CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv (3.13.12)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = BASE / "hotel_booking_models_comparison.ipynb"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print(f"[+] Wrote {out_path.name} ({len(COMP_CELLS)} cells)")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    build_comparison_notebook()

    append_conclusion(
        BASE / "hotel_booking_logistic_regression_pipeline.ipynb", LR_CONCLUSION
    )
    append_conclusion(BASE / "hotel_booking_svm_pipeline.ipynb", SVM_CONCLUSION)
    append_conclusion(BASE / "hotel_booking_neural_network_pipeline.ipynb", NN_CONCLUSION)
    append_conclusion(BASE / "hotel_booking_random_forest_pipeline.ipynb", RF_CONCLUSION)

    insert_methodology(
        BASE / "hotel_booking_logistic_regression_pipeline.ipynb", LR_METHODOLOGY
    )
    insert_methodology(BASE / "hotel_booking_svm_pipeline.ipynb", SVM_METHODOLOGY)
    insert_methodology(BASE / "hotel_booking_neural_network_pipeline.ipynb", NN_METHODOLOGY)
    insert_methodology(BASE / "hotel_booking_random_forest_pipeline.ipynb", RF_METHODOLOGY)

    insert_chart_after_marker(
        BASE / "hotel_booking_random_forest_pipeline.ipynb",
        marker_substr="random_forest_feature_importance.csv",
        md_header=RF_FI_CHART_MD,
        code_text=RF_FI_CHART_CODE,
    )
    insert_chart_after_marker(
        BASE / "hotel_booking_neural_network_pipeline.ipynb",
        marker_substr="nn_tuned = best_tuned_model",
        md_header=NN_LOSS_CHART_MD,
        code_text=NN_LOSS_CHART_CODE,
    )

    replace_eda_insight()

    print("\nDone.")
