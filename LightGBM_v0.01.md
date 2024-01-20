[LightGBM](https://github.com/Microsoft/LightGBM)

Light Gradient Boosting Machine
- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel and GPU learning.
- Capable of handling large-scale data.

새 트리 학습 시 모델 출력 결과와 정답 데이터의 차이를 종속 변수로 활용
차례로 결정 트리 학습 결과 추가로 정확도 개선

[Baseline](https://www.kaggle.com/code/tunguz/lightgbm-baseline)

```python
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/Kannada-MNIST/train.csv')
col = ['pixel%d'%i for i in range(784)]

lgb_params = {
    "objective" : "multiclass", # cross_entropy, regression_l2
    "metric" : "multi_logloss", # AUC, regression_l2
    "num_class" : 10,
    "max_depth" : 12,
    "num_leaves" : 15,
    "learning_rate" : 0.05,
    "bagging_fraction" : 0.9,
    "feature_fraction" : 0.9,
    "lambda_l1" : 0.01,
    "lambda_l2" : 0.0,
}

X_train, X_test, Y_train, Y_test = train_test_split(df[col], df['label'], test_size=0.1)

lgtrain = lgb.Dataset(X_train, label=Y_train)
lgtest = lgb.Dataset(X_test, label=Y_test)
lgb_clf = lgb.train(lgb_params, lgtrain, 1500, 
                    valid_sets=[lgtrain, lgtest], 
                    early_stopping_rounds=10, 
                    verbose_eval=20)

df = pd.read_csv('../input/Kannada-MNIST/test.csv')
res = lgb_clf.predict( df[col] ).argmax(axis=1)

df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
df['label'] = res
df.to_csv('submission.csv', index=False)
```
