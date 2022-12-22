### Eko-Housing: Apartment Estimates Lagos Nigeria.

![ekoh](https://user-images.githubusercontent.com/59312765/208224371-59073ffd-a4dc-4df2-99d5-145e315bf350.jpg)

#### About: 
##### Eko Housing is a simple web app that generates housing estimates based on your desired amenities and location(District).


#### creating the model (Eko's life force if you will)

#### 1. Import Dependencies.

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
df =pd.read_csv('gome1.csv')
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>housing_type</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>guest_toilet</th>
      <th>parking_space</th>
      <th>district</th>
      <th>address</th>
      <th>rent_per_annum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
      <td>Ikate, Lekki, Lagos</td>
      <td>5000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
      <td>Ikate Elegushi, Lekki, Lagos</td>
      <td>3000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
      <td>Oniru Estate, Oniru, Victoria Island (VI), Lagos</td>
      <td>4500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
      <td>Ikate, Lekki, Lagos</td>
      <td>4000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>36</td>
      <td>By Pinnacle Filling Station Marwa, Lekki, Lagos</td>
      <td>3500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19325 entries, 0 to 19324
    Data columns (total 8 columns):
     #   Column          Non-Null Count  Dtype 
    ---  ------          --------------  ----- 
     0   housing_type    19325 non-null  int64 
     1   bedrooms        19325 non-null  int64 
     2   bathrooms       19325 non-null  int64 
     3   guest_toilet    19325 non-null  int64 
     4   parking_space   19325 non-null  int64 
     5   district        19325 non-null  int64 
     6   address         19325 non-null  object
     7   rent_per_annum  19325 non-null  int64 
    dtypes: int64(7), object(1)
    memory usage: 1.2+ MB
    


```python
x= df.drop(['address','rent_per_annum'], axis = 1)
y= df.rent_per_annum
x.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>housing_type</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>guest_toilet</th>
      <th>parking_space</th>
      <th>district</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>

##### Training The model using XGBoosr Regressor.


```python
x_train, x_test, y_train, y_test = train_test_split(x.values,y.values,test_size = 0.3, random_state = 2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
```




    ((13527, 6), (13527,), (5798, 6), (5798,))




```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(x_train, y_train)
```



<pre>XGBRegressor(base_score=0.5, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy=&#x27;depthwise&#x27;, importance_type=None,
             interaction_constraints=&#x27;&#x27;, learning_rate=0.300000012, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
             max_depth=6, max_leaves=0, min_child_weight=1, missing=nan,
             monotone_constraints=&#x27;()&#x27;, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0, ...)</pre>

#### Metric Score for training data.

```python
model_train = model.predict(x_train)
error_scoree =metrics.r2_score(y_train, model_train)
print('error score is :', error_scoree)
```

    error score is : 0.8506554402342814
    
#### Metric Score for test data.

```python
model_train1 = model.predict(x_test)
error_scoree1 =metrics.r2_score(y_test, model_train1)
print('error score is :', error_scoree1)
```

    error score is : 0.8255315388877779
    
#### 82% .. Not bad. Now Lets see if the model actually works.



#### Testing the Model.

```python
inputed_data = (1,2,2,0,2,36)
inputed_as_np = np.asarray(inputed_data)
inputed_reshaped = inputed_as_np.reshape(1,-1)

prediction = model.predict(inputed_reshaped)

print(prediction)
```

    [3315030.5]
    
### ITS ALIIIIIIIIIIVVVVVEE!!!!!!. 
