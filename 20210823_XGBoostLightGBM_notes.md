# XGBoost, LightGBM study notes

## References

  * <https://statkclee.github.io/model/model-python-xgboost-hyper.html> (XGBoost paramter tuning basics)

  * <https://nurilee.com/2020/04/03/lightgbm-definition-parameter-tuning/> (LightGBM paramter tuning basics)

  * <https://neptune.ai/blog/lightgbm-parameters-guide?utm_source=datacamp&utm_medium=post&utm_campaign=blog-lightgbm-parameters-guide&utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com> (LightGBM parameters)
## XGBoost

## LightGBM

  * Light GBM은 Tree가 수직적으로 확장되는 반면에 다른 알고리즘은 Tree가 수평적으로 확장
  
  * 즉, Light GBM은 leaf-wise 인 반면 다른 알고리즘은 level-wise 
  
  * 확장하기 위해서 max delta loss를 가진 leaf를 선택하게 됨
  
  * 동일한 leaf를 확장할 때, leaf-wise 알고리즘은 level-wise 알고리즘보다 더 많은 loss, 손실을 줄일 수 있음

### Parameters

  * boosting: defining algorithm type (default: gbdt)
    
    - gdbt : Traditional Gradient Boosting Decision Tree
    - rf : Random Forest
    - dart : Dropouts meet Multiple Additive Regression Trees
    - goss : Gradient-based One-Side Sampling

 * objective: defining model to apply (defauly: regression)

    - regression: 회귀분석
    - binary: 이진 분류
    - multiclass: 다중 분류