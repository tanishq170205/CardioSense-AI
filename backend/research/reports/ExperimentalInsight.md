# Experimental Insight Report

## Best Overall Model
- **Model**: CatBoost
- **Task**: binary
- **Feature Set**: Fuzzy
- **AUC**: 0.8288
- **F1 Score**: 0.7670

## Does Fuzzification Help?
Average AUC across all models by Feature Set:
- **Raw**: 0.0000
- **Fuzzy**: 0.7574
- **Hybrid**: 0.0000

*Analysis*: 
Based on empirical evaluation, using Fuzzy features provided the highest average performance. 
