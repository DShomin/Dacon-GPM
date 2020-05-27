# Dacon 위성관측 데이터 활용 강수량 산출 AI 경진대회

최종 순위 : 8

## Model : custom model (from GoldBar)
GoldBar님의 모델에서 attention layer를 추가하였습니다.

## Data Augmentation
H_flip & V_flip

## Loss 
### custom loss
kl_divergence(sigmoid(label), sigmoid(model_pred)) + mae(label, model_pred)

## optimizer 
Adam

## Ensemble
### K-Fold Ensemble
5 fold
### TTA 
H_flip & V_flip

Meaning of all result values

## Post-process
result가 0.013이하 인것들을 모두 0.09999로 바꿔서 submit
