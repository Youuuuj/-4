getwd()
setwd('C:/Users/You/Desktop/빅데이터/빅데이터 수업자료/R/사례연구/사례연구 4/사례연구4 데이터셋')


rm(list = ls())

install.packages('rpart')
install.packages('rpart.plot')
install.packages('neuralnet')
library(rpart)  # 의사결정트리 기법
library(rpart.plot)  # 의사결정 트리 시각화
library(e1071)  # 나이브베이즈 기법 
library(randomForest)  # 랜덤포레스트 
library(nnet)  # 인공신경망
library(caret)  
library(car)
library(ggplot2)  # 시각화를 위한 패키지
library(neuralnet)  # 인공신경망

# 1) 위스콘신
# 데이터 가져오기
wisc <- read.csv('wisc.csv', header = T)
wisc

head(wisc)
str(wisc)

View(wisc)
wisc <- wisc[,-1]

length(wisc)

# 샘플링
set.seed(2)  # set.seed를 하지않으면 매번 다른 결과값을 보임.
wi <- sample(1:nrow(wisc), nrow(wisc) * 0.7)
train_wi <- wisc[wi,]
test_wi <- wisc[-wi,]



# 1-1) 의사결정트리 기법  - 정규화 필요X
rpart_wi <- rpart(diagnosis ~ ., data = train_wi)
rpart_wi
rpart.plot(rpart_wi)

# 예측 범주값 벡터 생성
pred_rpart_wi <- predict(rpart_wi, newdata = test_wi, type = 'class')

# 의사결정 트리 적용 분류 결과 도출
table(test_wi$diagnosis, pred_rpart_wi)

str(pred_rpart_wi)
test_wi$ diagnosis<- as.factor(test_wi$diagnosis)  

table(test_wi$diagnosis, pred_rpart_wi)

# 모델 성능 평가 지표
confusionMatrix(pred_rpart_wi, test_wi$diagnosis, positive = 'B')
# confusionMatrix함수를 사용하려면 안의 요소가 factor여야함


# 해석 
# 의사결정 트리기법을 적용 시 정확도(101+55)/(101+8+7+55)는 0.9123이다. 
# F-Measure 수치는 (2 * 0.9352 * 0.9266) / (0.9352 + 0.9266) = 0.9309 이다.



# 1-2) 나이브베이즈 머신러닝 기법
bayes_wi <- naiveBayes(diagnosis ~ ., data = train_wi)
bayes_wi

# 예측 범주값 벡터 생성
pred_bayes_wi <- predict(bayes_wi, newdata = test_wi, type = 'class')

# 나이브베이즈 적용 분류 결과 도출 
table(test_wi$diagnosis, pred_bayes_wi)

# 모델 성능 평가 지표
confusionMatrix(pred_bayes_wi, test_wi$diagnosis)

# 해석 
# 나이브베이즈 머신러닝 기법을 적용 시 정확도(101+60)/(101+3+7+60)는 0.9415이다.
# F-Measure 수치는 (2 * 0.9352 * 0.9712) / (0.9352 + 0.9712) = 0.9527 이다.



# 1-3) 랜덤 포레스트
random_wi <- randomForest(as.factor(diagnosis) ~ ., train_wi, ntree = 500)

# 예측 범주값 벡터 생성
pred_random_wi <- predict(random_wi, test_wi, type = 'response')

# 랜덤 포레스트 적용 분류 결과 도출
table(pred_random_wi, test_wi$diagnosis)

# 모델 성능 평가 지표
confusionMatrix(pred_random_wi,as.factor(test_wi$diagnosis))

# 해석 : 랜덤 포레스트 적용시 정확도(104+61)/(104+2+4+61)는 0.9649이다
# F-Measure 수치는 (2 * 0.9630 * 0.9811) / (0.9630 + 0.9811) = 0.9720 이다.


# 결론
# 정확도, F-measure으로 비교했을 때, 
# 의사결정트리기법 -> 나이브 베이즈 -> 랜덤 포레스트 기법 순으로 성능이 우수하다







# 2) 전복 나이
Abalone <-read.csv('abalone.csv')
str(Abalone)
head(Abalone)
summary(Abalone)

# 컬럼네임
colnames(Abalone) = c('Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings')
str(Abalone)
head(Abalone)

# 샘플링
set.seed(1)
ab <- sample(1:nrow(Abalone), nrow(Abalone) * 0.7)
train_ab <- Abalone[ab,]
test_ab <- Abalone[-ab,]
train_ab2 <- Abalone[ab,]
test_ab2 <- Abalone[-ab,]

nrow(test_ab)
nrow(train_ab)



# 2-1) 랜덤포레스트 
rf_ab <- randomForest(Rings ~ ., data = train_ab , mtry = 3)
rf_ab

# 시각화
plot(rf_ab)

# 중요변수확인, 시각화
importance(rf_ab)
varImpPlot(rf_ab)

# 예측
pre_rf_ab <- predict(rf_ab, newdata = test_ab)

# 시각화를 위한 처리
df_rf_ab <- data.frame(pre_rf_ab, test_ab$Rings)
head(df_rf_ab)

# 랜포 시각화
ggplot(df_rf_ab, aes(x=pre_rf_ab, y=test_ab.Rings)) + geom_point() + geom_smooth(method = 'auto', se = F)  # 시각 자료
cor(pre_rf_ab, test_ab$Rings) # 상관관계 분석
RMSE(pre_rf_ab, test_ab$Rings)
mean((pre_rf_ab - test_ab$Rings)^2)  # MES평균제곱오차 낮을수록 좋다




# 2-2) 다중회귀분석
# 귀무가설 : rings에 다른 변수들은 영향을 미치지않는다
# 대립가설 : rings에 다른 변수들은 영향을 미친다.

# 회귀분석 
lm_Ab <- lm(Rings ~ ., data = train_ab)
summary(lm_Ab)

# 변수선택법을 통한 다중회귀분석, 다중 공선성 확인
lm_Ab2 <- step(lm_Ab, method = 'both')
summary(lm_Ab2)
vif(lm_Ab2)

lm_Ab3 <- lm(Rings ~ Sex + Diam + Height + Shucked + Viscera + Shell, data = train_ab)
lm_Ab3
summary(lm_Ab3)
vif(lm_Ab3)

lm_Ab4 <- step(lm_Ab3, method = 'both')
summary(lm_Ab4)
vif(lm_Ab4)

plot(lm_Ab4, which = 1:6)


# 예측
pre_lm_ab <- predict(lm_Ab4, newdata = test_ab)


# 시각화를 위한 처리
df_lm_ab <- data.frame(pre_lm_ab, test_ab$Rings)
head(df_lm_ab)

# 회귀 시각화
ggplot(df_lm_ab, aes(x=pre_lm_ab, y=test_ab.Rings)) + geom_point() + geom_smooth(method = 'auto', se = F)  # 시각화 자료
cor(pre_lm_ab, test_ab$Rings)  # 상관관계 분석
RMSE(pre_lm_ab, test_ab$Rings)
mean((pre_lm_ab - test_ab$Rings)^2)  # MES평균제곱오차 낮을수록 좋다



# 2-3) 인공신경망

# 문자를 숫자형으로 변환
train_ab2$Sex[train_ab2$Sex == 'F'] <- '1'
train_ab2$Sex[train_ab2$Sex == 'M'] <- '2'
train_ab2$Sex[train_ab2$Sex == 'I'] <- '3'

test_ab2$Sex[test_ab2$Sex == 'F'] <- '1'
test_ab2$Sex[test_ab2$Sex == 'M'] <- '2'
test_ab2$Sex[test_ab2$Sex == 'I'] <- '3'

train_ab2$Sex <- as.numeric(train_ab2$Sex)
test_ab2$Sex <- as.numeric(test_ab2$Sex)
str(train_ab2)
str(test_ab2)

# 정규화
normal <- function(x){
  return((x - min(x))/(max(x) - min(x)))
}

train_ab_nor <- as.data.frame(sapply(train_ab2, normal))
test_ab_nor <- as.data.frame(sapply(test_ab2, normal))

str(train_ab_nor)
str(test_ab_nor)



#인공신경망 모델 생성
neur_ab <- neuralnet(Rings ~ ., data = train_ab_nor, hidden = 5, stepmax = 1e+05)
# 인공신경망 시각화
plot(neur_ab)
#예측 결과 생성
neur_ab_result <- compute(neur_ab, test_ab_nor[1:8])
neur_ab_result$net.result
pre_neur_ab <- predict(neur_ab, test_ab_nor)
str(pred_neur_ab)

# 시각화를 위한 처리
df_neur_ab <- data.frame(pre_neur_ab, test_ab_nor$Rings)
head(df_neur_ab)

# 시각화 
ggplot(df_neur_ab, aes(x=pre_neur_ab, y=test_ab_nor.Rings)) + geom_point() + geom_smooth(method = 'auto', se = F)
cor(pre_neur_ab, test_ab_nor$Rings)
RMSE(pre_neur_ab, test_ab_nor$Rings)
mean((pre_neur_ab - test_ab_nor$Rings)^2)





# 3) K-means시각화
data(iris)
# Species 컬럼 제거
iris2 <- iris[1:4] 
set.seed(3)

# 제곱합의 그래프
wssplot <- function(iris2, nc=15, seed=3){
  wss <- (nrow(iris2)-1)*sum(apply(iris2,2,var))
  for (i in 2:nc){
    set.seed(3)
    wss[i] <- sum(kmeans(iris2, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")} 

# 제곱합의 그래프 시각화
wssplot(iris2)

# 군집수의 결정을 위한 패키지
install.packages('NbClust')
library(NbClust)

nc_iris2 <- NbClust(iris2, min.nc = 2, max.nc = 15, method = 'kmeans')
table(nc_iris2$Best.nc[1,])

par(mfrow = c(1,1))
barplot(table(nc_iris2$Best.nc[1,]),
        xlab="Numer of Clusters", ylab="Number of Criteria")


# Kmeans 알고리즘 클러스터링 2개 생성
kmeans_result <- kmeans(iris2, 2, nstart = 25)
kmeans_result
kmeans_result$centers
kmeans_result$size


# 실측값과 클러스터링 값 비교
table(iris$Species, kmeans_result$cluster)



# 시각화
plot(iris2[c('Sepal.Length', 'Sepal.Width')], col = kmeans_result$cluster, pch = 15)
points(kmeans_result$centers[,c("Sepal.Length","Sepal.Width")], col = 1:3, pch = 8, cex = 4)

