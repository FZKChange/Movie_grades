
import csv
import jieba
from snownlp import SnowNLP
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



# 定义情感分析函数
def get_sentiment_scores_chinese(file_path):
    df = pd.read_csv(file_path)  # 使用pandas读取csv文件
    comments = df["comments"].tolist()   # 获取comments列并转换成列表
    ratings = df["ratings"].tolist()   # 获取ratings列并转换成列表
    scores = []
    # 遍历每一个评论
    for comment in comments:
        # 使用jieba对评论进行分词
        words = jieba.lcut(comment)
        score = 0

        # 对每个分词进行情感分析
        for word in words:
            s = SnowNLP(word)
            # 将情感分析得分累加到总分数中
            score += s.sentiments

        score /= len(words)    # 计算平均情感得分

        scores.append(score)   # 将得分添加到列表scores中
    # 将三个列表转换为数据帧
    df_result = pd.DataFrame({'comments': comments, 'ratings': ratings, 'scores': scores})
    # 保存数据帧为CSV文件
    df_result.to_csv('result3.csv', index=False)
    return comments, ratings, scores

c,r,s = get_sentiment_scores_chinese('newdata.csv')

#
#
#

def train_regression_model_svm(comments, ratings, scores):
    X = []
    for i in range(len(comments)):
        X.append([scores[i],])
    y = ratings

    # 设置参数范围
    param_grid = {'C': [0.8,  1, 5, 10, 100]}

    # 创建SVR模型
    svr = SVR(kernel='rbf', epsilon=0.2)

    # 使用GridSearchCV进行交叉验证
    model = GridSearchCV(svr, param_grid, cv=5)
    model.fit(X, y)

    return model.best_estimator_
#

def predict_rating(model, file_path):
    # 读取csv文件
    df = pd.read_csv(file_path)
    comments = df["comments"].tolist()
    ratings = df["ratings"].tolist()

    # 预测评论的评分并计算误差
    sentiment_scores = [SnowNLP(comment).sentiments for comment in comments]
    predicted_ratings = model.predict([[score] for score in sentiment_scores])
    mse = mean_squared_error(ratings, predicted_ratings)

    # 输出误差和预测结果
    print(f"Mean squared error: {mse:.2f}")
    print("Predicted ratings:")
    for i, rating in enumerate(predicted_ratings):
        print(f"{i + 1}. {rating:.1f}")



test=train_regression_model_svm(c,r,s) #得到已经训练好的模型
print(test) #打印已经训练好的模型的参数
#
#
# #找一条新的评论做预测
# predict_rating(test, 'new_test.csv')
