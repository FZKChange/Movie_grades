
import pickle
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import GridSearchCV

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import mean_squared_error


def get_sentiment_scores_chinese(file_path):
    df = pd.read_csv(file_path)  # 使用pandas读取csv文件
    comments = df["comments"].tolist()   # 获取comments列并转换成列表
    ratings = df["ratings"].tolist()   # 获取ratings列并转换成列表

    # 载入预训练的Bert模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    model.eval()   # 设置为评估模式

    scores = []
    with torch.no_grad():   # 禁用梯度计算
        for comment in comments:
            inputs = tokenizer.encode_plus(comment, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
            # 对评论进行tokenization并转换为PyTorch张量

            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            # 将输入传入模型进行推理

            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)[0]   # 计算情感分类得分

            # 将情感分类得分转换为情感分数
            score = float(probs[1].detach().cpu().numpy())   # 取第2个分类标签的概率作为情感分数
            scores.append(score)

    #     # 将三个列表转换为数据帧
    # df_result = pd.DataFrame({'comments': comments, 'ratings': ratings, 'scores': scores})
    # # 保存数据帧为CSV文件
    # df_result.to_csv('result.csv', index=False)

    return comments, ratings, scores




c,r,s = get_sentiment_scores_chinese('newdata.csv')


#
#
# def train_regression_model_svm(comments, ratings, scores):
#     X = []
#     for i in range(len(comments)):
#         X.append([scores[i],])
#     y = ratings
#
#     # 设置参数范围
#     param_grid = {'C': [0.8,  1, 5, 10, 100]}
#
#     # 创建SVR模型
#     svr = SVR(kernel='rbf', epsilon=0.2)
#
#     # 使用GridSearchCV进行交叉验证
#     model = GridSearchCV(svr, param_grid, cv=5)
#     model.fit(X, y)
#
#
#     return model.best_estimator_
#
# def load_regression_model(model_file):
#     # 加载保存的模型文件
#     with open(model_file, 'rb') as f:
#         model = pickle.load(f)
#
#     return model
#
#
#
# # def predict_rating(model, comment):
# #     # 使用Bert进行情感分析，得到情感分数
# #     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# #     model_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
# #     encoded_comment = tokenizer(comment, return_tensors='pt', padding=True, truncation=True)
# #     output = model_bert(**encoded_comment)
# #     sentiment_score = output.logits.detach().numpy()[0][1]
# #
# #     # 将情感分数作为输入，传入到SVR模型中进行预测
# #     predicted_rating = model.predict([[sentiment_score]])[0]
# #
# #     # 返回预测的打星数
# #     return predicted_rating
#
# def predict_rating(model, file_path):
#     # Load csv file
#     df = pd.read_csv(file_path)
#     comments = df["comments"].tolist()
#     ratings = df["ratings"].tolist()
#
#     # Use BERT for sentiment analysis and SVR for rating prediction
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     model_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
#     sentiment_scores = []
#     for comment in comments:
#         encoded_comment = tokenizer(comment, return_tensors='pt', padding=True, truncation=True)
#         output = model_bert(**encoded_comment)
#         score = output.logits.detach().numpy()[0][1]
#         sentiment_scores.append(score)
#     predicted_ratings = model.predict([[score] for score in sentiment_scores])
#
#     # Calculate mean squared error
#     mse = mean_squared_error(ratings, predicted_ratings)
#
#     # Output predicted ratings and mean squared error
#     print(f"Mean squared error: {mse:.2f}")
#     print("Predicted ratings:")
#     for i, rating in enumerate(predicted_ratings):
#         print(f"{i + 1}. {rating:.1f}")
#
#
# test=train_regression_model_svm(c,r,s) #得到已经训练好的模型
# print(test) #打印已经训练好的模型的参数
# predict_rating(test, 'new_test.csv')
