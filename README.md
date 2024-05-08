
![figure1](https://github.com/FZKChange/Movie_grades/assets/78149508/2718fdb8-1b21-4f59-af49-06cf1fdd924c)

# Movie_grades
  一个有关电影评分回归的集合，包括情感分析模型和打分回归算法，一共六个模型代码，两个回归算法结合三个语言分析模型。
  newdata.csv是爬取豆瓣上电影的中文短评和对应打分，new_test.csv用的是张艺谋的《无极》，newdata.csv一共是39部电影，3765条评论，1星到5星近似均衡分布，需要提前下载好torch等对应库。
![figure2](https://github.com/FZKChange/Movie_grades/assets/78149508/b92f8068-4156-40f4-bf53-240f785b4b11)

  
# 爬虫数据的信息

![figue3](https://github.com/FZKChange/Movie_grades/assets/78149508/e90404a6-fb2e-469f-8f9b-cffe6838fc09)
原始爬取6648条0-5星的评论，删除只有评论没有打星的数据样本，均衡采样1-5星的样本，去除包含大量表情的文本样本。最后用3765条均衡数据用于后续的情感分析。

# 情感分析结果

![figure4](https://github.com/FZKChange/Movie_grades/assets/78149508/964a84e9-7217-4555-97fd-d4e3a84f8b18)
![figure5](https://github.com/FZKChange/Movie_grades/assets/78149508/0699c4be-f0f9-431a-a887-d4b54292abd1)
Bert模型需要有良好的数据集和充分的训练量作为下游任务的微调，才能取得好的性能，Snownlp模型基于购买场景的评论，并不完全适用电影评论数据的情感分析，而Snownlp plus在Snownlp的基础上引入了3百万条电影评论数据的训练。

# 建立情感分析结果和电影评分的回归联系
![figure6](https://github.com/FZKChange/Movie_grades/assets/78149508/ebd1e3d1-2e1e-4a55-bec0-4ac603262e8b)
![image](https://github.com/FZKChange/Movie_grades/assets/78149508/96fd9e11-4e67-4de0-a855-a22e3791b55c)

# 实验总结
![image](https://github.com/FZKChange/Movie_grades/assets/78149508/800e17c3-db0a-424e-96e3-032204a5e072)

