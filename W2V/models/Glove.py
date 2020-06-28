# -*- coding: utf-8 -*-
class RiskEvent:

    def __init__(self):
        self.uid = "20200623_12345"           # 唯一编号
        self.title = "北京外卖小哥确诊新冠"     # 标题
        self.kind = "资讯"                     # *事件大类：资讯(默认)、金融交易、司法、工商、税务……
        self.catalog = "疾病"                  # *事件分类：（暂空）、(政策、社会、较难梳理)
        self.serial = "北京新冠肺炎复发"        # *事件序列，用于将单个事件串成序列，前期置空
        self.abstract = "饿了么外卖小哥确诊新冠，活动轨迹主要在方庄、成寿寺、肖村桥"   # 文章摘要，或事件描述
        self.occur_time = "2020-06-23 16:47"   # 发生时间
        self.opinion_level = 0                 # 情绪分级：0~6
        self.risk_score = 85                   # 风险评分：1~100
        self.risk_level = 5                    # 风险等级：1~5

    @property
    def detail_data(self):
        # （暂空）事件关联详细数据，用于评分评级，用子表存储
        return {"隔离人数": 100, "患病等级": "轻微"}

    @property
    def involve_targets(self):
        # 直接涉及标的，用子表存储：代码、标的、类型、关联程度分级(0~1)
        return [("elm", "饿了么", "公司", 1.0), ("123456", "张三", "人物", 1.0),
                ("", "外卖", "概念", 0.8), ("", "餐饮", "概念", 0.4)]

    @property
    def link_docs(self):
        # 关联资讯，存储资讯聚类结果
        return []