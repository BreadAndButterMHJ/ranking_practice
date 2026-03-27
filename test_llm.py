import requests

"""
会话接口
根据指定格式会话历史，模型生产新一轮对话结果

接口url：{BASE_URL}/v1/chat/completions

请求方式： POST

接口入参：

字段名	类型	必填	描述
model	string	是	服务使用的模型ID
temperature	float	否	生成文本的随机性，取值范围[0,1]
messages	list<Message>	是	对话历史，拼接顺序[system(option)_msg,user_msg,ai_msg,user_msg]
max_tokens	int	否	控制生成的回复的最大令牌数。该参数用于限制生成的文本长度，以防止生成的回复过长。
stream	bool	否	是否流式调用：默认false
Message对象详情

字段名	类型	描述
role	string	角色标签
system： 全局标签
user: 用户标签
assistant：AI标签
content	string	角色对应的对话内容
接口出参：

名称	类型	描述
id	string	请求ID
object	string	响应类型枚举：
chat.completion：会话结果
chat.completion.chunk：流式会话结果
created	long	请求时间戳
model	string	请求的模型ID
choices	list<ChatChoice>	模型生成内容，流式和非流式返回格式稍有区别
usage	long	请求token数量，（非流式返回）
非流式ChatChoice对象详情：

字段名	类型	描述
index	string	返回数据索引
message	Message	模型返回的文本内容
finish_reason	string	结束原因
stop: 正常结束,
length：达到请求指定长度,
abort：用户放弃请求
流式ChatChoice对象详情：

字段名	类型	描述
index	string	返回数据索引
delta	Delta	模型实时返回的部分文本内容
finish_reason	string	结束原因
stop: 正常结束,
length：达到请求指定长度,
abort：用户放弃请求
Delta对象详情

字段名	类型	描述
content	string	流式文本内容
非流式测试命令

curl ${BASE_URL}/v1/chat/completions
--header 'Content-Type: application/json' \
--data '{
"model": "/mnt/models",
"messages": [
{
"role": "user",
"content": "中国的首都在哪里？并详细介绍下"
}
],
"max_tokens": 1000,
"temperature" : 0,
"stream" : false
}'
非流式返回数据

{
	"id": "6f0b925c982e4767adc6aa224192c1af",
	"object": "chat.completion",
	"created": 1756891284,
	"model": "Qwen1.5-14B-Chat",
	"choices": [
		{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "中国的首都是北京。北京，位于中国北部，是中华人民共和国的首都，也是全国的政治中心、文化中心、国际交往中心和科技创新中心。其历史悠久，有着3000多年的建城历史，是世界上最早被确认的古都之一，曾是多个朝代的都城，包括金、元、明、清。\n\n以下是对北京的一些详细介绍：\n\n1. 地理位置：北京地处华北平原的北部边缘，东南部与天津相邻，西部紧邻河北省，北部和西部与内蒙古自治区接壤。\n\n2. 城市规模：北京是中国的超大城市，面积16411平方公里，人口超过2100万。\n\n3. 世界文化遗产：北京拥有众多世界文化遗产，如故宫、天坛、颐和园、长城、圆明园等，是世界上拥有世界文化遗产最多的城市。\n\n4. 高端科技：北京是中关村科技园区的所在地，汇集了众多高科技企业和研究机构，是中国的科技创新中心。\n\n5. 交通网络：北京的交通网络发达，拥有高速公路、铁路、地铁、机场等各种交通方式，是中国的交通枢纽。\n\n6. 城市特色：北京的城市建筑融合了古今中外的风格，既有传统的四合院、胡同，又有现代化的高楼大厦。北京的美食也十分丰富，如烤鸭、炸酱面、豆汁焦圈等。\n\n总之，北京是中国的象征，集历史文化、政治、经济、科技于一体，具有深厚的底蕴和活力。",
				"reasoning_content": null,
				"tool_calls": null
			},
			"logprobs": null,
			"finish_reason": "stop",
			"matched_stop": 151645
		}
	],
	"usage": {
		"prompt_tokens": 27,
		"total_tokens": 341,
		"completion_tokens": 314,
		"prompt_tokens_details": null
	}
}
"""
# ============ 大模型请求配置 ============
TOKEN = "bd3bfad70e7a4a669837f5670d3947dd"
BASE_URL = "https://inference-jdaip-cn-north-1.jdcloud.com/queue-5632367a09b1be30e541eb96255f9b06/api/predict/qwen3-5-2b-mhj-v1/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}
data = {
    "model": "Qwen3.5-2B",
    "messages": [
        {"role": "user", "content": "介绍你自己"},
    ],
    "max_tokens": 1000,
    "temperature": 0.7,
    "stream": False,
}
response = requests.post(BASE_URL, headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])
