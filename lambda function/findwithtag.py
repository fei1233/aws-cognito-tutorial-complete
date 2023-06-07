import json
import boto3

# 初始化数据库和 imageURL 列表
database = boto3.resource('dynamodb')

def lambda_handler(event, context):
    imageURL = []
    # 接收用户查询的 JSON
    queryTags = event['tags'] 
    table = database.Table('image_url_tag')

    response = table.scan()
    dbdata = response['Items']

    # 遍历数据库中的数据项
    for index, element in enumerate(dbdata):
        url = element['s3-url']
        tags = element['tags']
        imageInDatabase = set(tags)
        
        queryFromUser = set(queryTags)
        
        # 如果用户查询的标签与图像的标签匹配，则将该图像的 URL 添加到列表中
        # 如果用户查询的标签为空列表，则返回数据库中的所有 URL
        if len(queryTags) == 0:
            imageURL.append(url)
        else:
            if queryFromUser == imageInDatabase:
                imageURL.append(url)

    result = {"links": imageURL}
    jsonResult = json.dumps(result)
    
    return jsonResult
