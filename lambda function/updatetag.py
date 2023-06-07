import json
import boto3

# 初始化 DynamoDB 和项目存在标识符
database = boto3.resource('dynamodb')
itemIsExisted = True

def lambda_handler(event, context):
    # 检索事件中所需的字段
    queryURL = event['url']
    type = event['type']
    tags = event['tags']
    
    # 初始化数据库，键将为用户的查询键
    table = database.Table('image_url_tag')
    
    # 根据用户的查询 URL 从 DynamoDB 中检索项目
    response = table.get_item(
        Key={
            's3-url': queryURL
        }
    )

    # 检查用户的查询键是否与数据库中的项目匹配
    # 如果找到匹配项，则将标识符设置为 True，并打印图像 URL
    # 否则，将标识符设置为 False
    try:
        print('Modify tags for:', response['Item']['s3-url'])
        itemIsExisted = True
    except:
        itemIsExisted = False

    # 如果项目存在，则根据类型执行标签的添加或删除操作
    if itemIsExisted:
        databaseItemTags = response['Item']['tags']
        
        if type == 1:  # 添加标签
            for tag in tags:
                databaseItemTags.append(tag)
        elif type == 0:  # 删除标签
            for tag in tags:
                if tag in databaseItemTags:
                    databaseItemTags.remove(tag)
        
        # 使用更新后的标签列表更新 DynamoDB 中的项目
        table.update_item(
            Key={
                's3-url': queryURL
            },
            UpdateExpression='SET tags = :val1',
            ExpressionAttributeValues={
                ':val1': databaseItemTags
            }
        )
        
        print('Updated:', response['Item'])
        return 'Tags have been modified for the image'
    else:
        print("No such image in the database.")
        return "No matching image found in the database."
