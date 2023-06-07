import json
import boto3

s3_client = boto3.client("s3")
s3 = boto3.resource("s3")
database = boto3.resource('dynamodb')
itemIsExisted = True

def lambda_handler(event, context):
    userURL = event["s3-url"]


    # 查询 DynamoDB 查找图像
    table = database.Table('image_url_tag')
    # 根据用户的查询 URL 查找图像
    response = table.get_item(
        Key={
            's3-url': userURL
        }
    )

 
    try:
        print('This image exists in the DynamoDB', response['Item']['s3-url'])
        itemIsExisted = True
    except:
        print('This image does not exist in the DynamoDB')
        itemIsExisted = False

    # 调用 delete_item 方法从 DynamoDB 中删除此项
    if itemIsExisted:
        table.delete_item(
            Key={
                's3-url': userURL
            }
        )
        print("Item is deleted")
    else:
        print("No item is deleted")


    # 遍历 S3 存储桶中的所有对象
    bucket = 'detect123'
    objects = s3_client.list_objects(Bucket=bucket)['Contents']

    for obj in objects:
        s3_key = obj['Key']
        imageInS3 = f"https://{bucket}.s3.amazonaws.com/{s3_key}"

        if userURL == imageInS3:
            s3.Object(bucket, s3_key).delete()
        else:
            print("User's query does not match this image.")
