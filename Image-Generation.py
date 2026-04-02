import boto3
import json
import base64
import os
import time
import random
from botocore.exceptions import ClientError

# Initialize clients
s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")


def rewrite_prompt_with_claude(original_prompt):
    """Use Claude to sanitize the prompt for Titan Image Generator"""
    rewrite_instruction = f"""
    Rewrite the following image generation prompt to ensure it is completely safe,
    non-violent, non-sensitive, and compliant with AI content policies.
    Keep the meaning but remove or soften anything that could trigger filters.
    Only and only return the rewritten prompt.
    **Important** There should be only less than 500 characters in the prompt.

    Prompt:
    {original_prompt}
    """

    response = bedrock.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": rewrite_instruction}]
        })
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def process_video_generation():
    # 1. Environment Configuration (Set these in your ECS Task Definition)
    source_bucket = os.environ.get("SOURCE_BUCKET", "canyouimagine-video-assets")
    target_bucket = os.environ.get("TARGET_BUCKET", "canyouimagine-video-images")
    event_str = os.getenv("EVENT_DATA", "{}")
    event = json.loads(event_str)

    s3_key = event.get("scene_output", {}).get("s3_key")
    if not s3_key:
        raise ValueError("No s3_key found in input event")


    if not s3_key:
            print("Error: No S3_KEY environment variable provided.")
            return

    # 2. Load the original JSON from S3
    print(f"Fetching input from: {s3_key}")
    response = s3.get_object(Bucket=source_bucket, Key=s3_key)
    data = json.loads(response['Body'].read())
    request_id = data["request_id"]

    # 3. Iterate through scenes
    for scene in data["scenes"]:
        consistent_prompt = f"""
        Cinematic digital illustration of
        Narration scene: {scene['narration']}
        Lighting: {scene['lighting']}
        Camera angle: {scene['camera_angle']}
        Mood: {scene['mood']}
        """

        request_payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": consistent_prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 640,
                "width": 1152,
                "cfgScale": 10.0
            }
        }

        # Handle Generation with Retry Logic
        max_retries = 5
        image_response = None

        for attempt in range(max_retries):
            try:
                image_response = bedrock.invoke_model(
                    modelId="amazon.titan-image-generator-v2:0",
                    body=json.dumps(request_payload)
                )
                break

            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "ValidationException":
                    print("⚠️ Content filter triggered. Rewriting with Claude...")
                    safe_prompt = rewrite_prompt_with_claude(consistent_prompt)
                    request_payload["textToImageParams"]["text"] = safe_prompt
                    # Continues to next attempt loop with updated payload

                elif error_code == "ThrottlingException":
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Throttled. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)

                else:
                    print(f"❌ Permanent Error: {error_code}")
                    raise e

        # 4. Save the generated image
        if image_response:
            response_body = json.loads(image_response.get("body").read())
            image_bytes = base64.b64decode(response_body.get("images")[0])

            scene_num = scene["scene_number"]
            image_filename = f"outputs/{request_id}/scene_{scene_num}.png"

            s3.put_object(
                Bucket=target_bucket,
                Key=image_filename,
                Body=image_bytes,
                ContentType='image/png'
            )

            scene["image_s3_path"] = f"s3://{target_bucket}/{image_filename}"
            print(f"✅ Scene {scene_num} complete.")

    # 5. Save the FINAL JSON back to S3
    final_json_key = f"outputs/{request_id}/final_scenes.json"
    s3.put_object(
        Bucket=target_bucket,
        Key=final_json_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )

    # 6. Output for next Step Function state
    output = {
        "status": "Success",
        "final_json_path": f"s3://{target_bucket}/{final_json_key}",
        "s3_key": s3_key
    }
    print(json.dumps(output))
    print(f"🚀 Task Finished. Final JSON: s3://{target_bucket}/{final_json_key}")


if __name__ == "__main__":
    process_video_generation()
