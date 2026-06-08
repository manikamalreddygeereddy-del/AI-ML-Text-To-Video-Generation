import boto3
import json
import base64
import os
import time
import random
from botocore.exceptions import ClientError

# Initialize clients
s3 = boto3.client("s3")
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
bedrock_claude = boto3.client(service_name="bedrock-runtime")

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

    response = bedrock_claude.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": rewrite_instruction}]
        })
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def build_request_payload(prompt, reference_image_b64=None):
    """
    Build the request payload.
    If a reference image is provided, use image-to-image mode with low strength
    so the new scene visually inherits style/characters from the previous one.
    """
    if reference_image_b64:
        return {
            "prompt": prompt,
            "mode": "image-to-image",
            "image": reference_image_b64,
            "strength": 0.5,       # Low strength = strong reference to previous scene
            "output_format": "png",
            "aspect_ratio": "16:9"
        }
    else:
        return {
            "prompt": prompt,
            "mode": "text-to-image",
            "aspect_ratio": "16:9"
        }


def generate_image_with_retry(bedrock_client, prompt, reference_image_b64=None):
    """
    Generate an image with retry logic for throttling and content filter errors.
    Returns raw image bytes on success, raises on permanent failure.
    """
    max_retries = 10
    current_prompt = prompt

    for attempt in range(max_retries):
        request_payload = build_request_payload(current_prompt, reference_image_b64)

        try:
            response = bedrock_client.invoke_model(
                modelId="stability.sd3-5-large-v1:0",
                body=json.dumps(request_payload)
            )
            response_body = json.loads(response["body"].read())
            image_bytes = base64.b64decode(response_body["images"][0])
            return image_bytes

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "ValidationException":
                print(f"⚠️ Content filter triggered on attempt {attempt + 1}. Rewriting prompt with Claude...")
                current_prompt = rewrite_prompt_with_claude(current_prompt)
                # If image-to-image keeps failing after rewrite, fall back to text-to-image
                # on the next attempt to rule out the reference image as the cause
                if attempt >= 3 and reference_image_b64:
                    print("⚠️ Falling back to text-to-image after repeated content filter hits.")
                    reference_image_b64 = None

            elif error_code == "ThrottlingException":
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Throttled on attempt {attempt + 1}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)

            else:
                print(f"❌ Permanent error: {error_code} — {e.response['Error']['Message']}")
                raise e

    raise RuntimeError(f"Failed to generate image after {max_retries} attempts.")


def process_video_generation():
    # 1. Environment Configuration
    source_bucket = os.environ.get("SOURCE_BUCKET", "canyouimagine-video-assets")
    target_bucket = os.environ.get("TARGET_BUCKET", "canyouimagine-video-images")
    event_str = os.getenv("EVENT_DATA", "{}")
    event = json.loads(event_str)

    s3_key = event.get("scene_output", {}).get("s3_key")
    if not s3_key:
        raise ValueError("No s3_key found in input event")

    # 2. Load the original JSON from S3
    print(f"Fetching input from: {s3_key}")
    response = s3.get_object(Bucket=source_bucket, Key=s3_key)
    data = json.loads(response['Body'].read())
    request_id = data["request_id"]

    # 3. Iterate through scenes — carry the previous scene's image as a reference
    previous_scene_image_b64 = None  # None for scene 1 (pure text-to-image)

    for scene in data["scenes"]:
        scene_num = scene["scene_number"]
        print(f"\n🎬 Generating scene {scene_num}: {scene['title']}")

        consistent_prompt = f"""
            Cinematic digital illustration of
            Narration scene: {scene['narration']}
            Visual Prompt: {scene['visual_prompt']}
            Scene Description: {scene['description']}
            Lighting: {scene['lighting']}
            Camera angle: {scene['camera_angle']}
            Mood: {scene['mood']}
        """

        # Scene 1 → text-to-image. Scene 2+ → image-to-image using previous scene.
        image_bytes = generate_image_with_retry(
            bedrock_client=bedrock,
            prompt=consistent_prompt,
            reference_image_b64=previous_scene_image_b64
        )

        # 4. Save the generated image to S3
        image_filename = f"outputs/{request_id}/scene_{scene_num}.png"
        s3.put_object(
            Bucket=target_bucket,
            Key=image_filename,
            Body=image_bytes,
            ContentType='image/png'
        )
        scene["image_s3_path"] = f"s3://{target_bucket}/{image_filename}"
        print(f"✅ Scene {scene_num} saved → {image_filename}")

        # 5. Encode this scene's image as base64 to feed into the next scene
        previous_scene_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # 6. Save the final JSON back to S3
    final_json_key = f"outputs/{request_id}/final_scenes.json"
    s3.put_object(
        Bucket=target_bucket,
        Key=final_json_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json'
    )

    output = {
        "status": "Success",
        "final_json_path": f"s3://{target_bucket}/{final_json_key}",
        "s3_key": s3_key
    }
    print(json.dumps(output))
    print(f"🚀 Task Finished. Final JSON: s3://{target_bucket}/{final_json_key}")


if __name__ == "__main__":
    process_video_generation()