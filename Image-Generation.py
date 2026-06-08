import boto3
import json
import base64
import os
import time
import random
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
bedrock_claude = boto3.client(service_name="bedrock-runtime")

# ── Style anchor ────────────────────────────────────────────────────────────
# This block is injected into EVERY scene prompt.
# Changing it here changes the look of the whole video consistently.
STYLE_ANCHOR = (
    "Cinematic digital illustration, 16:9 aspect ratio. "
    "Consistent art style: painterly, warm color grading, soft volumetric lighting. "
    "Character designs must remain identical across all scenes. "
    "No text overlays, no watermarks."
)


def rewrite_prompt_with_claude(original_prompt):
    instruction = f"""
    Rewrite the following image generation prompt to be safe, non-violent,
    and compliant with AI content policies. Keep the meaning intact.
    Return only the rewritten prompt, under 500 characters.

    Prompt:
    {original_prompt}
    """
    response = bedrock_claude.invoke_model(
        modelId="global.anthropic.claude-sonnet-4-20250514-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": instruction}]
        })
    )
    return json.loads(response["body"].read())["content"][0]["text"]


def build_character_description_prompt(character):
    """
    Build a neutral reference-sheet prompt for a single character.
    Plain background + front-facing pose ensures the model captures
    the character cleanly without scene clutter contaminating it.
    """
    return (
        f"{STYLE_ANCHOR} "
        f"Character reference sheet. Full body portrait, neutral front-facing pose, "
        f"plain white background, soft studio lighting. "
        f"Character: {character['name']}. {character['description']}. "
        f"Highly detailed face and costume. No other characters in frame."
    )


def build_scene_prompt(scene):
    return (
        f"{STYLE_ANCHOR} "
        f"Scene: {scene['description']}. "
        f"Narration: {scene['narration']}. "
        f"Visual: {scene['visual_prompt']}. "
        f"Lighting: {scene['lighting']}. "
        f"Camera: {scene['camera_angle']}. "
        f"Mood: {scene['mood']}."
    )


def generate_image_with_retry(prompt, reference_image_b64=None, strength=0.25):
    """
    Generate an image with retry + content-filter rewrite logic.
    - reference_image_b64: base64 PNG/JPEG string (no data URI prefix)
    - strength: how much to deviate from the reference (0 = clone, 1 = ignore it)
    """
    max_retries = 10
    current_prompt = prompt

    for attempt in range(max_retries):
        if reference_image_b64:
            payload = {
                "prompt": current_prompt,
                "mode": "image-to-image",
                "image": reference_image_b64,
                "strength": strength,
                "output_format": "png"
            }
        else:
            payload = {
                "prompt": current_prompt,
                "mode": "text-to-image",
                "aspect_ratio": "16:9",
            }

        try:
            response = bedrock.invoke_model(
                modelId="stability.sd3-5-large-v1:0",
                body=json.dumps(payload)
            )
            body = json.loads(response["body"].read())
            return base64.b64decode(body["images"][0])

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "ValidationException":
                print(f"⚠️  Content filter on attempt {attempt + 1}. Rewriting with Claude...")
                current_prompt = rewrite_prompt_with_claude(current_prompt)
                # After 3 strikes, drop the reference image — it may be the trigger
                if attempt >= 3 and reference_image_b64:
                    print("⚠️  Dropping reference image after repeated filter hits.")
                    reference_image_b64 = None

            elif error_code == "ThrottlingException":
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳  Throttled. Retrying in {wait:.1f}s...")
                time.sleep(wait)

            else:
                raise e

    raise RuntimeError(f"Image generation failed after {max_retries} attempts.")


def composite_reference_images(image_b64_list):
    """
    Merge multiple character reference images into one composite before
    passing to the scene generator. Simple horizontal concatenation via
    Pillow — keeps all characters 'known' to the model at once.

    Returns base64 PNG string, or None if PIL is unavailable.
    """
    try:
        from PIL import Image
        import io

        images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in image_b64_list]
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        composite = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img in images:
            composite.paste(img, (x_offset, 0))
            x_offset += img.width

        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except ImportError:
        print("⚠️  Pillow not installed — using first character reference only.")
        return image_b64_list[0] if image_b64_list else None


def process_video_generation():
    source_bucket = os.environ.get("SOURCE_BUCKET", "canyouimagine-video-assets")
    target_bucket = os.environ.get("TARGET_BUCKET", "canyouimagine-video-images")
    event = json.loads(os.getenv("EVENT_DATA", "{}"))

    s3_key = event.get("scene_output", {}).get("s3_key")
    if not s3_key:
        raise ValueError("No s3_key found in input event")

    response = s3.get_object(Bucket=source_bucket, Key=s3_key)
    print(type(response))
    print(response)
    print("------------------------")
    data = json.loads(response["Body"].read())
    request_id = data["request_id"]
    characters = {c["name"]: c for c in data["context"]["characters"]}

    # ── Phase 1: Generate one reference sheet image per character ────────────
    print("\n📸  Generating character reference sheets...")
    character_refs = {}  # name → base64 PNG string

    for char_name, char in characters.items():
        print(f"  → {char_name}")
        ref_prompt = build_character_description_prompt(char)
        image_bytes = generate_image_with_retry(ref_prompt)

        # Save reference sheet to S3 for debugging / reuse
        ref_key = f"outputs/{request_id}/refs/{char_name.replace(' ', '_')}.png"
        s3.put_object(Bucket=target_bucket, Key=ref_key,
                      Body=image_bytes, ContentType="image/png")

        character_refs[char_name] = base64.b64encode(image_bytes).decode("utf-8")
        print(f"  ✅  {char_name} reference saved → {ref_key}")

    # ── Phase 2: Generate scene images using character refs as anchors ────────
    print("\n🎬  Generating scenes...")

    previous_scene_reference = None

    for scene in data["scenes"]:
        scene_num = scene["scene_number"]
        scene_chars = scene.get("characters", [])
        print(f"\n  Scene {scene_num}: {scene['title']} — characters: {scene_chars}")

        # Build a composite of only the characters that appear in this scene.
        # This keeps the reference tight — don't inject irrelevant characters.
        refs_for_scene = [character_refs[c] for c in scene_chars if c in character_refs]

        if len(refs_for_scene) > 1:
            reference_image = composite_reference_images(refs_for_scene)
        elif len(refs_for_scene) == 1:
            reference_image = refs_for_scene[0]

        elif previous_scene_reference:
            print("  ↪ No characters found. Using previous scene as reference.")
            reference_image = previous_scene_reference

        else:
            reference_image = None  # No named characters in scene

        scene_prompt = build_scene_prompt(scene)

        # strength=0.35 lets the scene diverge enough from the reference-sheet
        # pose/background while still anchoring the character's appearance.
        image_bytes = generate_image_with_retry(
            prompt=scene_prompt,
            reference_image_b64=reference_image,
            strength=0.35
        )
        previous_scene_reference = base64.b64encode(image_bytes).decode("utf-8")

        image_key = f"outputs/{request_id}/scene_{scene_num}.png"
        s3.put_object(Bucket=target_bucket, Key=image_key,
                      Body=image_bytes, ContentType="image/png")

        scene["image_s3_path"] = f"s3://{target_bucket}/{image_key}"
        print(f"  ✅  Scene {scene_num} saved.")

    # ── Phase 3: Write final JSON ────────────────────────────────────────────
    final_key = f"outputs/{request_id}/final_scenes.json"
    s3.put_object(Bucket=target_bucket, Key=final_key,
                  Body=json.dumps(data, indent=2), ContentType="application/json")

    output = {
        "status": "Success",
        "final_json_path": f"s3://{target_bucket}/{final_key}",
        "s3_key": s3_key
    }
    print(json.dumps(output))


if __name__ == "__main__":
    process_video_generation()