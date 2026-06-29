from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from pydantic import BaseModel, Field


class AvatarAppearanceInputs(BaseModel):
    ethnicity: str
    faceAndHair: str
    clothingAndAppearance: str
    ageRange: str
    background: str


class GenerateAvatarRequest(BaseModel):
    avatar_id: Optional[str] = None
    prompt: Optional[str] = None
    gender: Optional[str] = "person"
    appearance: Optional[AvatarAppearanceInputs] = None
    batch_size: int = 20
    bbox_shift: int = 0
    force_recreate: bool = True
    prepare: bool = True
    upload_video: bool = True
    motion_reference_video_url: Optional[str] = None
    motion_prompt: Optional[str] = None
    provider_order: List[str] = Field(
        default_factory=lambda: ["gpt_image_edit", "gpt_image", "getimg"]
    )


class AvatarGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "avatar_generation_failed",
        status_code: int = 500,
        detail: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.detail = detail or {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_size(value: str) -> Tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def _response_preview(response: requests.Response) -> str:
    try:
        return response.text[:2000]
    except Exception:
        return "<unreadable response body>"


def _public_s3_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def _upload_file_to_s3(path: Path, bucket: str, key: str, content_type: str) -> str:
    try:
        import boto3
    except Exception as exc:
        raise AvatarGenerationError(
            "boto3 is required for generated avatar S3 uploads",
            code="s3_dependency_missing",
            detail={"error": str(exc), "bucket": bucket, "key": key},
        ) from exc

    extra_args: Dict[str, str] = {"ContentType": content_type}
    acl = os.getenv("GENERATED_ASSET_ACL", "public-read").strip()
    if acl:
        extra_args["ACL"] = acl

    try:
        client = boto3.client(
            "s3",
            region_name=(
                os.getenv("AWS_DEFAULT_REGION")
                or os.getenv("AWS_REGION")
                or "us-east-1"
            ),
        )
        client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
    except Exception as exc:
        raise AvatarGenerationError(
            "Failed to upload generated avatar asset to S3",
            code="s3_upload_failed",
            detail={"error": str(exc), "bucket": bucket, "key": key, "path": str(path)},
        ) from exc

    return _public_s3_url(bucket, key)


def build_avatar_prompt(inputs: AvatarAppearanceInputs, gender: str = "person") -> str:
    base_prompt = (
        "Hyper-realistic, centered head-and-shoulders portrait of a "
        f"{inputs.ethnicity} {inputs.ageRange} year old {gender}. "
        f"{inputs.faceAndHair}. "
        f"{inputs.clothingAndAppearance}. "
        f"{inputs.background}."
    )
    professional_parameters = (
        "Vertical portrait composition for a 9:16 phone video source; "
        "match the stored pose reference for crop, camera distance, head position, shoulder placement, and direct eye-line only; "
        "centered head-and-shoulders portrait with ample headroom - ensure the crown of the head is fully visible and not cropped; "
        "face perfectly centered in the frame; "
        "crop should include shoulders and upper torso enough to align with the Kling Motion workflow reference video; "
        "perfectly symmetrical facial features; "
        "smooth natural skin with subtle, realistic pores and minimal makeup; "
        "neutral expression with relaxed facial muscles, lips gently closed, no smile, no frown, no grin, no teeth visible; "
        "studio-grade even softbox lighting; "
        "background should follow the requested setting while staying softly blurred, stable, and uncluttered; "
        "photographed on a full-frame DSLR (85 mm f/1.8) at f/2.0, razor-sharp focus on individual hair strands, eyelashes, and skin texture; "
        "lifelike catchlights in the eyes; "
        "8K resolution, no stylization or illustration feel."
    )
    return f"{base_prompt}\n\n{professional_parameters}"


def build_position_only_edit_prompt(prompt: str) -> str:
    return (
        "Use the reference image only for pose, composition, camera distance, "
        "head position, shoulder placement, eye line, and 9:16 crop. Do not "
        "copy the reference person's identity, face, hair, clothing, room, "
        "colors, lighting, or any other visual asset. Generate all character "
        "appearance, styling, clothing, and background strictly from this "
        f"prompt: {prompt}"
    )


def resolve_generation_prompt(request: GenerateAvatarRequest) -> str:
    if request.prompt and request.prompt.strip():
        return request.prompt.strip()
    if request.appearance is not None:
        return build_avatar_prompt(request.appearance, request.gender or "person")
    raise AvatarGenerationError(
        "Either prompt or appearance must be provided",
        code="avatar_prompt_missing",
        status_code=400,
    )


def normalize_image_to_jpeg(input_bytes: bytes, output_path: Path, size: str) -> Tuple[int, int]:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise AvatarGenerationError(
            "Pillow is required to normalize generated avatar images",
            code="pillow_dependency_missing",
            detail={"error": str(exc)},
        ) from exc

    width, height = _parse_size(size)
    try:
        image = Image.open(io.BytesIO(input_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG", quality=95, optimize=True)
    except Exception as exc:
        raise AvatarGenerationError(
            "Failed to normalize generated avatar image",
            code="image_normalization_failed",
            detail={"error": str(exc), "path": str(output_path)},
        ) from exc
    return width, height


def normalize_video(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(_env_int("GENERATED_AVATAR_VIDEO_SECONDS", 10)),
        "-vf",
        "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=30,setsar=1",
        "-an",
        "-c:v",
        os.getenv("GENERATED_AVATAR_VIDEO_ENCODER", "libx264"),
        "-preset",
        os.getenv("GENERATED_AVATAR_VIDEO_PRESET", "veryfast"),
        "-crf",
        os.getenv("GENERATED_AVATAR_VIDEO_CRF", "18"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AvatarGenerationError(
            "Failed to normalize generated avatar motion video",
            code="video_normalization_failed",
            detail={
                "cmd": cmd,
                "stdout": (exc.stdout or "")[-2000:],
                "stderr": (exc.stderr or "")[-4000:],
                "input_path": str(input_path),
                "output_path": str(output_path),
            },
        ) from exc


def _openai_models() -> List[str]:
    models = [os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()]
    models.extend(
        model.strip()
        for model in os.getenv(
            "OPENAI_IMAGE_FALLBACK_MODELS",
            "gpt-image-1.5,gpt-image-1,gpt-image-1-mini",
        ).split(",")
        if model.strip()
    )
    deduped = []
    for model in models:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _extract_image_bytes(data: Dict[str, Any], timeout: float) -> bytes:
    items = data.get("data")
    if not items:
        raise AvatarGenerationError(
            "OpenAI image response did not include data",
            code="openai_image_response_invalid",
            status_code=502,
            detail={"response": data},
        )
    first = items[0]
    b64_json = first.get("b64_json")
    if b64_json:
        return base64.b64decode(b64_json)
    url = first.get("url")
    if url:
        response = requests.get(url, timeout=timeout)
        if response.status_code >= 400:
            raise AvatarGenerationError(
                "Failed to download OpenAI image URL",
                code="openai_image_download_failed",
                status_code=502,
                detail={"status": response.status_code, "body": _response_preview(response), "url": url},
            )
        return response.content
    raise AvatarGenerationError(
        "OpenAI image response did not include b64_json or url",
        code="openai_image_response_invalid",
        status_code=502,
        detail={"response": data},
    )


def _call_openai_generation(prompt: str, model: str) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AvatarGenerationError(
            "OPENAI_API_KEY is required for GPT avatar image generation",
            code="openai_api_key_missing",
        )

    timeout = _env_float("AI_IMAGE_REQUEST_TIMEOUT", 180.0)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": os.getenv("OPENAI_IMAGE_SIZE", "720x1280"),
        "quality": os.getenv("OPENAI_IMAGE_QUALITY", "medium"),
        "output_format": os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "jpeg"),
    }
    moderation = os.getenv("OPENAI_IMAGE_MODERATION")
    if moderation:
        payload["moderation"] = moderation

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise AvatarGenerationError(
            "OpenAI image generation request failed",
            code="openai_image_generation_failed",
            status_code=502,
            detail={"status": response.status_code, "body": _response_preview(response), "model": model},
        )
    return _extract_image_bytes(response.json(), timeout)


def _call_openai_edit(prompt: str, model: str) -> bytes:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AvatarGenerationError(
            "OPENAI_API_KEY is required for GPT avatar image editing",
            code="openai_api_key_missing",
        )
    reference_url = os.getenv(
        "OPENAI_IMAGE_EDIT_REFERENCE_URL",
        "https://linguaprofilepics.s3.amazonaws.com/generated-images/reference/liveportrait_pose_reference_9x16.png",
    )
    timeout = _env_float("AI_IMAGE_REQUEST_TIMEOUT", 180.0)
    reference_response = requests.get(reference_url, timeout=timeout)
    if reference_response.status_code >= 400:
        raise AvatarGenerationError(
            "Failed to download OpenAI image edit pose reference",
            code="openai_reference_download_failed",
            status_code=502,
            detail={"status": reference_response.status_code, "body": _response_preview(reference_response), "url": reference_url},
        )

    data: Dict[str, Any] = {
        "model": model,
        "prompt": build_position_only_edit_prompt(prompt),
        "size": os.getenv("OPENAI_IMAGE_EDIT_SIZE", "720x1280"),
        "quality": os.getenv("OPENAI_IMAGE_QUALITY", "medium"),
        "output_format": os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "jpeg"),
    }
    moderation = os.getenv("OPENAI_IMAGE_MODERATION")
    if moderation:
        data["moderation"] = moderation

    files = {
        "image": (
            "pose_reference.png",
            reference_response.content,
            reference_response.headers.get("content-type", "image/png"),
        )
    }
    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {api_key}"},
        data=data,
        files=files,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise AvatarGenerationError(
            "OpenAI image edit request failed",
            code="openai_image_edit_failed",
            status_code=502,
            detail={"status": response.status_code, "body": _response_preview(response), "model": model},
        )
    return _extract_image_bytes(response.json(), timeout)


def _call_getimg(prompt: str) -> bytes:
    api_key = os.getenv("GETIMG_API_KEY")
    if not api_key:
        raise AvatarGenerationError(
            "GETIMG_API_KEY is not configured",
            code="getimg_api_key_missing",
        )

    response = requests.post(
        os.getenv("GETIMG_API_URL", "https://api.getimg.ai/v1/flux-schnell/text-to-image"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"prompt": prompt, "height": 512, "width": 512, "steps": 4},
        timeout=_env_float("GETIMG_REQUEST_TIMEOUT", 120.0),
    )
    if response.status_code >= 400:
        raise AvatarGenerationError(
            "GetImg avatar image request failed",
            code="getimg_request_failed",
            status_code=502,
            detail={"status": response.status_code, "body": _response_preview(response)},
        )

    data = response.json()
    b64_value = (
        data.get("image")
        or data.get("b64_json")
        or data.get("data", [{}])[0].get("b64_json")
    )
    if b64_value:
        return base64.b64decode(b64_value)
    url = data.get("url") or data.get("data", [{}])[0].get("url")
    if url:
        download = requests.get(url, timeout=_env_float("GETIMG_REQUEST_TIMEOUT", 120.0))
        download.raise_for_status()
        return download.content
    raise AvatarGenerationError(
        "GetImg response did not include image bytes or URL",
        code="getimg_response_invalid",
        status_code=502,
        detail={"response": data},
    )


def generate_still_image(prompt: str, output_path: Path, provider_order: List[str]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    size = os.getenv("OPENAI_IMAGE_SIZE", "720x1280")

    for provider in provider_order:
        provider = provider.strip()
        if provider == "gpt_image_edit":
            for model in _openai_models():
                try:
                    image_bytes = _call_openai_edit(prompt, model)
                    width, height = normalize_image_to_jpeg(image_bytes, output_path, size)
                    return {
                        "provider": "openai-gpt-image-edit",
                        "model": model,
                        "path": str(output_path),
                        "width": width,
                        "height": height,
                    }
                except Exception as exc:
                    errors.append({"provider": "openai-gpt-image-edit", "model": model, "error": str(exc)})
        elif provider == "gpt_image":
            for model in _openai_models():
                try:
                    image_bytes = _call_openai_generation(prompt, model)
                    width, height = normalize_image_to_jpeg(image_bytes, output_path, size)
                    return {
                        "provider": "openai-gpt-image",
                        "model": model,
                        "path": str(output_path),
                        "width": width,
                        "height": height,
                    }
                except Exception as exc:
                    errors.append({"provider": "openai-gpt-image", "model": model, "error": str(exc)})
        elif provider == "getimg":
            if not os.getenv("GETIMG_API_KEY"):
                errors.append({"provider": "getimg", "error": "GETIMG_API_KEY is not configured"})
                continue
            try:
                image_bytes = _call_getimg(prompt)
                width, height = normalize_image_to_jpeg(image_bytes, output_path, size)
                return {
                    "provider": "getimg",
                    "model": "flux-schnell",
                    "path": str(output_path),
                    "width": width,
                    "height": height,
                }
            except Exception as exc:
                errors.append({"provider": "getimg", "error": str(exc)})
        else:
            errors.append({"provider": provider, "error": "unknown provider"})

    raise AvatarGenerationError(
        "All avatar image providers failed",
        code="avatar_image_generation_failed",
        status_code=502,
        detail={"errors": errors},
    )


def _extract_video_url(value: Any) -> Optional[str]:
    if isinstance(value, str):
        if value.startswith("http://") or value.startswith("https://"):
            return value
        return None
    if isinstance(value, list):
        for item in value:
            found = _extract_video_url(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        for key in ("video_url", "url", "video", "output", "file", "download_url"):
            found = _extract_video_url(value.get(key))
            if found:
                return found
        for item in value.values():
            found = _extract_video_url(item)
            if found:
                return found
    return None


def generate_kling_motion(
    still_url: str,
    output_path: Path,
    *,
    motion_reference_video_url: Optional[str] = None,
    motion_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    api_key = os.getenv("KLING_2_6_MOTION_API_KEY") or os.getenv("SEGMIND_API_KEY")
    if not api_key:
        raise AvatarGenerationError(
            "KLING_2_6_MOTION_API_KEY or SEGMIND_API_KEY is required for avatar motion generation",
            code="segmind_api_key_missing",
        )

    payload = {
        "image": still_url,
        "input_video": motion_reference_video_url or os.getenv("KLING_2_6_MOTION_INPUT_VIDEO_URL"),
        "prompt": motion_prompt
        or os.getenv(
            "KLING_MOTION_PROMPT",
            "Create a seamless forever-looping idle avatar video with subtle natural breathing and micro-movements. Keep the head, shoulders, scale, camera framing, and eye line stable.",
        ),
        "keep_original_sound": _env_bool("KLING_2_6_MOTION_KEEP_ORIGINAL_SOUND", True),
        "character_orientation": os.getenv("KLING_2_6_MOTION_CHARACTER_ORIENTATION", "video"),
    }
    payload = {key: value for key, value in payload.items() if value not in (None, "")}
    timeout = _env_float("KLING_2_6_MOTION_TIMEOUT_SECONDS", 1200.0)
    response = requests.post(
        os.getenv("KLING_2_6_MOTION_API_URL", "https://api.segmind.com/v1/kling-2.6-pro-motion-control"),
        headers={"x-api-key": api_key, "Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise AvatarGenerationError(
            "Segmind Kling motion request failed",
            code="segmind_motion_failed",
            status_code=502,
            detail={"status": response.status_code, "body": _response_preview(response), "still_url": still_url},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("video/") or response.content[:8].endswith(b"ftyp"):
        output_path.write_bytes(response.content)
        return {"provider": "kling_2_6_pro_motion_control", "source": "bytes"}

    data = response.json()
    video_url = _extract_video_url(data)
    if not video_url:
        raise AvatarGenerationError(
            "Could not find a video URL in Segmind Kling response",
            code="segmind_response_invalid",
            status_code=502,
            detail={"response": data, "still_url": still_url},
        )

    download = requests.get(video_url, timeout=timeout)
    if download.status_code >= 400:
        raise AvatarGenerationError(
            "Failed to download Segmind Kling motion video",
            code="segmind_video_download_failed",
            status_code=502,
            detail={"status": download.status_code, "body": _response_preview(download), "url": video_url},
        )
    output_path.write_bytes(download.content)
    return {"provider": "kling_2_6_pro_motion_control", "source": "url", "url": video_url}


def generate_avatar_assets(request: GenerateAvatarRequest) -> Dict[str, Any]:
    prompt = resolve_generation_prompt(request)
    avatar_id = request.avatar_id or f"gpt_avatar_{uuid.uuid4().hex[:12]}"
    safe_avatar_id = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in avatar_id)
    base_dir = Path(os.getenv("GENERATED_AVATAR_ASSET_DIR", "generated/avatar_assets")) / safe_avatar_id
    base_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = base_dir / "prompt.txt"
    source_image_path = base_dir / "source.jpg"
    raw_motion_path = base_dir / "motion.raw.mp4"
    normalized_motion_path = base_dir / "motion.720x1280.30fps.10s.mp4"
    metadata_path = base_dir / "metadata.json"

    prompt_path.write_text(prompt, encoding="utf-8")
    started_at = time.time()

    still_result = generate_still_image(prompt, source_image_path, request.provider_order)
    image_bucket = os.getenv("GENERATED_IMAGE_BUCKET", "linguaprofilepics")
    image_prefix = os.getenv("GENERATED_IMAGE_PREFIX", "generated-images").strip("/")
    source_image_key = f"{image_prefix}/{safe_avatar_id}/source.jpg"
    source_image_url = _upload_file_to_s3(
        source_image_path,
        image_bucket,
        source_image_key,
        "image/jpeg",
    )

    motion_result = generate_kling_motion(
        source_image_url,
        raw_motion_path,
        motion_reference_video_url=request.motion_reference_video_url,
        motion_prompt=request.motion_prompt,
    )
    normalize_video(raw_motion_path, normalized_motion_path)

    motion_video_url = None
    motion_video_key = None
    if request.upload_video:
        video_bucket = os.getenv("IDLE_VIDEO_BUCKET", "lingua-ai-idle-vids")
        video_prefix = os.getenv("IDLE_VIDEO_PREFIX", "musetalk-generated").strip("/")
        motion_video_key = f"{video_prefix}/{safe_avatar_id}/motion.720x1280.30fps.10s.mp4"
        motion_video_url = _upload_file_to_s3(
            normalized_motion_path,
            video_bucket,
            motion_video_key,
            "video/mp4",
        )

    metadata = {
        "avatar_id": safe_avatar_id,
        "prompt": prompt,
        "image": {
            **still_result,
            "url": source_image_url,
            "bucket": image_bucket,
            "key": source_image_key,
        },
        "motion": {
            **motion_result,
            "raw_path": str(raw_motion_path),
            "normalized_path": str(normalized_motion_path),
            "url": motion_video_url,
            "key": motion_video_key,
            "reference_video_url": request.motion_reference_video_url
            or os.getenv("KLING_2_6_MOTION_INPUT_VIDEO_URL"),
            "prompt": request.motion_prompt or os.getenv("KLING_MOTION_PROMPT"),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "avatar_id": safe_avatar_id,
        "prompt": prompt,
        "image_provider": still_result["provider"],
        "image_model": still_result.get("model"),
        "source_image_url": source_image_url,
        "source_image_path": str(source_image_path),
        "motion_provider": "kling_2_6_pro_motion_control",
        "motion_video_url": motion_video_url,
        "motion_video_path": str(normalized_motion_path),
        "raw_motion_video_path": str(raw_motion_path),
        "metadata_path": str(metadata_path),
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
