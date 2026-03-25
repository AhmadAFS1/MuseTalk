from PIL import Image
import numpy as np
import cv2


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）


def get_image_blending(image, face, face_box, mask_array, crop_box):
    """
    Blend a resized talking-face ROI back into the avatar frame.

    The original implementation converted the whole frame and ROI to PIL for
    every output frame. That adds significant CPU overhead on the live HLS hot
    path. This version keeps the operation in NumPy/OpenCV space while
    preserving the same masked-paste semantics.
    """
    plan = prepare_image_blending_plan(image.shape if image is not None else None, face_box, mask_array, crop_box)
    return get_image_blending_with_plan(image, face, plan)


def prepare_image_blending_plan(image_shape, face_box, mask_array, crop_box):
    """
    Precompute the static blending geometry for one avatar-cycle frame.

    The avatar frame size, bbox, crop box, and soft mask do not change during
    live generation, so this data can be cached once and reused for every
    composed frame in that cycle position.
    """
    if image_shape is None or mask_array is None:
        return None

    x, y, x1, y1 = [int(v) for v in face_box]
    x_s, y_s, x_e, y_e = [int(v) for v in crop_box]
    height, width = image_shape[:2]

    face_w = max(0, x1 - x)
    face_h = max(0, y1 - y)
    if face_w <= 0 or face_h <= 0:
        return None

    crop_w = max(0, x_e - x_s)
    crop_h = max(0, y_e - y_s)
    if crop_w <= 0 or crop_h <= 0:
        return None

    clip_x0 = max(0, x_s)
    clip_y0 = max(0, y_s)
    clip_x1 = min(width, x_e)
    clip_y1 = min(height, y_e)
    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
        return None

    mask_x0 = clip_x0 - x_s
    mask_y0 = clip_y0 - y_s
    mask_x1 = mask_x0 + (clip_x1 - clip_x0)
    mask_y1 = mask_y0 + (clip_y1 - clip_y0)

    full_face_x0 = x - x_s
    full_face_y0 = y - y_s
    full_face_x1 = full_face_x0 + face_w
    full_face_y1 = full_face_y0 + face_h

    place_x0 = max(mask_x0, full_face_x0)
    place_y0 = max(mask_y0, full_face_y0)
    place_x1 = min(mask_x1, full_face_x1)
    place_y1 = min(mask_y1, full_face_y1)

    overlay_dst_slice = None
    face_src_slice = None
    if place_x0 < place_x1 and place_y0 < place_y1:
        dst_x0 = place_x0 - mask_x0
        dst_y0 = place_y0 - mask_y0
        dst_x1 = place_x1 - mask_x0
        dst_y1 = place_y1 - mask_y0

        src_x0 = place_x0 - full_face_x0
        src_y0 = place_y0 - full_face_y0
        src_x1 = place_x1 - full_face_x0
        src_y1 = place_y1 - full_face_y0

        overlay_dst_slice = (slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
        face_src_slice = (slice(src_y0, src_y1), slice(src_x0, src_x1))

    mask_roi = mask_array[mask_y0:mask_y1, mask_x0:mask_x1]
    if mask_roi.ndim == 3:
        mask_roi = mask_roi[:, :, 0]
    mask_roi = np.ascontiguousarray(mask_roi)
    alpha = (mask_roi.astype(np.float32) / 255.0)[:, :, None]

    return {
        "face_size": (face_w, face_h),
        "clip_slice": (slice(clip_y0, clip_y1), slice(clip_x0, clip_x1)),
        "overlay_dst_slice": overlay_dst_slice,
        "face_src_slice": face_src_slice,
        "alpha": alpha,
    }


def get_image_blending_with_plan(image, face, plan):
    """Apply a precomputed blending plan to one resized talking-face frame."""
    if image is None or face is None or plan is None:
        return image

    clip_y_slice, clip_x_slice = plan["clip_slice"]
    base_roi = image[clip_y_slice, clip_x_slice]
    overlay_roi = base_roi.copy()

    overlay_dst_slice = plan["overlay_dst_slice"]
    face_src_slice = plan["face_src_slice"]
    if overlay_dst_slice is not None and face_src_slice is not None:
        dst_y_slice, dst_x_slice = overlay_dst_slice
        src_y_slice, src_x_slice = face_src_slice
        overlay_roi[dst_y_slice, dst_x_slice] = face[src_y_slice, src_x_slice]

    alpha = plan["alpha"]
    blended_roi = (
        overlay_roi.astype(np.float32) * alpha
        + base_roi.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    image[clip_y_slice, clip_x_slice] = blended_roi
    return image


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
