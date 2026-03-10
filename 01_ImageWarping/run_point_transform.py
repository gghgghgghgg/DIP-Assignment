import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    warped_image = np.array(image)
    ### FILL: Implement MLS or RBF based image warping

    if len(source_pts) == 0 or len(target_pts) == 0 or len(source_pts) != len(target_pts):
        return warped_image

    h, w = image.shape[:2]
    p = target_pts.astype(np.float64)  # (n, 2)
    q = source_pts.astype(np.float64)  # (n, 2)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)  # (h*w, 2)

    diff = v[:, np.newaxis, :] - p[np.newaxis, :, :]   # (h*w, n, 2)
    dist2 = np.sum(diff ** 2, axis=-1)                  # (h*w, n)
    dist2 = np.maximum(dist2, eps)

    #权重
    w_i = 1.0 / (dist2 ** alpha)                        # (h*w, n)
    w_sum = w_i.sum(axis=1, keepdims=True)               # (h*w, 1)

    # 加权重心
    p_star = (w_i[:, :, np.newaxis] * p[np.newaxis]).sum(axis=1) / w_sum  # (h*w, 2)
    q_star = (w_i[:, :, np.newaxis] * q[np.newaxis]).sum(axis=1) / w_sum  # (h*w, 2)

    # 中心控制点
    p_hat = p[np.newaxis] - p_star[:, np.newaxis, :]    # (h*w, n, 2)
    q_hat = q[np.newaxis] - q_star[:, np.newaxis, :]    # (h*w, n, 2)

    wp_hat = w_i[:, :, np.newaxis] * p_hat              # (h*w, n, 2)

    # MLS Affine
    PtWP = np.einsum('bni,bnj->bij', wp_hat, p_hat)     # (h*w, 2, 2)
    PtWQ = np.einsum('bni,bnj->bij', wp_hat, q_hat)     # (h*w, 2, 2)

    PtWP[:, 0, 0] += eps
    PtWP[:, 1, 1] += eps
    A = np.linalg.solve(PtWP, PtWQ)                     # (h*w, 2, 2)

    # Source position: f(v) = (v - p*) @ A + q*
    v_hat = v - p_star                                   # (h*w, 2)
    f_v = np.einsum('bi,bij->bj', v_hat, A) + q_star    # (h*w, 2)

    map_x = f_v[:, 0].reshape(h, w).astype(np.float32)
    map_y = f_v[:, 1].reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(image, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
