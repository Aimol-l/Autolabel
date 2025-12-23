import cv2
import onnx
import yaml
import json
import glob
import os
import random
import numpy as np
import onnxruntime

from tqdm import tqdm


class YoloObb:
    def __init__(self, model_path,size = (640,640)):

        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)
        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] 
                                                    if onnxruntime.get_device() == "GPU" else ["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        self.nms = 0.5
        self.conf = 0.5
        self.classes = {}
        self.needs = []
        # 推断类别数量,有可能是动态模型,需要提前推理一次
        output_shape = self.session.get_outputs()[0].shape  # [1, N, 84] 或 [batch_size, N, 84]
        # print("model output shape:", output_shape)
        # 判断output_shape[0]数据类型
        if isinstance(output_shape[0], str) or  output_shape[0] < 0:
            self.dynamic = True
            self.input_shape = [1, 3, size[1], size[0]]
            self.input_width ,self.input_height = size
            # 推理一次得到输出shape
            input_test = np.random.randn(*self.input_shape).astype(np.float32)
            outputs = self.session.run(self.output_names, {self.input_name: input_test})[0]
            output_shape = outputs.shape
            print("dynamic model, output shape:", output_shape)

        self.num_classes = output_shape[2] - 5  # 5 = (cx,cy,w,h,angle)
        # 生成每个类别的颜色映射（通过固定的随机种子确保每次都一致）
        random.seed(43)  # 使得每次生成的颜色一致
        self.color_map = {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
            for i in range(self.num_classes)
        }

    def set_params(self, needs = [], conf=0.5, nms=0.5,classes="config/obb.yaml"):
        self.nms = nms
        self.conf = conf

        with open(classes, 'r') as f:
            data = yaml.safe_load(f)
            self.classes = data.get('names', {})
        if(len(needs) == 0):
            self.needs = [i for i in range(self.num_classes)]
        else:
            for need in needs:
                # 如果need为数字，则添加到needs中
                if isinstance(need, int):
                    if need > self.num_classes:
                        # 抛出异常
                        raise Exception("need index out of range,max is {}".format(self.num_classes))
                    if need < 0:
                        need = self.num_classes + need
                    self.needs.append(need)
                # 如果need为字符串，则根据字符串查找对应的数字并添加到needs中
                elif isinstance(need, str):
                    # 判断self.classes里是否包含need
                    if need in self.classes.values():
                        self.needs.append(list(self.classes.keys())[list(self.classes.values()).index(need)])
                    else:
                        raise Exception("{} not found in .yaml".format(need))
    def inference(self, input_path, output_dir, label_dir, label_type="none",padding=True):
        if os.path.isdir(input_path):
            exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_paths = []
            for ext in exts:
                image_paths.extend(glob.glob(os.path.join(input_path, ext)))
            image_paths.sort()
        else:
            image_paths = [input_path]

        # 先删除输出文件夹
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
        if os.path.exists(label_dir):
            for f in os.listdir(label_dir):
                os.remove(os.path.join(label_dir, f))
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        no_detections = []

        for img_path in tqdm(image_paths, desc="Running Inference"):
            end = os.path.splitext(img_path)[1]
            ori_path = os.path.basename(img_path)
            fname = os.path.splitext(os.path.basename(img_path))[0]

            out_img_path = os.path.join(output_dir, fname + ".jpg")

            if padding:
                ori_img, input_tensor, scale, pad = self._preprocess(img_path)
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
                roboxes, scores, class_ids = self._postprocess(outputs[0], scale, pad, ori_img.shape[:2])
            else:
                ori_img, input_tensor = self._preprocess_resize(img_path)
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
                roboxes, scores, class_ids = self._postprocess_resize(outputs[0],ori_img.shape[:2])
            if self.needs:
                mask = np.isin(class_ids, self.needs)
                roboxes = roboxes[mask]
                scores = scores[mask]
                class_ids = class_ids[mask]
            if len(roboxes) == 0:
                no_detections.append(fname+f"{end}")
                continue

            # 可视化旋转框
            for box, score, class_id in zip(roboxes, scores, class_ids):
                cx, cy, w, h, angle = box
                label = self.classes.get(int(class_id), str(class_id))
                color = self.color_map.get(int(class_id), (255, 255, 0))  # 默认颜色为青色
                # 创建 RotatedRect 格式：((cx, cy), (w, h), angle)
                rect = ((cx, cy), (w, h), angle)
                # 获取旋转矩形的 4 个顶点
                box_points = cv2.boxPoints(rect)
                box_points = np.int32(box_points)  # 转换为整数坐标
                # 绘制旋转矩形
                cv2.drawContours(ori_img, [box_points], 0, color, 3)
                # 添加标签文本
                text = f"{label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                # 获取文本尺寸用于背景矩形绘制
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                top_left = box_points[0]  # 取第一个点作为左上角位置
                text_origin = (top_left[0], top_left[1] - 5)  # 文字在框上方一点
                # 绘制文字背景
                cv2.rectangle(ori_img,
                            (text_origin[0], text_origin[1] - text_height - 2),
                            (text_origin[0] + text_width, text_origin[1] + 2),
                            color, -1)
                # 绘制文字
                cv2.putText(ori_img, text, text_origin, font, font_scale, (0, 0, 0), thickness)

            cv2.imwrite(out_img_path, ori_img)

            if label_type == "labelyou":
                label_path = os.path.join(label_dir, fname + f".json")
                self._gen_custom_label(roboxes, scores, class_ids, label_path,ori_path,ori_img.shape)

        if no_detections:
            print("\nNo detections in the following images:")
            for index,name in enumerate(no_detections):
                print(f" {index} -", name)

    def _preprocess_resize(self,input_path):
        image = cv2.imread(input_path)
        img = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_width, self.input_height), swapRB=True)
        return image, img
    
    def _postprocess_resize(self, output, orig_shape):
        predictions = output
        if predictions.ndim == 3:
            predictions = predictions[0]  # [1, N, C] → [N, C]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]
        scores = predictions[..., 4:-1]
        angle = predictions[..., -1] * 180 / np.pi  # 弧度转角度

        class_ids = np.argmax(scores, axis=-1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # 置信度过滤
        mask = confidences > self.conf
        cx, cy, w, h, angle, confidences, class_ids = (
            cx[mask], cy[mask], w[mask], h[mask], angle[mask],
            confidences[mask], class_ids[mask]
        )
        # 反缩放和裁剪坐标
        scale_x = self.input_width / orig_shape[1]
        scale_y = self.input_height / orig_shape[0]
        cx /= scale_x
        cy /= scale_y
        w /= w
        h /= h
        final_roboxes = []      # 存储 (cx, cy, w, h, angle_rad)
        final_scores = []
        final_classes = []
        for cls in np.unique(class_ids):
            cls_mask = (class_ids == cls)
            cls_cx = cx[cls_mask]
            cls_cy = cy[cls_mask]
            cls_w = w[cls_mask]
            cls_h = h[cls_mask]
            cls_angle_deg = angle[cls_mask]
            cls_conf = confidences[cls_mask]
            cls_rorect = []
            for i in range(len(cls_cx)):
                cls_rorect.append((
                    (float(cls_cx[i]), float(cls_cy[i])),
                    (float(cls_w[i]), float(cls_h[i])),
                    float(cls_angle_deg[i])
                ))
            # 执行 Rotated NMS
            indices = cv2.dnn.NMSBoxesRotated(
                bboxes=cls_rorect,
                scores=cls_conf.astype(float).tolist(),
                score_threshold=self.conf,
                nms_threshold=self.nms
            )
            if len(indices) == 0:
                continue
            # 标准化 indices
            if isinstance(indices, (list, tuple)):
                indices = np.array(indices, dtype=int).flatten()
            else:
                indices = indices.flatten()
            # 收集原始值（保持 angle 为 radian）
            keep_cx = cls_cx[indices]
            keep_cy = cls_cy[indices]
            keep_w = cls_w[indices]
            keep_h = cls_h[indices]
            keep_angle = cls_angle_deg[indices]
            roboxes = np.stack([keep_cx, keep_cy, keep_w, keep_h, keep_angle], axis=1)
            final_roboxes.append(roboxes)
            final_scores.append(cls_conf[indices])
            final_classes.append(class_ids[cls_mask][indices])
        # 合并
        if not final_roboxes:
            return np.array([]), np.array([]), np.array([])
        final_roboxes = np.vstack(final_roboxes)
        final_scores = np.hstack(final_scores)
        final_classes = np.hstack(final_classes)
        return final_roboxes, final_scores, final_classes
        

    def _preprocess(self, input_path):
        image = cv2.imread(input_path)
        h0, w0 = image.shape[:2]
        r = min(self.input_width / w0, self.input_height / h0)
        new_size = (int(w0 * r), int(h0 * r))
        resized = cv2.resize(image, new_size)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad = ((self.input_height - new_size[1]) // 2, (self.input_width - new_size[0]) // 2)
        canvas[pad[0]:pad[0]+new_size[1], pad[1]:pad[1]+new_size[0]] = resized

        img = canvas.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        return image, img.astype(np.float32), r, pad

    def _postprocess(self, output, scale, pad, orig_shape):
        predictions = output
        if predictions.ndim == 3:
            predictions = predictions[0]  # [1, N, C] → [N, C]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        cx = predictions[:, 0]
        cy = predictions[:, 1]
        w = predictions[:, 2]
        h = predictions[:, 3]
        scores = predictions[..., 4:-1]
        angle = predictions[..., -1] * 180 / np.pi  # 弧度转角度

        class_ids = np.argmax(scores, axis=-1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # 置信度过滤
        mask = confidences > self.conf
        cx, cy, w, h, angle, confidences, class_ids = (
            cx[mask], cy[mask], w[mask], h[mask], angle[mask],
            confidences[mask], class_ids[mask]
        )
        # 反缩放和裁剪坐标
        scale_factor = scale
        cx = (cx - pad[1]) / scale_factor
        cy = (cy - pad[0]) / scale_factor
        w /= scale_factor
        h /= scale_factor
        final_roboxes = []      # 存储 (cx, cy, w, h, angle_rad)
        final_scores = []
        final_classes = []
        for cls in np.unique(class_ids):
            cls_mask = (class_ids == cls)
            cls_cx = cx[cls_mask]
            cls_cy = cy[cls_mask]
            cls_w = w[cls_mask]
            cls_h = h[cls_mask]
            cls_angle_deg = angle[cls_mask]
            cls_conf = confidences[cls_mask]
            cls_rorect = []
            for i in range(len(cls_cx)):
                cls_rorect.append((
                    (float(cls_cx[i]), float(cls_cy[i])),
                    (float(cls_w[i]), float(cls_h[i])),
                    float(cls_angle_deg[i])
                ))
            # 执行 Rotated NMS
            indices = cv2.dnn.NMSBoxesRotated(
                bboxes=cls_rorect,
                scores=cls_conf.astype(float).tolist(),
                score_threshold=self.conf,
                nms_threshold=self.nms
            )
            if len(indices) == 0:
                continue
            # 标准化 indices
            if isinstance(indices, (list, tuple)):
                indices = np.array(indices, dtype=int).flatten()
            else:
                indices = indices.flatten()
            # 收集原始值（保持 angle 为 radian）
            keep_cx = cls_cx[indices]
            keep_cy = cls_cy[indices]
            keep_w = cls_w[indices]
            keep_h = cls_h[indices]
            keep_angle = cls_angle_deg[indices]
            roboxes = np.stack([keep_cx, keep_cy, keep_w, keep_h, keep_angle], axis=1)
            final_roboxes.append(roboxes)
            final_scores.append(cls_conf[indices])
            final_classes.append(class_ids[cls_mask][indices])
        # 合并
        if not final_roboxes:
            return np.array([]), np.array([]), np.array([])
        final_roboxes = np.vstack(final_roboxes)
        final_scores = np.hstack(final_scores)
        final_classes = np.hstack(final_classes)
        return final_roboxes, final_scores, final_classes


    def _xywh2xyxy(self, boxes):
        xyxy = np.zeros_like(boxes)
        xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
        return xyxy

    def _gen_custom_label(self, roboxes, scores, classes, save_path, ori_path, shape):
        annotations = {
            "image_path": ori_path,
            "autoGen": 1,
            "imageWidth": shape[1],
            "imageHeight": shape[0],
            "Rectangle_label": [],
            "RoRectangle_label": [],
            "shape": [],
            "Circle_label": []
        }
        for box, score, cls in zip(roboxes, scores, classes):
            cx, cy, w, h, angle = box
            # 转为字符串（保持与你示例一致）
            sub = {
                "label": str(cls),
                "coordinates": {
                    "cx": str(float(cx)),
                    "cy": str(float(cy)),
                    "width": str(float(w)),
                    "height": str(float(h)),
                    "angle": str(float(angle))  # 弧度
                }
            }
            annotations["RoRectangle_label"].append(sub)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

