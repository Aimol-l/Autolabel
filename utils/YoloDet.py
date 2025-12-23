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


class YoloDet:
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

        self.num_classes = output_shape[2] - 4  # 84 - 4 = 80 个类别
        # 生成每个类别的颜色映射（通过固定的随机种子确保每次都一致）
        random.seed(43)  # 使得每次生成的颜色一致
        self.color_map = {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
            for i in range(self.num_classes)
        }

    def set_params(self, needs = [], conf=0.5, nms=0.5,classes="config/coco.yaml"):
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
                boxes, scores, class_ids = self._postprocess(outputs[0], scale, pad, ori_img.shape[:2])
            else:
                ori_img, input_tensor = self._preprocess_resize(img_path)
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
                boxes, scores, class_ids = self._postprocess_resize(outputs[0],ori_img.shape[:2])

            if self.needs:
                mask = np.isin(class_ids, self.needs)
                boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

            if len(boxes) == 0:
                no_detections.append(fname+f"{end}")
                continue
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.classes.get(int(class_id), str(class_id))
                color = self.color_map.get(int(class_id), (255, 255, 0))  # 默认颜色为白色
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(ori_img, f"{label} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imwrite(out_img_path, ori_img)

            if label_type == "labelyou":
                label_path = os.path.join(label_dir, fname + f".json")
                self._gen_custom_label(boxes, scores, class_ids, label_path,ori_path,ori_img.shape)

        if no_detections:
            print("\nNo detections in the following images:")
            for index,name in enumerate(no_detections):
                print(f" {index} -", name)

    def _preprocess_resize(self,input_path):
        image = cv2.imread(input_path)
        img = cv2.dnn.blobFromImage(image, 1/255.0, (self.input_width, self.input_height), swapRB=True)
        return image, img
    def _postprocess_resize(self,output,orig_shape):
        predictions = output[0] if len(output.shape) == 3 else output
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        boxes = predictions[..., :4]
        scores = predictions[..., 4:]
        class_ids = np.argmax(scores, axis=-1)
        confidences = scores[np.arange(len(scores)), class_ids]
        # 置信度过滤
        mask = confidences > self.conf
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        scale_x = self.input_width / orig_shape[1]
        scale_y = self.input_height / orig_shape[0]
        # 转换坐标格式并还原到原始图像尺寸
        boxes = self._xywh2xyxy(boxes)
        boxes[:, [0, 2]] /= scale_x
        boxes[:, [1, 3]] /= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
        # 按类别分组，分别做 NMS
        final_boxes = []
        final_scores = []
        final_class_ids = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = (class_ids == cls)
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_ids = class_ids[cls_mask]
            keep = self._nms(cls_boxes, cls_scores, self.nms)
            if len(keep) > 0:
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_class_ids.append(cls_ids[keep])

        if not final_boxes:
            return np.array([]), np.array([]), np.array([])
        # 合并所有类别的结果
        final_boxes = np.vstack(final_boxes)
        final_scores = np.hstack(final_scores)
        final_class_ids = np.hstack(final_class_ids)
        return final_boxes, final_scores, final_class_ids

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
        predictions = output[0] if len(output.shape) == 3 else output
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        boxes = predictions[..., :4]
        scores = predictions[..., 4:]
        class_ids = np.argmax(scores, axis=-1)
        confidences = scores[np.arange(len(scores)), class_ids]
        # 置信度过滤
        mask = confidences > self.conf
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = self._cxcywh2xyxy(boxes)
        # 还原到原始图像尺寸（去 pad & 反缩放）
        boxes[:, [0, 2]] -= pad[1]   # x 方向 padding
        boxes[:, [1, 3]] -= pad[0]   # y 方向 padding
        boxes /= scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])  # width
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])  # height
        # 按类别分组，分别做 NMS
        final_boxes = []
        final_scores = []
        final_class_ids = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_mask = (class_ids == cls)
            cls_boxes = boxes[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_ids = class_ids[cls_mask]
            keep = self._nms(cls_boxes, cls_scores, self.nms)
            if len(keep) > 0:
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_class_ids.append(cls_ids[keep])

        if not final_boxes:
            return np.array([]), np.array([]), np.array([])
        # 合并所有类别的结果
        final_boxes = np.vstack(final_boxes)
        final_scores = np.hstack(final_scores)
        final_class_ids = np.hstack(final_class_ids)
        return final_boxes, final_scores, final_class_ids
    def _nms(self, boxes, scores, iou_threshold):
        # boxes: (N, 4) in xyxy
        # scores: (N,)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (area_i + area_rest - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]  # +1 because we skipped the first
        return np.array(keep)
    def _xywh2xyxy(self, boxes):
        xyxy = np.zeros_like(boxes)
        xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
        return xyxy
    def _cxcywh2xyxy(self, boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _gen_custom_label(self, boxes, scores, classes, save_path,ori_path,shape):
        annotations = {}
        annotations["image_path"] = ori_path
        annotations["autoGen"] = 1
        annotations["imageWidth"] = shape[1]
        annotations["imageHeight"] = shape[0]

        annotations["Rectangle_label"] = []
        annotations["RoRectangle_label"] = []
        annotations["shape"] = []
        annotations["Circle_label"] = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x, y, w, h = box[0],box[1],box[2] - box[0],box[3] - box[1]
            sub = {"label":str(cls),"coordinates":{"x":str(x),"y":str(y),"width":str(w),"height":str(h)}}
            annotations["Rectangle_label"].append(sub)

        with open(save_path, "w") as f:
            json.dump(annotations, f, indent=2)

