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

class YoloSeg:
    def __init__(self, model_path, size=(640,640)):
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)
        self.session = onnxruntime.InferenceSession(model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if onnxruntime.get_device() == "GPU" else ["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        self.nms = 0.5
        self.conf = 0.5
        self.bin = 0.5
        self.classes = {}
        self.needs = []

        output_shape = self.session.get_outputs()[0].shape  # [1,116, 8400]
        if isinstance(self.input_shape[0],str):
            self.input_shape = [1, 3, size[1], size[0]]
            self.input_width ,self.input_height = size
            input_test = np.random.randn(*self.input_shape).astype(np.float32)
            outputs = self.session.run(self.output_names, {self.input_name: input_test})[0]
            output_shape = outputs.shape
            print("dynamic model, output shape:", output_shape)

        self.num_classes = output_shape[1] - 4 - 32
        random.seed(42)
        self.color_map = {
            i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for i in range(self.num_classes)
        }

    def set_params(self, needs=[], conf=0.5, nms=0.5, bin=0.5, classes="config/coco.yaml"):
        self.nms = nms
        self.conf = conf
        self.bin = bin
        with open(classes, 'r') as f:
            data = yaml.safe_load(f)
            self.classes = data.get('names', {})
        if not needs:
            self.needs = list(range(self.num_classes))
        else:
            self.needs = []
            for need in needs:
                if isinstance(need, int):
                    self.needs.append(need if need >= 0 else self.num_classes + need)
                elif isinstance(need, str):
                    if need in self.classes.values():
                        self.needs.append(list(self.classes.keys())[list(self.classes.values()).index(need)])
                    else:
                        raise Exception(f"{need} not found in .yaml")

    def inference(self, input_path, output_dir,label_dir,label_type="none",padding=True):
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
        for img_path in tqdm(image_paths, desc="Segmenting"):
            end = os.path.splitext(img_path)[1]
            ori_path = os.path.basename(img_path)
            fname = os.path.splitext(os.path.basename(img_path))[0]
            out_img_path = os.path.join(output_dir, fname + ".jpg")

            if padding:
                ori_img, input_tensor, scale, pad = self._preprocess_padding(img_path)
                output0, output1 = self.session.run(self.output_names, {self.input_name: input_tensor})
                boxes, scores, class_ids, mask_coeffs = self._postprocess(output0, scale, pad, ori_img.shape[:2])
            else:
                ori_img, input_tensor = self._preprocess_resize(img_path)
                # output0 :[1,8400,38]
                # output1 :[1,32,160,160]
                output0, output1 = self.session.run(self.output_names, {self.input_name: input_tensor})
                boxes, scores, class_ids, mask_coeffs = self._postprocess(output0, ori_img.shape[:2])

            if self.needs:
                mask = np.isin(class_ids, self.needs)
                boxes, scores, class_ids,mask_coeffs = boxes[mask], scores[mask], class_ids[mask],mask_coeffs[mask]

            if len(boxes) == 0:
                no_detections.append(fname+f"{end}")
                continue

            protos = output1[0]  # [32,160,160]
            protos = protos.reshape(32, -1)  # [32,25600]
            masks = mask_coeffs @ protos  # [N,25600]
            masks = 1 / (1 + np.exp(-masks))
            masks = masks.reshape(-1,output1[0].shape[1],output1[0].shape[2])
            mask_images = []
            blur_size = (int(ori_img.shape[1]/masks.shape[2]),int(ori_img.shape[0]/masks.shape[1]))
            for m in masks:
                # m不能直接resize到原图大小,因为推理图片是经过padding的.得先去除padding再resize
                if padding:
                    m = cv2.resize(m, (m.shape[0]*4,m.shape[1]*4))
                    m = m[pad[0]:self.input_shape[2]-pad[0],pad[1]:self.input_shape[3]-pad[1]]
                    m = cv2.resize(m, (ori_img.shape[1], ori_img.shape[0]))
                else:
                    m = cv2.resize(m, (ori_img.shape[1], ori_img.shape[0]),interpolation=cv2.INTER_CUBIC)
                # 高斯模糊
                m = cv2.blur(m,blur_size)
                m = (m > self.bin).astype(np.uint8) * 255
                mask_images.append(m)

            for box, score, class_id, mask in zip(boxes, scores, class_ids, mask_images):
                x1, y1, x2, y2 = map(int, box)
                label = self.classes.get(int(class_id), str(class_id))
                color = self.color_map.get(int(class_id), (255, 255, 0))

                color_mask = np.zeros_like(ori_img)
                for c in range(3):
                    color_mask[:, :, c] = color[c]
                mask_ = np.zeros(ori_img.shape[:2], dtype=np.uint8) * 255
                mask_ = cv2.rectangle(mask_, (x1, y1), (x2, y2), (255), -1)
                mask = cv2.bitwise_and(mask, mask_)
                mask_bool = mask > 0

                if label_type == "labelyou":
                    label_path = os.path.join(label_dir, fname + f".json")
                    self._gen_labelyou(class_id,label_path,ori_path,mask,ori_img.shape)

                ori_img[mask_bool] = cv2.addWeighted(ori_img, 0.5, color_mask, 0.5, 0)[mask_bool]
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(ori_img, (x1, y1-17), (x2, y1), color, -1)
                cv2.putText(ori_img, f"{label}:{score:.2f}", (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imwrite(out_img_path, ori_img)
       
        if no_detections:
            print("\nNo Segments in the following images:")
            for index,name in enumerate(no_detections):
                print(f" {index} -", name)

    def _gen_labelyou(self, label, json_save_path, image_path, mask,shape):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_list = []
        for contour in contours:
            if contour.size == 0:
                continue
            # 外接矩形（x, y, w, h）
            x, y, w, h = cv2.boundingRect(contour)
            bbox = {
                "x": str(float(x)),
                "y": str(float(y)),
                "width": str(float(w)),
                "height": str(float(h))
            }
            points = [[float(pt[0][0]), float(pt[0][1])] for pt in contour]
            shape_list.append({
                "label": str(label),
                "bbox": bbox,
                "points": points
            })

        h, w = shape[:2]

        labelme_dict = {
            "image_path": os.path.basename(image_path),
            "autoGen": 0,
            "imageWidth": int(w),
            "imageHeight": int(h),
            "Rectangle_label": [],
            "RoRectangle_label": [],
            "shape": shape_list,
            "Circle_label": []
        }

        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_dict, f, indent=2, ensure_ascii=False)


    def _preprocess_padding(self, input_path):
        image = cv2.imread(input_path)
        h0, w0 = image.shape[:2]
        r = min(self.input_width / w0, self.input_height / h0)
        new_size = (int(w0 * r), int(h0 * r))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad = ((self.input_height - new_size[1]) // 2, (self.input_width - new_size[0]) // 2)

        canvas[pad[0]:pad[0]+new_size[1], pad[1]:pad[1]+new_size[0]] = resized

        # cv2.imshow("resized", canvas)
        # cv2.waitKey(0)

        img = canvas.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        return image, img.astype(np.float32), r, pad
    
    def _preprocess_resize(self, input_path):
        image = cv2.imread(input_path)
        temp = cv2.resize(image, (self.input_width, self.input_height))
        img = cv2.dnn.blobFromImage(temp, 1/255.0, (self.input_width, self.input_height), swapRB=True)
        return image, img

    def _postprocess(self, output, orig_shape):
        predictions = output[0] if len(output.shape) == 3 else output
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes = predictions[..., :4]
        class_scores = predictions[..., 4:4+self.num_classes]
        mask_coeffs = predictions[..., 4+self.num_classes:]

        class_ids = np.argmax(class_scores, axis=-1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        mask = confidences > self.conf

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]

        boxes[:, [0, 2]] /= self.input_width
        boxes[:, [1, 3]] /= self.input_height
        boxes[:, [0, 2]] *= orig_shape[1]
        boxes[:, [1, 3]] *= orig_shape[0]
        boxes = self._xywh2xyxy(boxes)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])

         # 按类别分组，分别做 NMS
        final_boxes = []
        final_scores = []
        final_class_ids = []
        final_masks = []
        unique_classes = np.unique(class_ids)
        for cls in unique_classes:
            cls_idx_mask = (class_ids == cls)
            cls_boxes = boxes[cls_idx_mask]
            cls_scores = confidences[cls_idx_mask]
            cls_ids = class_ids[cls_idx_mask]
            cls_mask = mask_coeffs[cls_idx_mask]
            keep = self._nms(cls_boxes, cls_scores, self.nms)
            if len(keep) > 0:
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_class_ids.append(cls_ids[keep])
                final_masks.append(cls_mask[keep])

        if not final_boxes:
            return np.array([]), np.array([]), np.array([])
        # 合并所有类别的结果
        final_boxes = np.vstack(final_boxes)
        final_scores = np.hstack(final_scores)
        final_class_ids = np.hstack(final_class_ids)
        final_masks = np.vstack(final_masks)

        return final_boxes, final_scores, final_class_ids, final_masks

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
