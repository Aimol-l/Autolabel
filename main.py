from utils.YoloDet import YoloDet
from utils.YoloSeg import YoloSeg
from utils.YoloObb import YoloObb

if __name__ == "__main__":

    label_path = "assets/labels/"
    output_path = "assets/outputs/"

    # det = YoloDet("models/detect.onnx",size=(1024,1024))
    # det.set_params(needs=[],conf=0.1, nms=0.5, classes="config/detect.yaml")
    # det.inference("assets/images/detect", output_path,label_path,"labelyou",padding=True)

    # obb = YoloObb("models/obb.onnx",size=(1024,1024))
    # obb.set_params(needs=[0],conf=0.5, nms=0.5, classes="config/obb.yaml")
    # obb.inference("assets/images/obb", output_path,label_path,"none",padding=True)

    seg = YoloSeg("models/seg-yolo-fp32.onnx",size=(224,512))
    seg.set_params(needs=[],conf=0.5, nms=0.8, bin=0.5, classes="config/seg.yaml")
    seg.inference("assets/images/segment", output_path,label_path,label_type="none",padding=False)