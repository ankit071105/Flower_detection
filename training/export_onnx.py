import torch
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large

def main():
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    model.head.classification_head.num_classes = 2
    model.load_state_dict(torch.load("flower_ssd.pth"))
    model.eval()

    dummy = torch.randn(1, 3, 320, 320)

    torch.onnx.export(
        model,
        dummy,
        "flower_ssd.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )

    print("Export complete → flower_ssd.onnx")

if __name__ == "__main__":
    main()
