from ultralytics import YOLO

def main():
    # First step, selecting the model, we'll use the lighter one
    model = YOLO('yolov8n.pt')

    # Second, we fine tune the model with our dataset
    results = model.train(
        data='HandSigns_v2/data.yaml', 
        epochs=100,       
        imgsz=640,         
        batch=16,         
        patience=20       
    )

    # We then evaluate the model's performance on the validation set
    results = model.val()

    # And we export the model for deployment
    model.export(format='onnx')

if __name__ == '__main__':
    main()