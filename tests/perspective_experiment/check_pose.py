import torch, warnings
warnings.filterwarnings('ignore')

d = torch.load(r'c:\Users\Abel\Desktop\InfractiVision\models\yolov8n-pose.pt', 
               map_location='cpu', weights_only=False)

print("Keys:", list(d.keys()))
print("\nTrain args:", d.get('train_args', {}))

model = d.get('model', None)
if model is not None:
    print("\nModel type:", type(model))
    if hasattr(model, 'names'):
        print("Classes:", model.names)
    if hasattr(model, 'yaml'):
        print("YAML:", model.yaml)
    if hasattr(model, 'kpt_shape'):
        print("Keypoint shape:", model.kpt_shape)
    # Check model structure
    if hasattr(model, 'model'):
        for i, m in enumerate(model.model):
            print(f"  Layer {i}: {m.__class__.__name__}", end="")
            if hasattr(m, 'cv1'):
                print(f" (in={m.cv1.conv.in_channels}, out={m.cv1.conv.out_channels})", end="")
            print()
            if i > 25:
                break
