# is-your-cat-hungry

## Run inference
```python
import onnxruntime as rt
import numpy as np
import cv2

providers = ['CPUExecutionProvider']
labels = ['Brushing','Waiting For Food','Isolation']

model = './models/cats_224_best_02-0.68.onnx'

model = rt.InferenceSession(model, providers=providers)
print(model.get_inputs(), model.get_outputs())

def resize_img(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image.tolist()


x_0 = process_img_1('B_ANI01_MC_FN_SIM01_101_1.jpg')
x_1 = process_img_1('B_ANI01_MC_FN_SIM01_101_2.jpg')


onnx_pred = m.run(None, {'input_onnx_0':x_0, 'input_onnx_1':x_1})
preds = dict(zip(labels,onnx_pred[0][0].tolist()))

print(preds)

# results
{'Brushing': 0.8013108968734741, 'Waiting For Food': 0.08099100738763809, 'Isolation': 0.117698073387146}
```
