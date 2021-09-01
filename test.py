import os
import numpy as np
from src.dataset import MRIdataset
from src.model import get_model


# Load best weights.
checkpoint_path = './model/3d_image_classification.h5'
model.load_weights(checkpoint_path)
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )