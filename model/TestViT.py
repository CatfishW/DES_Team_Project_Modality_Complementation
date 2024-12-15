import unittest
import torch
from ViT import CustomViTForImageClassification

class TestCustomViTForImageClassification(unittest.TestCase):
    def setUp(self):
        self.model = CustomViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.rgb_tensor = torch.rand(1, 3, 224, 224)  # Random RGB image tensor
        self.depth_tensor = torch.rand(1, 1, 224, 224)  # Random Depth image tensor

    def test_rgb_image(self):
        with torch.no_grad():
            outputs = self.model(pixel_values=self.rgb_tensor)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class for RGB image:", self.model.config.id2label[predicted_class_idx])
        self.assertIsInstance(predicted_class_idx, int)

    def test_rgb_and_depth_image(self):
        with torch.no_grad():
            outputs = self.model(pixel_values=self.rgb_tensor, depth_values=self.depth_tensor)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class for RGB and Depth image:", self.model.config.id2label[predicted_class_idx])
        self.assertIsInstance(predicted_class_idx, int)

if __name__ == '__main__':
    unittest.main()