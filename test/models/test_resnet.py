import unittest
from test.helpers import TrackedTestCase
from extra.models import resnet

class TestResnet(TrackedTestCase):
  def test_model_load(self):
    model = resnet.ResNet18()
    model.load_from_pretrained()

    model = resnet.ResNeXt50_32X4D()
    model.load_from_pretrained()


if __name__ == '__main__':
  unittest.main()