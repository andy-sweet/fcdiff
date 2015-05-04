import sys, os, unittest

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir, '..', 'src'))
import fcdiff.models

class OneSampleModelTestCase(unittest.TestCase):

    def test_sample_R_1(self):
        """
        Tests the sample method with some valid arguments.
	"""
	model = fcdiff.models.OneSampleModel()
	R = model.sample_R(3, 4)
	assert R.shape == (3, 4, 2)

