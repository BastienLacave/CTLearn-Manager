import unittest
import ctlearn_manager

class TestVersion(unittest.TestCase):
    def test_version(self):
        expected_version = "0.1.dev1+g4ad0be7"  # Replace with the expected version
        self.assertEqual(ctlearn_manager.__version__, expected_version)

if __name__ == '__main__':
    unittest.main()