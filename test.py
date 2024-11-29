import unittest

import data


class TestData(unittest.TestCase):
    def test_get_data(self):
        data_set = data.DataSet()
        for i in range(data_set.__len__()):
            print(data_set.__getitem__(i))


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(TestData('test_upper'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
