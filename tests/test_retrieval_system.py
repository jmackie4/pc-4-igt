import unittest,types,tempfile,os
from src import retrieval_system as rs
from datasets import Dataset

class TestOverlapScore(unittest.TestCase):
    def setUp(self):
        self.sting_w_space = 'I like potatoes. I also like to eat cheese'
        self.string_no_space = 'Ilikepotatoes'
        self.int_input = 10

    def test_int_input(self):
        with self.assertRaises(AttributeError):
            rs.get_word_overlap_score(self.int_input,self.string_w_space)

    def test_spaced_w_no_space(self):
        self.assertEqual(rs.get_word_overlap_score(self.string_no_space,self.sting_w_space),0)


class TestInformationRetrievalSystem(unittest.TestCase):
    def setUp(self):
        self.test_dataset = Dataset.from_dict({'key1':['this is for row 1','this is for row 2'],
                                               'key2':['this is column 2 row 1','this is column 2 row 2'],
                                               'key3':['this is column 3 row 1','this is column 3 row 2'],})

    def test_string_w_non_zero_wos(self):
        test_irs = rs.InformationRetrievalSystem(self.test_dataset,num_examples=2)
        output = test_irs.get_examples_from_query('I want something for row 1','key1')

        self.assertEqual(output,self.test_dataset.to_dict())

    def test_irs_no_examples(self):
        test_irs = rs.InformationRetrievalSystem(self.test_dataset,num_examples=0)
        output = test_irs.get_examples_from_query('I want something for row 1','key1')

        self.assertEqual(output,{'key1':[],'key2':[],'key3':[]})

    def test_irs_no_matches(self):
        test_irs = rs.InformationRetrievalSystem(self.test_dataset,num_examples=1)
        output = test_irs.get_examples_from_query('Give me something to ear now','key2')

        self.assertEqual(output,self.test_dataset[[0]])
        self.assertNotEqual(output,self.test_dataset[0])
        self.assertNotEqual(output,self.test_dataset[[1]])