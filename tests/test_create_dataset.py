import unittest,types,tempfile,os
from src import create_dataset as cd
from datasets import Dataset


class Test_Path_Transformer(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.gold_path = self.temp_directory.name
        self.gold_output = {'main_path': self.gold_path,
                            'result_list': os.listdir(self.gold_path)}
        self.transformer = cd.Path_Transformer()

    def test_gold_path(self):
        output = self.transformer.transform(self.gold_path)
        self.assertEqual(output, self.gold_output)

    def test_non_existing_path(self):
        with self.assertRaises(FileNotFoundError):
            self.transformer.transform('non_existing_path')

    def tearDown(self):
        self.temp_directory.cleanup()


class Test_List_Transformer(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.gold_dict = {'main_path': self.temp_directory.name,'result_list':['gold_list.txt']}
        self.list_transformer = cd.List_To_Dicts()

    def test_gold_input_format(self):
        with open(os.path.join(self.temp_directory.name,'gold_list.txt'),'w') as f:
            f.write('hello world!')
        output = self.list_transformer.transform(self.gold_dict)
        self.assertTrue(isinstance(output,dict))

    def test_invalid_input_path(self):
        invalid_input1 = {'main_path':'this/path/does/not/exist','result_list':['gold_list.txt']}
        with self.assertRaises(FileNotFoundError):
            self.list_transformer.transform(invalid_input1)


    def test_invalid_dict_values(self):
        invalid_input2 = {'main_path': self.temp_directory.name, 'result_list': 10}
        with self.assertRaises(TypeError):
            self.list_transformer.transform(invalid_input2)


    def tearDown(self):
        self.temp_directory.cleanup()


class Test_Dict_Transformer(unittest.TestCase):
    def setUp(self):
        self.gold_dict = {'key1':[['apple','bannana','orange','grapes'],
                                  ['Sea Salts','Pepper','Paprika','Protein Powder']],
                          }
        self.dict_transformer = cd.Dict_To_Dataset()

    def test_gold_input_format(self):
        output = self.dict_transformer.transform(self.gold_dict)['key1']
        self.assertTrue(isinstance(output,Dataset))

    def test_invalid_input_type(self):
        invalid_input = 'I like apples and banana'
        with self.assertRaises(AttributeError):
            self.dict_transformer.transform(invalid_input)


class Test_Create_Dataset(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.dir_path = self.temp_directory.name
        self.gold_path = os.path.join(self.dir_path,'gold_list.txt')

    def test_valid_file_format(self):
        example_lines = ['This is the first sentence',
                         'this is the second sentence',
                         'this is the third sentence',
                         'This is the fourth sentence',
                         ]
        with open(os.path.join(self.dir_path,'gold_list.txt'),'w') as f:
            test_file_content = f.write('\n'.join(example_lines))

        output = cd.main(self.dir_path)

        self.assertTrue(isinstance(output,dict))
        self.assertTrue(isinstance(output['gold_list'],Dataset))
