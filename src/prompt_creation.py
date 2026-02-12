from retrieval_system import InformationRetrievalSystem

class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def return_message(self):
        return {'role': self.role, 'content': self.content}

    def change_content(self, new_content):
        self.content = new_content

    def change_role(self, new_role):
        self.role = new_role


class MessageList:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def add_string_message(self, message):
        role, content = message.split(',', 1)
        new_message = Message(role, content)
        self.add_message(new_message.return_message())

    def delete_message(self, message_index):
        self.messages.remove(message_index)

    def return_message_list(self):
        return self.messages


class PromptBuilder:
    def __init__(self):
        self.message_list = MessageList()

    def add_prompt(self, prompt, dynamic_inputs, option='user'):
        option_list = {'user': 'user', 'assistant': 'assistant', 'system': 'system'}
        filled_prompt = prompt.format(*dynamic_inputs)
        self.message_list.add_string_message(f'{option_list.get(option, 'user')},{filled_prompt}')

    def delete_message(self, message_index):
        self.message_list.delete_message(message_index)

    def return_prompt(self):
        return self.message_list.return_message_list()

    def print_prompt(self):
        print(self.message_list.return_message_list())


class PromptCreationSystem:
    def __init__(self, main_prompt, example_pool, n_shots=3):
        self.main_prompt = main_prompt
        self.example_pool = InformationRetrievalSystem(example_pool, num_examples=n_shots)
        self.n_shots = n_shots

    def create_full_prompt(self, information_list, user_input_idxs, answer_idx,search_key):
        query = information_list[0]
        prompt_builder = PromptBuilder()
        examples = self.example_pool.get_examples_from_query(query,search_key)
        parsed_examples = parse_retrieval_results(examples)

        for i in range(self.n_shots):
            current_information = parsed_examples[i]
            prompt_builder.add_prompt(self.main_prompt, [current_information[i] for i in user_input_idxs], option='user')
            prompt_builder.add_prompt('Output: {}', [current_information[answer_idx]], option='assistant')
        prompt_builder.add_prompt(self.main_prompt, information_list, option='user')
        return prompt_builder.return_prompt()

    def set_new_main_prompt(self, new_prompt):
        self.main_prompt = new_prompt

def parse_retrieval_results(results, n_results=3):
    final_results = []
    for i in range(n_results):
        final_results.append([value[i] for key, value in results.items()])
    return final_results

