from prompt_creation import PromptCreationSystem
import transformers

class PromptChain:
    def __init__(self, model_id, prompt_1, prompt_2, carryover_positions, answer_idxs, example_pool, n_examples=2):
        self.model = transformers.pipeline('text-generation', model=model_id)
        self.prompt_1 = prompt_1
        self.prompt_2 = prompt_2
        self.carryover_idxs = [int(i) for i in carryover_positions.split(',')]
        self.answer_idx = [int(i) for i in answer_idxs.split(',')]
        self.prompt_creator = PromptCreationSystem(prompt_1, example_pool, n_shots=n_examples)

    def use_chain_to_generate(self, information_list, search_key):
        prompt_1_input_idxs = self.carryover_idxs
        full_prompt_1 = self.prompt_creator.create_full_prompt(information_list, prompt_1_input_idxs,
                                                               self.answer_idx[0], search_key)
        initial_first_output = self.model(full_prompt_1, max_new_tokens=50, do_sample=False)
        print('Here\'s my first output!!!')
        print(initial_first_output[0]['generated_text'][-1]['content'])

        self.prompt_creator.set_new_main_prompt(self.prompt_2)
        information_list.append(initial_first_output[0]['generated_text'][-1]['content'])
        prompt_2_input_idxs = self.carryover_idxs + [self.answer_idx[0]]
        full_prompt_2 = self.prompt_creator.create_full_prompt(information_list, prompt_2_input_idxs,
                                                               self.answer_idx[1], search_key)
        final_output = self.model(full_prompt_2, max_new_tokens=50, do_sample=False)
        return {'prompt1': initial_first_output,
                'prompt2': final_output}


