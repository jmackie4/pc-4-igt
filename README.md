This is the main code that I'm using for an experiment on utilizing the prompt chaining technique on creating interlinear glossed texts. There's a few main parts that build off of each other a bit.

Create Datasets - Takes a directory path and creates huggingface dataset objects for each file. The processing is based on the files in the SIGMORPHON 2023 shared task!

Retrieval System - Takes a dataset and allows for someone to input a query string to get relevant examples based on number of items in the intersection between the query and the items in the specific dataset
column

Prompt Creation - Creates a full prompt for an autocausal model on HuggingFace. Honestly, this works for any chat model that takes a list of messages (dictionaries). I created this with TinyLlama model in mind so yea.

Chain Prompts - This allows you to load in two prompts, as well as what parts of the items in the dataset should be used for the user prompt as well as which ones should be used for the assistant's answer. That's assuming you're doing in-context learning.

Oh yea, you'll also need to load in your dataset with the examples you want to use, and you'll need to give the huggingface model id of whatever chat model you want to use.

I still have a lot of things to clean up and a bunch of tests to write for the actual program, but the code here does work if you meet the following requirements:
1. You need to be using the SIGMORPHON 2023 IGT shared task data!
2. You need to establish yourself which dataset will serve as your pool of examples for the information retrieval system
3. You'll also need to define your own prompts to use for the prompt chain class. Just note that the first prompt should have enough template variables to fit the number of items in the input list,
but the second prompt should have one more template variable since it'll use the original input list and add the LLM's generated output before going to the next prompt creator
4. In addition, in order to use the Prompt Chain item, you'll need to define what columns of the dataset can be use in user input with a string of numbers separated by commas. You'll need to do the same thing
for the answers index too!
5. Oh yea, you'll also need to use your own HuggingFace token so this code doesn't run super duper slowly!

Again, this is super messy and needs to be cleaned! But I'm gonna be working on that so hopefully this placeholder readme won't be up for too long! If you do decide to use this version, let me know if you're having problems
always glad to help ya out with any problems! 
