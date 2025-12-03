# overall

The overall goal here is that we're trying to make a test project to train some simple probes linear probes or logistic probes on a language model. 

model unsloth/gemma-3-270m (bfloat16 is good, float16 is bad causes nan, float32 is ok but larger obviously)
dataset https://huggingface.co/datasets/manu/project_gutenberg

# data

For data we can just use Project Gutenberg books right so we've got text We'll run the text to the LLM and try to predict the word 'the' , so looking for the token ' the'.
So um the first thing we should do is have a script that downloads that database umm processes it in some way to get a balanced version so you know we need 50% samples where the probe should be one and 50% samples where the probe should be zero.
So I would also really like just a way to quickly have a look at that data and inspect it and check that it's correct.

# probe

So when you perform inference on a sequence of tokens you get a sequence of activations throughout every layer Our probes should attach to the activations of the NTH layer of the final token. Now in the case that the next token is ' the' our probe should output one and otherwise it should output zero

We're going to want two train probes that attach to the hidden layers maybe the residuals as well. We'll probably do training runs that scan through various layers like the first the last and then like 3 throughout the middle just to create groups of training runs and see how being in different places affects it.

So basically if an inference pass is given a sequence of tokens and it's extremely likely to output the token the next that's something that should be really easy for our probe to figure out if it's attached to one of the later layers. That's why this is such a good initial project for building a probe

# training (wandb)

It's really really good to have TQBN and progress bars on anything that takes time and we want all our runs to log to wand B so that we can watch them that's just really great and also we should have the GPU utilized for everything so we can train we got 16 gigs of V ram here

Also obviously it's really important that we have a validation data set that we that we test against but it's also really really valuable that we do confusion matrix we wanna know about false positives false negatives accuracy Positive accuracy negative accuracy that kind of stuff is really valuable 'cause most tokens are not the token we're looking for so it's easy to reward hack in quotes And we want to avoid that from happening

# organization

Tried to put all the data in a data folder all the checkpoints in a models folder or the 1B logs in the 1B folder stuff like that keep everything tidy so and all the scripts in its own folder and also we should umm have a document that makes it really easy to use every script explains how to do it gives a copy pastable command line

# result visualization

It's really crucial to be able to visualize the results so that we can really see if the probe is working as it should A nice way to do that is just an HTML page that lets me scrub a slider through the different layers that we have probes attaching to and then it visualizes say like a random page of Gutenberg like 4000 tokens white background on the tokens where the probe is zero red background on the probe giving that token a one score.

# extras

If you can think of any additional techniques or tools or anything you can like come up with that will help build this system as robustly as possible enable insight and understanding into how it's working absolutely throw those in that would be fantastic
