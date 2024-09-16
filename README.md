# EdgeRunner AI Assignment

## Overview

The proposed approach involves of the following processes:
1. Evaluating the pretrained models on GSM8K using lm-eval-harness
2. Fine-tuning the models using ORPO
3. Implementing a debate-style evaluation where the two models models collaboratively solve GSM8K problems.

I used a compute instance with 2xV100 GPUs in AzureML Studio while preparing this assignment. For the last (longer) run that I submitted, I used a compute instance with 4xv100 GPUs.

To run from scratch, simply run debate_model.ipynb. It will run the initial evaluations, fine-tuning, and proceed to solve the GSM8K with the proposed debate/collaboration methodology. An HF token to upload those models to HF.
In order to skip the initial evaluations and finetuning, simply load the libraries at the beginning of the notebook, and then run the last two cells.

The results of the debate model (the questions, the discussion between the two models and the results) are available in file result.txt. Any new run will append new data to this file.

The visualization of the training process during fine-tuning is available at the following wandb link:
https://api.wandb.ai/links/neural_scientist-aristotle-univesity-of-thessaloniki/upnc51sx 
Fine-tuned models and their adapters are included in this repo, and are also available at HF hub: https://huggingface.co/huggingscientist10 

Requirements.txt includes the basic libraries used.

(Note: This repo will become private after being reviewed)

Notes:

### Fine-tuning

- ORPO was chosen for fine-tuning due to memory constraints when attempting to use DPO since it requires a reference model.
- I Used the orpo-dpo-mix-40k dataset, focusing on mathematics-related content.
- Fine-tuning was performed on a subset of the data for 1 epoch, again due to resource limitations.

### Model Debate Evaluation

- I designed a debate-style process where models discuss the given maths problems.
- Models review and improve upon each other's responses until converging on a solution.
- Since this approach required manual design, using lm-eval-harness was not applicable, so accuracy was measured manually, by processing the final response string to extract the resulting scalar value and comparing it with the ground truth result.
.

## Implementation Notes

- Due to memory limitations, Llama was loaded and used as a quantized model with 4-bit precision at all steps of the process, in order to fit in the available memory.
- However, it was not possible to do the same with EdgeRunner, since after experimenting to load the model in 4bits I realized that it would cause its generated responses to be completely incoherent (mostly sequences of symbols). Fortunately, the model could fit in the compute instance (requiring freeing up all the memory frequently), but for the initial evaluation using lm-eval-harness, the 4bit precision model version was used, thus accuracy in that test might be inaccurate.
- Results from fine-tuning are not apparent when looking at the training metrics, which is natural since training was very short, so the chosen answers and rejected answers fluctuate in a similar fashion, so the margins remain relatively constant so far. One could argue that performance degrades, since rejected answers become chosen slightly more than before. For this reason, I decided to use the original models (not the finetuned) for the debate process. The idea of first optimizing the pretrained models on maths datasets, and then applying the debate scheme on those would be more promising given enough resources.
- For the debate process, the idea is to, instead of optimizing the models separately in order to improve their individual performance at GSM8K, combine the models in a collaborative manner in order to solve the questions more accurately by having the models discuss and review their answers and eventually improve them and converge to an accurate solution within a specified number of rounds.
- The discussion prompt includes special tokens (e.g. <|end_header_id|>). This is the only way that I could stop Llama from generating infinite text (until max_new_tokens threshold was hit). I tried to troubleshoot in several other ways but adding the special tokens was the only way that I could do it successfully. Researching the issue showed that this is not uncommon, but the provided solutions did not work for me (modifying pad token, config, updating libraries, ...). In some of the cases during the debate, Llama would still not stop generating text.
- There is a manual evaluation performed, since the current setup was more difficult to implement to make use of lm-eval-harness. The evaluation method extracts the final answer that the models converged to (as a scalar value), and compares it to the ground-truth value. If they are exactly the same, then the model it counts as a correct solution, otherwise it counts as wrong.


## Some observations:

- Sometimes the models would have a correct step-by-step thinking, and the computations would seem correct, but the result of the computations would be off by a very small amount. For example, although the model would end up calculating the difference "200000-130000", it would output the result "69999" instead of 70000. This would affect accuracy of course, even though both reasoning and calculations was correct, but maybe it has to do with rounding errors due to the 4bit precision?
- With some discussion prompts, there was a chance that the models would derail the conversation by assuming that they have to generate the whole discussion (i.e. both sides of the discussion) instead of waiting for the input, or proceed to generate new questions themselves. This would result conversations that would affect the outcome.
- Debate process sometimes led to correction of errors, improving overall accuracy.
- Memory constraints limited the scope of fine-tuning and model loading.

## As for the end results:

- The debate-style approach achieved an accuracy of 60-70% on a limited subset (restricted by resources), as it can also be seen from the corresponding plot in the notebook. Since evaluation was manual, it is computed on few samples only (computational constraints did not allow for longer evaluation). Also, the computed accuracy is not comparable to the evaluation with lm-eval-harness, since it is not as sophisticated. However, I was not expecting to achieve higher accuracy with this proof-of-concept method.
- There are almost no cases as far as I have seen (only one that I observed) where one model discards a correct answer presented by the other, and provides a wrong answer instead. On the contrary, there are cases where one model corrects the other and the correct answer is reached (e.g. results for questions Q2, Q8, Q28, almost Q13 since the error was corrected but one more wrong step was added in the end)
- This proof-of-concept method shows potential for improvement with refinement.
- A few-shot learning methodology could improve performance in the debate process (e.g. allowing the model to remember previous questions, discussions and results).

Other things I attempted to do was implement a simple MCTS-based method, or a traditional Reinforcement Learning technique, but it required far too much effort for the purposes and scope of this assignment.
