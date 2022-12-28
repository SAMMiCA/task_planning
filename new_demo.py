import json
import random
import numpy as np
import torch
from sentence_transformers import util as st_utils
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed, BertForNextSentencePrediction, BertTokenizer


# helper function for finding similar sentence in a corpus given a query
# def find_most_similar(query_str, corpus_embedding):
#     query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
#     # calculate cosine similarity against each candidate sentence in the corpus
#     cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
#     # retrieve high-ranked index and similarity score
#     most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
#     return most_similar_idx, matching_score
# def set_random_seed():
#     return random.randint(1, 9999)
# seed = set_random_seed()
# print(seed)
# set_seed(111)
seed = 190
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# translation_lm_id = 'stsb-roberta-base'
# translation_lm = SentenceTransformer(translation_lm_id).to(device)
# with open('./src/available_actions.json', 'r') as f:
#     action_list = json.load(f)
# action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

generator = pipeline('text-generation', model='./distilgpt_results', tokenizer='distilgpt2')
nsp_check = BertForNextSentencePrediction.from_pretrained('./nsp_results').eval()


def test_with_testset(test_path):
    acc_list = []
    count = 0
    correct = 0
    total = 0
    with open("result.txt", 'a') as tf:
        tf.write("seed: {}".format(str(seed)))
        tf.write("\n")
        with open(test_path) as f:
            lines = f.readlines()
        for k, data in enumerate(lines[0].split("Task")[1:]):
            if len(data.split("Step")) > 10:
                continue
            prompt = "Task: pick cup place cabinet Step 1: navigate to cup Step 2: pick up the cup Step 3: navigate to cabinet Step 4: put cup to cabinet "
            # prompt = ""
            task = 'Task' + data.split('Step')[0][:-1]
            tf.write(task)
            tf.write("\n")
            tf.write("\n")
            # obj = task.split(" ")[-3]
            # recep = task.split(" ")[-1]
            prompt += task + ' Step 1:'
            print("{}".format(task))
            max_len = 64
            for i, step in enumerate(data.split("Step")[1:]):
                if i == 0:
                    # sen_A = prompt.split("Step")[i][:-1]
                    sen_A = "Task" + prompt.split("Task")[-1].split("Step")[0][:-1]
                else:
                    # sen_A = prompt.split("Step")[i].split(":")[1][1:-1]
                    sen_A = prompt.split("Step")[i][:-1]
                result = generator(prompt, max_length=max_len, num_return_sequences=5)#[0]['generated_text']
                logits = []
                next_sen = []
                for j, options in enumerate(result):
                    options = options["generated_text"]
                    # sen_B = options.split("Step")[i+1].split(":")[-1][1:-1]
                    sen_B = ''
                    if len(options.split("Step")[i + 5].split(":")) > 2:
                        for w in options.split("Step")[i + 5].split(":")[1].split(" ")[1:-1]:
                            sen_B += w + " "
                    else:
                        sen_B = options.split("Step")[i + 5].split(":")[-1][1:-1]
                    sen_B = sen_B.strip()
                    next_sen.append(sen_B)
                    encoding = bert_tokenizer(sen_A, sen_B, return_tensors="pt")
                    output = nsp_check(**encoding)
                    logits.append(output.logits[0][0])     # index 0: sequence B is a continuation of sequence A
                max_arg = logits.index(max(logits))     # index 1: sequence B is a random sequence
                prompt += " " + next_sen[max_arg] + " Step {}:".format(i+2)
                # most_similar_idx, matching_score = find_most_similar(next_sen[max_arg][1:-1], action_list_embedding)
                # translated_action = action_list[most_similar_idx]
                # print(next_sen[max_arg])
                # print(translated_action)
                gt = step.split(":")[1][1:-1]
                print("ground truth: {}".format(gt))
                print("predicted: {}".format(next_sen[max_arg]))
                tf.write(gt)
                tf.write("\n")
                tf.write(next_sen[max_arg])
                tf.write("\n")
                tf.write("\n")
                if gt == next_sen[max_arg]:
                    correct += 1
                total += 1
                if i >= 1:
                    max_len = 80
            tf.write("\n\n\n")
                # print("translated: {}".format(translated_action))
                # if gt == translated_action:
                #     correct += 1
        #     acc = correct/(i+1)
        #     acc_list.append(acc)
        #     count +=1
        #     if count == 5:
        #         break
        # print('Accuracy: {}'.format(sum(acc_list)/len(acc_list)))
        print("correct: {}".format(str(correct)))
        print("total: {}".format(str(total)))
        print('Accuracy: {}'.format(str(correct/total)))
        tf.write("correct: {}".format(str(correct)))
        tf.write("\n")
        tf.write("total: {}".format(str(total)))
        tf.write("\n")
        tf.write('Accuracy: {}'.format(str(correct / total)))
        tf.write("\n")




# def arbitary_test():
#     prompt = 'Task: Pick up phone Step 1: Walk to home office Step 2: Walk to phone Step 3: Find phone Step 4: Grab phone '
#     new_info = 'Now the phone is in the bathroom '
#     prompt += new_info
#     task = 'Task: Pick up phone'
#     prompt += task + " Step 1:"
#     ct = 0
#     sen_A = task
#     while True:
#         result = generator(prompt, max_length=128, num_return_sequences=5)  # [0]['generated_text']
#         logits = []
#         next_sen = []
#         for i, options in enumerate(result):
#             options = options["generated_text"]
#             sen_B = options.split("Task")[2].split("Step")[ct + 1].split(":")[-1]
#             next_sen.append(sen_B)
#             encoding = bert_tokenizer(sen_A, sen_B[1:-1], return_tensors="pt")
#             output = nsp_check(**encoding)
#             logits.append(output.logits[0][0])  # index 0: sequence B is a continuation of sequence A
#         max_arg = logits.index(max(logits))  # index 1: sequence B is a random sequence
#         prompt += next_sen[max_arg] + "Step {}:".format(ct + 2)
#         sen_A = next_sen[max_arg][1:-1]
#         # most_similar_idx, matching_score = find_most_similar(next_sen[max_arg][1:-1], action_list_embedding)
#         # translated_action = action_list[most_similar_idx]
#         # print(next_sen[max_arg])
#         # print(translated_action)
#         print("predicted: {}".format(next_sen[max_arg]))
#         ct += 1
#         if ct == 10:
#             break

test_with_testset('test_dataset_c.txt')
# arbitary_test()


