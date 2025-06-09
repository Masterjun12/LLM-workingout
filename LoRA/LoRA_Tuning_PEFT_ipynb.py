#!/usr/bin/env python
# coding: utf-8

# # LoRA 튜닝
# 
# 이 노트북에서는 PEFT 라이브러리를 사용하여 사전 학습된 모델에 LoRA 튜닝을 적용하는 방법을 소개합니다.
# 
# ## LoRA 튜닝 간단 소개
# 
# LoRA(Low-Rank Adaptation)는 재매개변수화 기법입니다. 큰 가중치 행렬을 두 개의 작은 행렬로 분해하여 학습 시 이 작은 행렬만 업데이트합니다. 이 과정을 통해 전체 파라미터 수를 줄이면서도 성능을 유지하거나 향상시킬 수 있습니다.
# 
# 기존 가중치는 고정(freeze)하고, 새롭게 추가된 저랭크 행렬만 학습되므로 메모리 효율적이며 학습 속도도 빠릅니다.
# 

# ![Matrices_LoRA.webp](data:image/webp;base64,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)

# ## LoRA의 작동 방식
# 
# 예를 들어, 50x50 크기의 원래 가중치 행렬은 총 2,500개의 파라미터를 가지고 있습니다. 하지만 LoRA에서는 이 행렬을 두 개의 작은 행렬 — (2x50)과 (50x2) — 로 분해하여 근사합니다. 이 두 행렬은 각각 100개의 파라미터만 가지므로, 총 200개의 파라미터만 학습하면 됩니다. 이는 약 92%의 파라미터 수 절감을 의미하며, 원래 행렬이 클수록 절감 비율은 더 커집니다.
# 
# GPT-3와 같은 대형 언어 모델에서는 전체 파라미터의 약 0.02%만 학습하면 되는 경우도 있습니다. 놀라운 점은 이렇게 극단적으로 적은 수의 파라미터만 학습해도 전체 모델을 전부 파인튜닝한 것과 유사한 성능을 얻을 수 있다는 것입니다. 경우에 따라서는 성능이 더 나은 경우도 있습니다.
# 

# ## PEFT 및 Datasets 라이브러리 불러오기
# 
# PEFT 라이브러리는 Hugging Face에서 제공하는 다양한 파인튜닝 기법(Lora 등)을 포함하고 있습니다.
# 
# Datasets 라이브러리는 다양한 공개 데이터셋에 손쉽게 접근할 수 있도록 도와줍니다.
# 

# In[28]:


get_ipython().system('pip install -q peft==0.8.2')
get_ipython().system('pip install -q datasets==2.16.1')


# ## 모델 및 토크나이저 불러오기
# 
# `transformers` 라이브러리에서 모델과 토크나이저를 불러오는 데 필요한 클래스들을 가져옵니다.
# 
# 이 예제에서는 Bloom 모델을 사용합니다. Bloom은 PEFT 라이브러리에서 Prompt Tuning을 적용할 수 있는 작은 크기의 효율적인 모델 중 하나입니다. 학습 시간과 Colab 환경의 메모리 한계를 고려하여 가장 작은 모델을 사용합니다.
# 
# Bloom 계열의 다양한 모델을 사용해보고 성능 차이를 비교해보는 것도 좋은 방법입니다.
# 

# In[50]:


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloomz-560m"
#model_name="bigscience/bloom-1b1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
foundation_model = AutoModelForCausalLM.from_pretrained(model_name)


# ## 사전학습된 모델로 추론하기
# 
# 먼저 파인튜닝 없이 사전학습된(pre-trained) 모델을 사용해 간단한 추론을 수행합니다.  
# 이후 파인튜닝을 진행한 후 결과가 어떻게 달라지는지 비교할 수 있습니다.
# 

# In[51]:


# 이 함수는 입력과 모델을 받아 모델의 출력을 반환합니다.
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,       # 생성할 최대 토큰 수
        repetition_penalty=1.5,              # 반복되는 문장 생성을 방지
        early_stopping=True,                 # 최대 길이에 도달하지 않아도 종료 가능
        eos_token_id=tokenizer.eos_token_id  # 종료 토큰 ID 지정
    )
    return outputs


# ## 파인튜닝에 사용되는 데이터셋
# 
# 이번 파인튜닝에 사용되는 데이터셋은 대형 언어 모델(LLM)에 입력할 프롬프트들을 포함하고 있습니다.
# 
# 사전학습된 모델을 대상으로 "동기부여 코치"처럼 행동하도록 요청해보겠습니다.
# 

# In[52]:


#Inference original model
input_sentences = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundation_model, input_sentences, max_new_tokens=200)

print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))


# ## 출력 결과 확인
# 
# 모델의 출력이 정확한지는 확실하지 않지만, 적어도 우리가 기대한 "프롬프트 형태"는 아닙니다.  
# 따라서, 모델이 프롬프트 엔지니어처럼 동작하도록 하려면 파인튜닝이 필요합니다.
# 

# ## 데이터셋 준비
# 
# 사용할 데이터셋은 다음과 같습니다:  
# [awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)
# 
# 이 데이터셋은 ChatGPT에 입력할 수 있는 다양한 프롬프트들을 포함하고 있어, 모델을 프롬프트 엔지니어처럼 학습시키기에 적합합니다.
# 

# In[53]:


from datasets import load_dataset
dataset = "fka/awesome-chatgpt-prompts"

# 프롬프트 생성에 사용할 데이터셋 불러오기
data = load_dataset(dataset)

# 데이터셋의 "prompt" 컬럼에 토크나이저 적용 (배치 처리)
data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

# 학습 샘플로 데이터셋의 처음 50개 선택
train_sample = data["train"].select(range(50))

# 불필요한 'act' 컬럼 제거
train_sample = train_sample.remove_columns('act')

# 데이터셋 확인
display(train_sample)


# In[54]:


print(train_sample[:1])


# ## 데이터셋 설명 및 출력 형태
# 
# - 이 데이터셋은 ChatGPT와 같은 대형 언어 모델에 입력할 다양한 **프롬프트(prompt)** 들을 포함하고 있습니다.
# - 각 데이터 샘플은 사용자가 모델에 제공할 텍스트 프롬프트와, 이를 토크나이즈(tokenize)한 결과를 함께 가지고 있습니다.
# 
# ### 출력 형태 예시
# - `prompt`: 원본 텍스트 프롬프트 (문장 형태)
# - `input_ids`: 프롬프트를 토크나이저가 숫자 토큰 ID로 변환한 리스트
# - `attention_mask`: 입력 시퀀스에서 실제 토큰과 패딩 토큰을 구분하기 위한 마스크 (1은 실제 토큰, 0은 패딩)
# 

# ## 파인튜닝 시작
# 
# 먼저 LoRA 설정(config)을 생성하는 것이 필요합니다.
# 

# 
# 
# 
# 

# In[55]:


# TARGET_MODULES
# https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220

import peft
from peft import LoraConfig, get_peft_model, PeftModel

lora_config = LoraConfig(
    r=4,                     # R 값이 클수록 학습할 파라미터가 많아짐
    lora_alpha=1,            # 가중치 행렬의 크기를 조절하는 스케일링 팩터, 보통 1로 설정
    target_modules=["query_key_value"],  # LoRA를 적용할 모듈 목록 (위 URL에서 확인 가능)
    lora_dropout=0.05,       # 과적합 방지를 위한 드롭아웃 비율
    bias="lora_only",        # bias 파라미터를 학습할지 여부 지정
    task_type="CAUSAL_LM"    # 작업 유형: 인과적 언어 모델링
)


# ## LoRA 주요 파라미터 설명
# 
# - **r**  
#   학습할 파라미터 수를 결정하는 가장 중요한 값입니다.  
#   값이 클수록 더 많은 파라미터가 학습되며, 이는 모델이 입력과 출력 간의 더 복잡한 관계를 학습할 수 있음을 의미합니다.
# 
# - **target_modules**  
#   LoRA를 적용할 모델 내 모듈 목록입니다.  
#   [Hugging Face 문서](https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220)에서 확인할 수 있습니다.
# 
# - **lora_dropout**  
#   일반적인 드롭아웃과 동일하게 과적합을 방지하기 위해 사용됩니다.
# 
# - **bias**  
#   보통 텍스트 분류(task)에서는 `"none"`을, 챗봇이나 질문답변과 같은 작업에는 `"all"` 또는 `"lora_only"`를 사용합니다.  
#   저는 `"lora_only"`를 선택했습니다.
# 
# - **task_type**  
#   모델이 학습될 작업 유형을 지정합니다. 이번 예제에서는 텍스트 생성(`"CAUSAL_LM"`)입니다.
# 
# ---
# 
# | Task Type                      | 설명                                                                  |
# | ------------------------------ | ------------------------------------------------------------------- |
# | `TaskType.CAUSAL_LM`           | GPT와 같은 **Auto-regressive Language Modeling** (예: 텍스트 생성)           |
# | `TaskType.SEQ_2_SEQ_LM`        | T5, BART와 같은 **Sequence-to-Sequence Language Modeling** (예: 번역, 요약) |
# | `TaskType.TOKEN_CLS`           | **Token Classification** (예: 개체명 인식 NER)                            |
# | `TaskType.SEQ_CLS`             | **Sequence Classification** (예: 감성 분석, 문장 분류)                       |
# | `TaskType.MULTI_LABEL_CLS`     | **Multi-label Classification** (하나의 입력에 대해 여러 개의 라벨 분류)             |
# | `TaskType.QUESTION_ANSWERING`  | **질문-응답 시스템** (예: SQuAD 스타일)                                        |
# | `TaskType.FILL_MASK`           | **Masked Language Modeling** (예: BERT 스타일 마스크 채우기)                  |
# | `TaskType.CAUSAL_LM_STREAMING` | **Streaming 텍스트 생성** (스트리밍 환경에서 Causal LM)                          |
# | `TaskType.SPAN_ANNOTATION`     | **스팬 기반 주석** (예: BIO 태깅 등 NER 변형)                                   |
# 
# ---
# 
# ## PEFT 모델 생성하기
# 

# In[56]:


peft_model = get_peft_model(foundation_model, lora_config)
print(peft_model.print_trainable_parameters())


# 학습 가능한 파라미터 수는 사전학습된 모델의 전체 파라미터 수에 비해 매우 적습니다.
# 

# In[57]:


#Create a directory to contain the Model
import os
working_dir = './'

output_directory = os.path.join(working_dir, "peft_lab_outputs")


# `TrainingArguments`에서는 학습할 에포크 수, 출력 디렉터리, 학습률(learning rate) 등을 설정합니다.

# In[59]:


# TrainingArguments 생성하기
import transformers
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=output_directory,      # 모델과 체크포인트를 저장할 경로
    auto_find_batch_size=True,        # 데이터 크기에 맞는 적절한 배치 크기 자동 탐색
    learning_rate=3e-2,               # 전체 파인튜닝보다 높은 학습률 설정
    num_train_epochs=3,               # 학습할 에포크 수
    use_cpu=False                      # CPU 사용 설정 (GPU가 없을 경우)
)


# ## 모델 학습 준비
# 
# 모델을 학습하려면 다음이 필요합니다:
# 
# - PEFT 모델
# - `training_args` (학습 설정)
# - 학습에 사용할 데이터셋
# - `DataCollator`의 결과물 (데이터셋을 블록 단위로 처리할 수 있도록 준비된 형태)
# 

# In[60]:


trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_sample,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
trainer.train()


# In[61]:


#Save the model.
peft_model_path = os.path.join(output_directory, f"lora_model")

trainer.model.save_pretrained(peft_model_path)


# In[62]:


#Load the Model.
loaded_model = PeftModel.from_pretrained(foundation_model,
                                        peft_model_path,
                                        is_trainable=False)


# ## 파인튜닝된 모델로 추론하기

# In[65]:


import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델과 입력 모두 동일한 디바이스로 이동
loaded_model.to(device)
input_sentences = tokenizer("I want you to act as a motivational coach.", return_tensors="pt").to(device)
foundational_outputs_sentence = get_outputs(loaded_model, input_sentences, max_new_tokens=300)

print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True))


# ## 결과 비교
# 
# 
# - **사전학습 모델:**  
#   *I want you to act as a motivational coach.* Don't be afraid of being challenged.
# 
# - **파인튜닝 모델:**  
#   I want you to act as a motivational coach. I will provide some information about someone's motivation and goals, but it should be your job in order my first request – "I need someone who can help me find the best way for myself stay motivated when competing against others." My suggestion is “I have...
# 
# 보시다시피, 파인튜닝한 모델의 결과는 학습에 사용된 데이터셋 샘플과 매우 유사합니다.  
# 그리고 우리는 단 10 에포크, 아주 적은 데이터만으로 학습을 진행했습니다.
# 
# ---
# 
# ## 계속 학습해보기
# 
# 노트북 내 모든 변수들을 자유롭게 바꾸며 다양한 실험을 해보시고, 자신만의 결론을 도출해보세요.
# 
# 특히 **lora_config** 값들을 조절해보면 더 적은 에포크 수로도 더 좋은 결과를 얻어  
# 시간과 비용을 절약할 수 있을지도 모릅니다. :-)
# 

# In[ ]:




