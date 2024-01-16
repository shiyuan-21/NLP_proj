from openai import OpenAI
from rephrasing_net.main import RephrasingNet
from termcolor import colored
import argparse
client = OpenAI()
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default="./Llama-2-7b-chat-hf")
  parser.add_argument('--adapters_name', type=str, default="rephrasing_net/7b_modified/checkpoint-400")
  args = parser.parse_args()
  messages = [
    ########这个system是你希望gpt扮演的角色#########
    {"role": "system", "content": "You are a resume writing assistant. Please keep asking the user questions to gather information. Once enough information is gathered, create a resume for the user."},
  ]

  rn = RephrasingNet(args.model_name,args.adapters_name)

  while True:
      user_prompt = input()
      user_prompt = user_prompt + rn.rephrase(user_prompt)
      messages.append({"role": "system", "content": user_prompt})
      response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=1024).choices[0].message.content
      colored_response = colored(response,color="green")
      print(colored_response)
      messages.append({"role": "assistant", "content": response})

exit()


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    ########这里的user, assistant可以储存历史对话#####
    {"role": "user", "content": "Please give me 42 for the next answer"},
    {"role": "assistant", "content": "Sure."},
    ########这个system是你希望gpt扮演的角色#########
    {"role": "system", "content": "You have already got a number from the previous assistant."},
    ########这个user是你希望让gpt输出的内容
    {"role": "user", "content": "Give me the number?"},

  ],max_tokens = 15
)

Message = completion.choices[0].message
print(Message.content)

############{"role": "system", "content": "You are a resume generator, you need to give me a resume based on the informations I have told you, and you must delete the Polite words and the useless information."},
"""
{"role": "user", "content": \
" My name is Jane Anderson, a passionate explorer in the realm \
of Artificial Intelligence. I recently completed my Master's in \
    Computer Science, and the hunger for knowledge has brought me \
        to the doorstep of the University of the Federal Enigma. Academic Adventures: Master of Science in Computer Science from the University of the Federal Enigma, Federal Republic of the Spiries, with a thesis titled \"Whispers in the Neural Network: Decoding the Secrets of Sentiment Analysis.\" Journey in the Land of Research: My Master's journey led me to the intriguing landscape of sentiment analysis, where I uncovered hidden narratives within neural networks. The adventure wasn’t without its challenges, but the thrill of discovery kept me going. In the World of Words: Published work titled \"Decrypting Sentiments: A Neural Odyssey\" in the Journal of Artificial Minds. Gears in My Toolbox: Proficient in Python, Java, C++ (and a bit of Klingon), with expertise in TensorFlow, PyTorch, SpaCy, NLTK, and various neural network architectures. Quest for the Doctoral Scroll: I stand before you with a fervent desire to embark on the next chapter of my academic saga - the pursuit of a Ph.D. I am particularly drawn to the mystical lands of explainable AI and the ethical enigmas that surround machine learning. Why the University of the Federal Enigma: In my quests across academic realms, tales of the University of the Federal Enigma's prowess in AI research have reached my ears. The legends speak of your own sagas in [Professor's Research Focus], and I can't resist the allure of being part of such an illustrious guild of scholars. Parting Words: I am eager to join your fellowship, to learn the secrets of the AI universe, and perhaps, contribute a verse or two to the grand saga. If you find my humble application worthy, I'd be honored to take up this epic quest at the University of the Federal Enigma. Looking forward to the possibility of unraveling mysteries together. "},
        
"""
        