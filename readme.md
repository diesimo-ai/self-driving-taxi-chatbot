![GitHub repo size](https://img.shields.io/github/repo-size/diesimo-ai/self-driving-taxi-chatbot.svg) ![Packagist Stars](https://img.shields.io/github/stars/diesimo-ai/self-driving-taxi-chatbot.svg) ![Packagist forks](https://img.shields.io/github/forks/diesimo-ai/self-driving-taxi-chatbot.svg) ![GitHub](https://img.shields.io/github/license/diesimo-ai/self-driving-taxi-chatbot)
# Self-Driving Taxi Chatbot 

## Overview

This is an AI chatbot application for (future) self-driving taxis using NLP (Natural Language Processing) and deep learning.

Imagine a future where self-driving cars and robotaxis are no longer science fiction. A system like this could be used to make driving more enjoyable and safer.

Here are examples of the type of interactions a user could perform:

- Ask the car to drive from the current location to a desired destination
- During the ride, ask the car to recommend songs, videos, or news, and other kinds of entertainment
- Have the system recommend local places to visit, such as restaurants and shops
- Have the system monitor passenger comfort and safety during the ride (seat posture, safety belt, etc.)
...and many more!

`Edit`: The project was created in 2020-12-09, before LLMs and generative AI became popular, especially ChatGPT. Adding an LLM API handling module would allow direct communication between the car and LLMs. 


## Requirements

To install the required dependencies, run the script below

```sh
pip install -r requirements.txt
```

## Applications

This project can be used as baseline for In-Vehicle infotainement (IVI)  applications: 

- Futuristic cockpits
- Vehicle interior comfort
- digital cluster interaction 
- ...
## Usage 

To run the app:

```python
python3 main.py
```

## Expected Results

### > Prompting ... 

**Input data**: question/prompt from the user/passage r( strings/text format)

```
> What do you rent? ...
```

**Outputs**

```
We rent Tesla ...
```

## Contributing

Submit a PR if you want to help this project grow, or 
open an issue, if you enconter any problem running the project.

`@TODO - List:`

```
- Save the chat conversation into a logfile
- Add a logger module for debugging
- Use RNN model for large data exchanges (edit:2023 - this could be done using LLMs)
- Add GPS/Google Maps API based on sensors datas like LIDAR/RADAR to suggest activities to the users 
- New: Add an LLM API handling module would allow direct communication between the car and LLMs
- Add speech to text module handler, for voice commands
```


## References

- [1] [Contextual Chatbots with Tensorflow](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077?gi=68480999d4c7) 
- [2] [Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial (Part 1)](https://www.youtube.com/watch?v=RpWeNzfSUHw)
- [3] [Natural Language Toolkit - NLTK](https://www.nltk.org)


