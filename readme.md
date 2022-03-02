# Driveless Taxi Chatbot 

## Overview

This is an AI chatbot application for (future) self-driving taxis using NLP (Natural Language Processing ) and Deep Learning.
The main goal of the app is to interact with the client/passenger after he gets in the car.
- It asks the passenger to enter the final destination adress and runs it
- During the traffic it recommends songs, places to visit such as: restaurant, shop and so on
- It alerts/monitors passenger comfort and the safety during the driving (seat posture, safety belt (OPS)...)
- The passenger can share its device connection with the car(blutooth...) and start interating(chating) with the car ask whatever he wants, and available.  

## Requirements tools  
- Python 3.7
- NLTK 
- PyTorch 

## Libraries

Install the required libaries by running the script below 
```bash
./run_setup.sh
```

## Usage 

To run the app :

```bash
./run_app.sh
```

## Input sinals
- Questions from the users (Strings/text format)

## Output sinals
- Answers based on a AI/ML+DL model (Strings/text format)
- log files

## If you want to Colaborate you are welcome and your fresh ideas too!!!
@TODO
- Save the chat conversation into a logfile
- Add a log file for debugging
- Use RNN model for large data exchanges
- Add GPS/Google Maps API based on sensors datas like LIDAR/RADAR to suggest activities to the users 


## References

This app get its inspiration from : 

- https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077?gi=68480999d4c7 
- https://www.youtube.com/watch?v=RpWeNzfSUHw
- https://www.nltk.org
