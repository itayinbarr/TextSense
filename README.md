TextSense
==============================

A recurrent neural network model I trained, to realize whether text is positive or negative.

This project demonstrates the process of building, training and using a RNN.

Demo video
------------

### Frontend Overview

![Demo](./textsensedemo.gif?raw=true)

Getting Started
------------

I recommend storing patient images in

From within the repo directory run

`./TextSense/runner.py`

You can now type in the console any text you wish.

After pressing Return button, you will get an emotional analysis of the text.

-----
About Training & Dataset
--

The dataset was derived from Keras datasets. It consists of 25,000 labeled movie reviews.

Project Organization
------------

    ├── README.md                    <- The top-level README for developers using this project
    ├── LICENSE.md                   <- MIT
    ├── .gitignore                   <- For environment directories
    │
    ├── TextSense                    <- Containing the software itself
    │   ├── sense_model              <- Directory of trained model .gitignored
    │   ├── back.py                  <- backend code
    │   └── runner.py                <- Running the software
    │
    └── tests                        <- Tests directory, .gitignored
        └── backend_tests.py         <- Unit tests of backend
 
Dependencies
------------

- Python
- Keras
- TensorFlow
- NumPy
- IMDB
--------
