# QA

Can machines comprehend and answer simple questions about a given text. Getting the answer right doesn't necessarily mean that
the meaning was actually understood but this is a good step forward.

### Problem Statement
Given a context paragraph p and a set of questions q related to the paragraph can we obtain a set of answers a such that
answers a are accurate within certain bounds.


### Disclaimer
A lot of code has been taken from bi-attn-flow model and cs224d course with the following modifications

 - It works with tensorflow 1.4.0
 - The character embed layer is removed for now
 - Instead of using context vector of the form [batch_size, sentences, words, hidden_dim] I use [batch_size, words, hidden_dim]

NOTE: This is WIP
