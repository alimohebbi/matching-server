# Semantic Matching Server

## Overview

The Semantic Matching Server is designed to score target candidates based on their semantic similarity to a given GUI event in an Android application. The server supports 337 different semantic matching configurations, allowing users to customize the scoring process based on various parameters.

## Semantic Matching Configuration

A semantic matching configuration consists of four essential components:

1. **Word Embedding Technique:**
   - Options: wm, w2v, nnlm, use, bert, glove, fast, jaccard, edit_distance, random

2. **Corpus of Document for Training a Word Embedding Model:**
   - Options: android, blogs, standard, empty, googleplay

3. **Event Descriptors:**
   - Options: union, intersection, craftdroid, atm

4. **Algorithm to Aggregate Similarity Scores of Attributes:**
   - Options: craftdroid, custom, atm_0, adaptdroid_0

For detailed information on these components, please refer to our paper: [“Semantic matching of GUI events for test reuse: are we there yet?”](https://dl.acm.org/doi/abs/10.1145/3460319.3464827)

## Example Request and Response

```json
{
  "candidates": {
    // Candidate 1 details...
    // Candidate 2 details...
  },
  "sourceEvent": {
    // Source event details...
  },
  "sourceLabels": {
    // Source Labels 1 details...
    // Source Labels  2 details...
  },
  "targetLabels": {
    // Target Labels  1 details...
    // Target Labels 2 details...
  },
  "smConfig": "{\"algorithm\": \"custom\", \"word_embedding\": \"edit_distance\", \"descriptors\": \"union\", \"training_set\": \"empty\", \"app_pair\": \"craftdroid\"}"
}

{"1": 0.65, "2": 0.5416666667}
```

> For concrete examples please look into [input_sample.txt](server/input_sample.txt) and [input_sample_minimal.txt](server/input_sample_minimal.txt)
## Usage

1. Clone the repository.
2. Install the required packages from `requirements.txt` file.
3. Run the server using the provided script.

```bash
./run_server.sh
```
