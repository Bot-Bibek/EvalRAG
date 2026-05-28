Your RAG system answers questions, but does that mean it works well? Not necessarily. Confident responses can still be wrong. 
RAGAS helps evaluate retrieval and generation quality using metrics, making it easier to identify problems and improve performance.

RAGAS evaluates your RAG system using four key metrics:
## The Four Metrics (RAGAS)

| Metric | Simple Meaning | What it checks | Inputs |
|--------|----------------|-----------------|--------|
| Faithfulness | Is the answer hallucinating? | Whether every claim is supported by retrieved context | question, answer, retrieved context |
| Answer Relevancy | Does it answer the question? | Semantic match between question and generated answer | question, answer, embeddings |
| Context Precision | Is retrieval clean? | How much retrieved content is actually useful | question, retrieved context, reference answer |
| Context Recall | Did we retrieve everything needed? | Whether all relevant info was retrieved from corpus | question, retrieved context, reference answer |

## How RAGAS Works

RAGAS evaluates your RAG system using a separate LLM called an evaluator. This model checks your outputs and gives scores based on how good they are.

For example, in Faithfulness:

1. The system sends the answer and retrieved context to the evaluator  
2. The evaluator checks if the answer is supported by the context  
3. It breaks the answer into points and verifies each one  
4. The final score is based on how many points are supported  

The evaluator is not part of your RAG pipeline. It only reviews the input and output.

In this project, we use Gemini 2.5 Flash as the evaluator because it is fast and efficient.
