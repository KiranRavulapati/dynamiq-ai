import json
import logging

from dynamiq.components.evaluators.llm_evaluator import LLMEvaluator
from dynamiq.nodes.llms import BaseLLM, OpenAI

# Configure logging
logger = logging.getLogger(__name__)


class AnswerCorrectnessEvaluator:
    def __init__(self, llm: BaseLLM, weights: list[float] = [0.75, 0.25]):
        self.llm = llm
        self.weights = weights
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        # Initialize the LLMEvaluators and store them as class attributes

        # Extract Statements Evaluator
        extract_instructions = (
            "For each input text, extract the key statements.\n"
            "- Keep statements concise and focused.\n"
            "- Return the statements as a JSON array of strings.\n"
            "- Ensure that your response is valid JSON, using double quotes for all strings."
        )
        self.statement_extractor = LLMEvaluator(
            instructions=extract_instructions.strip(),
            inputs=[("texts", list[str])],
            outputs=["statements"],
            examples=[
                {
                    "inputs": {"texts": ["The sun is powered by nuclear fusion. It provides heat and light."]},
                    "outputs": {
                        "statements": [["The sun is powered by nuclear fusion.", "It provides heat and light."]]
                    },
                },
            ],
            llm=self.llm,
        )

        # Classify Statements Evaluator
        classify_instructions = (
            'Given a "Question", "Answer Statements", and "Ground Truth Statements", '
            "classify the statements:\n"
            '- "TP" (True Positive): Statements present in both Answer and Ground Truth.\n'
            '- "FP" (False Positive): Statements in Answer but not in Ground Truth.\n'
            '- "FN" (False Negative): Statements in Ground Truth but not in Answer.\n'
            "Each statement should only belong to one category.\n"
            'Provide the classifications as a JSON object with keys "TP", "FP", "FN", '
            "and values as lists of statements.\n"
            "Ensure that your response is valid JSON, using double quotes for all strings."
        )
        self.statement_classifier = LLMEvaluator(
            instructions=classify_instructions.strip(),
            inputs=[
                ("question", list[str]),
                ("answer_statements", list[list[str]]),
                ("ground_truth_statements", list[list[str]]),
            ],
            outputs=["classifications"],
            examples=[
                {
                    "inputs": {
                        "question": ["What powers the sun and what is its primary function?"],
                        "answer_statements": [
                            [
                                "The sun is powered by nuclear fission.",
                                "The sun's primary function is to provide light to " "the solar system.",
                            ]
                        ],
                        "ground_truth_statements": [
                            [
                                "The sun is powered by nuclear fusion.",
                                "The sun provides heat and light essential for life " "on Earth.",
                            ]
                        ],
                    },
                    "outputs": {
                        "classifications": {
                            "TP": ["The sun's primary function is to provide light " "to the solar system."],
                            "FP": ["The sun is powered by nuclear fission."],
                            "FN": [
                                "The sun is powered by nuclear fusion.",
                                "The sun provides heat and light essential for " "life on Earth.",
                            ],
                        }
                    },
                },
            ],
            llm=self.llm,
        )

        # Compute Similarity Evaluator
        similarity_instructions = (
            'For each pair of "Answer" and "Ground Truth", evaluate their semantic similarity.\n'
            "- Score the similarity from 0 to 1.\n"
            "- Use 1 if the Answer is semantically identical to the Ground Truth.\n"
            "- Use 0 if the Answer is completely dissimilar to the Ground Truth.\n"
            "- Provide the similarity score as a single number between 0 and 1.\n"
            "Ensure that your response is valid JSON, using double quotes for all strings."
        )
        self.similarity_evaluator = LLMEvaluator(
            instructions=similarity_instructions.strip(),
            inputs=[("answers", list[str]), ("ground_truths", list[str])],
            outputs=["similarity_score"],
            examples=[
                {
                    "inputs": {
                        "answers": ["Paris is the capital of France."],
                        "ground_truths": ["The capital of France is Paris."],
                    },
                    "outputs": {"similarity_score": 1},
                },
                {
                    "inputs": {
                        "answers": ["Berlin is the capital of Germany."],
                        "ground_truths": ["The capital of France is Paris."],
                    },
                    "outputs": {"similarity_score": 0},
                },
            ],
            llm=self.llm,
        )

    def extract_statements(self, texts: list[str]) -> list[list[str]]:
        results = self.statement_extractor.run(texts=texts)
        statements_list = [result["statements"] for result in results["results"]]
        return statements_list

    def classify_statements(
        self,
        questions: list[str],
        answer_statements_list: list[list[str]],
        ground_truth_statements_list: list[list[str]],
    ) -> list[dict[str, list[str]]]:
        results = self.statement_classifier.run(
            question=questions,
            answer_statements=answer_statements_list,
            ground_truth_statements=ground_truth_statements_list,
        )
        classifications_list = [result["classifications"] for result in results["results"]]
        return classifications_list

    def compute_similarity_scores(self, answers: list[str], ground_truths: list[str]) -> list[float]:
        results = self.similarity_evaluator.run(answers=answers, ground_truths=ground_truths)
        similarity_scores = [float(result["similarity_score"]) for result in results["results"]]
        return similarity_scores

    @staticmethod
    def compute_f1_score(classifications: dict[str, list[str]]) -> float:
        tp = len(classifications.get("TP", []))
        fp = len(classifications.get("FP", []))
        fn = len(classifications.get("FN", []))
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0  # No statements to compare.
        precision_denom = tp + fp
        recall_denom = tp + fn
        precision = tp / precision_denom if precision_denom > 0 else 1.0
        recall = tp / recall_denom if recall_denom > 0 else 1.0
        if (precision + recall) == 0.0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        ground_truths: list[str],
        verbose: bool = False,
    ) -> list[float]:
        if not (len(questions) == len(answers) == len(ground_truths)):
            raise ValueError("Questions, answers, and ground truths must have the same length.")

        # Extract statements
        answer_statements_list = self.extract_statements(answers)
        ground_truth_statements_list = self.extract_statements(ground_truths)

        # Classify statements
        classifications_list = self.classify_statements(questions, answer_statements_list, ground_truth_statements_list)

        # Compute F1 scores
        f1_scores = [self.compute_f1_score(classifications) for classifications in classifications_list]

        # Compute similarity scores
        similarity_scores = self.compute_similarity_scores(answers, ground_truths)

        # Combine scores
        final_scores = []
        for i in range(len(questions)):
            f1_score = f1_scores[i]
            sim_score = similarity_scores[i]
            final_score = self.weights[0] * f1_score + self.weights[1] * sim_score
            final_scores.append(final_score)

            if verbose:
                logger.debug(f"Question: {questions[i]}")
                logger.debug(f"Answer: {answers[i]}")
                logger.debug(f"Ground Truth: {ground_truths[i]}")
                logger.debug("Classifications:")
                logger.debug(json.dumps(classifications_list[i], indent=2))
                logger.debug(f"F1 Score: {f1_score}")
                logger.debug(f"Similarity Score: {sim_score}")
                logger.debug(f"Final Score: {final_score}")
                logger.debug("-" * 50)
        return final_scores


# Example usage
if __name__ == "__main__":
    import sys

    from dotenv import find_dotenv, load_dotenv

    # Load environment variables for OpenAI API
    load_dotenv(find_dotenv())

    # Configure logging level (set to DEBUG to see verbose output)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # Uncomment the following line to enable verbose logging
    # logging.getLogger().setLevel(logging.DEBUG)

    # Initialize the LLM (replace 'gpt-4' with your available model)
    llm = OpenAI(model="gpt-4")

    # Sample data (can be replaced with your data)
    questions = [
        "What powers the sun and what is its primary function?",
        "What is the boiling point of water?",
    ]
    answers = [
        (
            "The sun is powered by nuclear fission, similar to nuclear reactors on Earth."
            " Its primary function is to provide light to the solar system."
        ),
        "The boiling point of water is 100 degrees Celsius at sea level.",
    ]
    ground_truths = [
        (
            "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium."
            " This fusion process releases a tremendous amount of energy. The sun provides"
            " heat and light, which are essential for life on Earth."
        ),
        (
            "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at"
            " sea level. The boiling point can change with altitude."
        ),
    ]

    # Initialize evaluator and evaluate
    evaluator = AnswerCorrectnessEvaluator(llm)
    # Set verbose=True to enable detailed logging
    correctness_scores = evaluator.evaluate(
        questions, answers, ground_truths, verbose=False  # Set verbose=True to enable logging
    )

    # Print the results
    for idx, score in enumerate(correctness_scores):
        print(f"Question: {questions[idx]}")
        print(f"Answer Correctness Score: {score}")
        print("-" * 50)

    print("Answer Correctness Scores:")
    print(correctness_scores)