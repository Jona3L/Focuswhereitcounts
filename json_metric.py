import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# For CIDEr (pycocoevalcap)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider




HYPOTHESIS_JSON = "converstaion.json"
REFERENCE_JSON  = "Dataset.json"
# ------------------------------------------------------------------


# Download NLTK resources quietly (if not present)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def compute_text_metrics(reference: str, hypothesis: str):
    """
    Compute BLEU (1-4), METEOR, and ROUGE (1,2,L) metrics 
    between a reference string and a hypothesis string.
    Returns a dictionary of scores.
    """
    # If either string is empty, return zeros
    if not reference.strip() or not hypothesis.strip():
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "meteor": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    # Tokenize
    ref_tokens = word_tokenize(reference)
    hyp_tokens = word_tokenize(hypothesis)

    # Smooth function for BLEU (helps with short sentences)
    smoothie = SmoothingFunction().method1

    # BLEU-1
    bleu1 = sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie
    )
    # BLEU-2
    bleu2 = sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothie
    )
    # BLEU-3
    bleu3 = sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(1/3, 1/3, 1/3, 0),
        smoothing_function=smoothie
    )
    # BLEU-4
    bleu4 = sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    # METEOR: pass list of tokens
    meteor = meteor_score([ref_tokens], hyp_tokens)

    # ROUGE: pass strings (rejoin tokens)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens)
    rouge_scores = scorer.score(ref_str, hyp_str)

    rouge1 = rouge_scores["rouge1"].fmeasure
    rouge2 = rouge_scores["rouge2"].fmeasure
    rougeL = rouge_scores["rougeL"].fmeasure

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL
    }


def extract_for_strings(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    id_to_text = {}
    for item in data:
        image_id = item["id"]
        gpt_text = None
        for turn in item.get("conversations", []):
            if turn["from"].lower() == "gpt":
                gpt_text = turn["value"]
                break
        if gpt_text is not None:
            id_to_text[image_id] = gpt_text

    return id_to_text


def extract_for_coco_eval(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    coco_dict = {}
    for item in data:
        image_id = item["id"]
        gpt_text = None
        for turn in item.get("conversations", []):
            if turn["from"].lower() == "gpt":
                gpt_text = turn["value"]
                break

        if gpt_text is not None:
            coco_dict[image_id] = [{"caption": gpt_text}]

    return coco_dict


def main():

    ref_dict = extract_for_strings(REFERENCE_JSON)
    hyp_dict = extract_for_strings(HYPOTHESIS_JSON)


    coco_references = extract_for_coco_eval(REFERENCE_JSON)
    coco_hypotheses = extract_for_coco_eval(HYPOTHESIS_JSON)

    common_ids = set(ref_dict.keys()) & set(hyp_dict.keys())
    if not common_ids:
        print("No matching IDs in the two JSON files.")
        return


    sum_bleu1 = sum_bleu2 = sum_bleu3 = sum_bleu4 = 0.0
    sum_meteor = 0.0
    sum_rouge1 = sum_rouge2 = sum_rougeL = 0.0
    count = 0

    for image_id in common_ids:
        reference_text = ref_dict[image_id]
        hypothesis_text = hyp_dict[image_id]

        scores = compute_text_metrics(reference_text, hypothesis_text)
        sum_bleu1   += scores["bleu1"]
        sum_bleu2   += scores["bleu2"]
        sum_bleu3   += scores["bleu3"]
        sum_bleu4   += scores["bleu4"]
        sum_meteor  += scores["meteor"]
        sum_rouge1  += scores["rouge1"]
        sum_rouge2  += scores["rouge2"]
        sum_rougeL  += scores["rougeL"]
        count       += 1

    if count > 0:
        avg_bleu1   = sum_bleu1 / count
        avg_bleu2   = sum_bleu2 / count
        avg_bleu3   = sum_bleu3 / count
        avg_bleu4   = sum_bleu4 / count
        avg_meteor  = sum_meteor / count
        avg_rouge1  = sum_rouge1 / count
        avg_rouge2  = sum_rouge2 / count
        avg_rougeL  = sum_rougeL / count
    else:
        avg_bleu1   = avg_bleu2 = avg_bleu3 = avg_bleu4 = 0
        avg_meteor  = avg_rouge1 = avg_rouge2 = avg_rougeL = 0


    tokenizer = PTBTokenizer()
    tokenized_refs = tokenizer.tokenize(coco_references)
    tokenized_hyps = tokenizer.tokenize(coco_hypotheses)

    # Then compute CIDEr
    cider_scorer = Cider()
    cider_score, cider_scores_per_image = cider_scorer.compute_score(
        tokenized_refs, tokenized_hyps
    )

    if len(cider_scores_per_image) > 0:
        cider_score_avg = sum(cider_scores_per_image) / len(cider_scores_per_image)
    else:
        cider_score_avg = 0.0


    print("\n===== EVALUATION RESULTS ({} samples) =====".format(count))
    print(f"BLEU-1 : {avg_bleu1:.4f}")
    print(f"BLEU-2 : {avg_bleu2:.4f}")
    print(f"BLEU-3 : {avg_bleu3:.4f}")
    print(f"BLEU-4 : {avg_bleu4:.4f}")
    print(f"METEOR : {avg_meteor:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"CIDEr  : {cider_score:.4f} (corpus-level)")
    print(f"CIDEr  : {cider_score_avg:.4f} (mean of individual scores)")


if __name__ == "__main__":
    main()
