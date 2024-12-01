import re
import evaluate
# from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import MeCab


class TextNormalizer:
    def __init__(self):
        self.wakati = MeCab.Tagger("-Owakati")
        # self.normalizer = BasicTextNormalizer()
        self.FULLWIDTH_TO_HALFWIDTH = str.maketrans(
            "　０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ！゛＃＄％＆（）＊＋ー／：；〈＝〉？＠［］＾＿'｛｜｝～",
            ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&()*+-/:;<=>?@[]^_`{|}~',
        )

    def normalize(self, text: str, do_lower: bool = False) -> str:
        if do_lower:
            text = text.lower()
        text = text.translate(self.FULLWIDTH_TO_HALFWIDTH)
        # Remove Japanese punctuation
        text = re.sub(r"[、。]", "", text)
        text = re.sub(r"\s+", "", text.strip())
        return self.wakati.parse(text)

    def __call__(self, text: str, do_lower: bool = False) -> str:
        return self.normalize(text, do_lower)


class MetricsCalculator:
    def __init__(self, tokenizer, do_normalize_eval: bool = True):
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.tokenizer = tokenizer
        self.do_normalize_eval = do_normalize_eval
        self.normalizer = TextNormalizer()

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        pred.label_ids[pred.label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(
            pred.label_ids, skip_special_tokens=True
        )

        if self.do_normalize_eval:
            pred_str = [self.normalizer(pred) for pred in pred_str]
            label_str = [self.normalizer(label) for label in label_str]

        wer = 100 * self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * self.cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}
