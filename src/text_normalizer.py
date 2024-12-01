import re
import MeCab
# from transformers.models.whisper.english_normalizer import BasicTextNormalizer


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
