import html
import re
import string
import unicodedata
import urllib.parse
from typing import Literal

import hazm
import nltk
import six
import unicodeblock.sequence
from lxml import etree
from parsel import Selector
from tqdm import tqdm

from .tools import MapRed, defaultname, replace_fromstart


def init_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")  # For tokenization
    nltk.download("wordnet")  # For lemmatization
    nltk.download("omw-1.4")  # For wordnet lemmatizer support
    nltk.download("punkt_tab")


def get_wordnet_pos(self, word):
    from nltk.corpus import wordnet

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


class Preprocessor:
    def __init__(
        self,
        normalize_fa_chars=True,
        remove_special_chars=True,
        remove_cjk=True,
        strip_accents=True,
        remove_ctrl_chars=True,
        remove_fa_stopwords=True,
        remove_en_stopwords=True,
        remove_punctuation=True,
        only_eng_unicode_range=False,
        only_fa_unicode_range=False,
        only_num_unicode_range=False,
        do_fa_normalizer=False,
        do_fa_stemmer=False,
        do_fa_lemmatizer=False,
        do_en_normalizer=False,
        do_en_stemmer=False,
        do_en_lemmatizer=False,
        fa_tokenizer=False,
        en_tokenizer=False,
        fa_sentence_tokenizer=False,
        en_sentence_tokenizer=False,
        en_lemmatize_method: Literal["all", "verbs", "known"] = "all",
        remove_parentheticals=False,
        remove_singular_digis=False,
        do_hard_limit=False,
        high_hard_limit=26,
        low_hard_limit=3,
        parse_html=False,
        remove_uri=False,
        remove_www=False,
        split_group_annots=False,
    ):
        """Preprocessor from aggregated tools.
        Functions included:
            - sentence_preprocessor
            - fa_sent_normalizer
            - en_sent_normalizer
            - parse_html
            - html_to_text
            - product_title_preprocessor
        Functions to be included:
            - sentence_normalizer
            - custom_en_stop_words
            - calculated_stop_words
            - custom keep alphabet
            - input type json to text
        Functions half included:
            - word_post_process (we need to remove low freq words using tf-idf)
            - char_post_process (we need to remove low freq chars using mapred)
        """

        init_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        self.normalize_fa_chars = normalize_fa_chars
        self.remove_special_chars = remove_special_chars
        self.remove_cjk = remove_cjk
        self.strip_accents = strip_accents
        self.remove_ctrl_chars = remove_ctrl_chars
        self.remove_fa_stopwords = remove_fa_stopwords
        self.remove_en_stopwords = remove_en_stopwords
        self.remove_punctuation = remove_punctuation
        self.only_eng_unicode_range = only_eng_unicode_range
        self.only_fa_unicode_range = only_fa_unicode_range
        self.only_num_unicode_range = only_num_unicode_range
        self.do_fa_normalizer = do_fa_normalizer
        self.do_fa_stemmer = do_fa_stemmer
        self.do_fa_lemmatizer = do_fa_lemmatizer
        self.do_en_normalizer = do_en_normalizer
        self.do_en_stemmer = do_en_stemmer
        self.do_en_lemmatizer = do_en_lemmatizer
        self.fa_tokenizer = fa_tokenizer
        self.en_tokenizer = en_tokenizer
        self.fa_sentence_tokenizer = fa_sentence_tokenizer
        self.en_sentence_tokenizer = en_sentence_tokenizer
        self.en_lemmatize_method = en_lemmatize_method
        self.remove_parentheticals = remove_parentheticals
        self.remove_singular_digis = remove_singular_digis
        self.do_hard_limit = do_hard_limit
        self.high_hard_limit = high_hard_limit
        self.low_hard_limit = low_hard_limit
        self.parse_html = parse_html
        self.remove_uri = remove_uri
        self.remove_www = remove_www
        self.split_group_annots = split_group_annots

        if self.remove_fa_stopwords:
            self.fa_stopwords = set(hazm.stopwords_list())
            # TODO: calculated_stop_words
        if self.remove_en_stopwords:
            self.en_stopwords = set(stopwords.words("english"))
            # self.en_stopwords = [s for s in self.en_stopwords if len(s) > 2]
            # TODO: custom_en_stop_words

        # TODO: custom keep alphabet
        # Filter out characters not in ok_alphabet
        # text = "".join([t for t in text if t in ok_alphabet])

        if self.do_fa_normalizer:
            # !!! Changes en digit to fa digit !!!
            fa_normalizers = hazm.Normalizer()
            self.fa_normalizer = fa_normalizers.normalize
        if self.do_fa_stemmer:
            fa_stemmer = hazm.Stemmer()
            self.fa_stemmer = fa_stemmer.stem
        if self.do_fa_lemmatizer:
            fa_lemmatizer = hazm.Lemmatizer()
            self.fa_lemmatizer = fa_lemmatizer.lemmatize

        if self.do_en_stemmer:
            en_stemmer = PorterStemmer()
            self.en_stemmer = en_stemmer.stem
        if self.do_en_lemmatizer:
            en_lemmatizer = WordNetLemmatizer()
            self.en_lemmatizer = en_lemmatizer.lemmatize

        # Splits based on family (en, fa, numeric)
        self.fa_word_tokenizer = WordTokenizer(self.fa_tokenizer)
        self.en_word_tokenizer = WordTokenizer(self.en_tokenizer)
        self.fa_sent_tokenizer = SentTokenizer(self.fa_sentence_tokenizer)
        self.en_sent_tokenizer = SentTokenizer(self.en_sentence_tokenizer)

        self.ok_unicode_ranges = []
        if only_eng_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.english_letters_ranges
            )
        if only_fa_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.persian_letters_ranges
            )
        if only_num_unicode_range:
            self.ok_unicode_ranges = (
                self.ok_unicode_ranges + UnicodeRanges.numerical_digits_ranges
            )

    def en_normalizer(self, s: str):
        return s.lower()

    def parenthetical_remover(self, s: str):
        """Remove text inside parantheses or brackets"""
        return re.sub(r"\([^)]*\)|\[[^\]]*\]", "", s)

    def singular_digits_remover(self, tokens: list[str]):
        isnum = re.compile("^\d+$")
        return [t for t in tokens if not re.match(isnum, t)]

    def hard_limiter(self, tokens: list[str], low: int, high: int):
        """Keep words longer and shorter than [a, b] chars"""
        return [t for t in tokens if (len(t) >= low) and len(t) <= high]

    def uri_remover(self, s: str):
        return re.sub("( )?http(s)?://[^\s]+( )?", " ", s).strip()

    def www_remover(self, s: str):
        return re.sub("( )?www\.[^\s]+\.[^\s]( )?", " ", s).strip()

    def group_annots_splitter(self, s: str):
        return re.sub("/|,", " ", s).strip()

    def __call__(self, s: str, keep_lf=False) -> str:
        # TODO: Check with "from parsel import Selector" if type of input is json (double quotes escaped).
        # and `loads` it .

        if not s:
            return ""

        if self.parse_html:
            s = html_to_text(s)
        if self.remove_uri:
            s = self.uri_remover(s)
        if self.remove_www:
            s = self.www_remover(s)
        if self.split_group_annots:
            s = self.group_annots_splitter(s)

        if self.do_en_normalizer == True:
            s = self.en_normalizer(s)

        # Replace with equivalent characters
        if self.normalize_fa_chars == True:
            for mapping in Chars.get_mappings():
                for x, y in mapping:
                    s = s.replace(x, y)

        # Remove trash chars
        if self.remove_special_chars:
            s = re.sub(Chars.get_burn_chars(), " ", s)
        if self.remove_cjk:
            s = re.sub(Chars.get_cjk(), " ", s)
            s = Unicode.sanitize_char_sets(s)
        if self.strip_accents:
            s = Unicode.strip_accents(s)
        if self.remove_ctrl_chars:
            s = Unicode.sanitize_cmp(s)
        if self.remove_punctuation:
            s = "".join([c for c in s if c not in string.punctuation])

        if self.ok_unicode_ranges:
            ur_regex = UnicodeRanges.regex_from_range(
                self.ok_unicode_ranges, negate=True
            )
            s = re.sub(ur_regex, "", s)

        if self.remove_parentheticals:
            s = self.parenthetical_remover(s)

        # Farsi
        sentenecs = self.fa_sent_tokenizer(s)
        procd_sentenecs = []
        for sent in sentenecs:
            words = self.fa_word_tokenizer(sent)

            if self.remove_fa_stopwords:
                words = [w for w in words if w not in self.fa_stopwords]
            if self.do_fa_normalizer:
                # !!! Changes en digit to fa digit !!!
                words = [self.fa_normalizer(w) for w in words]
            if self.do_fa_stemmer:
                words = [self.fa_stemmer(w) for w in words]
            if self.do_fa_lemmatizer:
                words = [self.fa_lemmatizer(w) for w in words]

            # words = [w for w in words if " " not in w]

            sent = " ".join(words)
            procd_sentenecs.append(sent)

        s = "\n".join(procd_sentenecs)

        # English + general
        sentences = self.en_sent_tokenizer(s)
        procd_sentences = []
        for sent in sentences:
            words = self.en_word_tokenizer(sent)

            if self.remove_singular_digis:
                words = self.singular_digits_remover(words)
            if self.do_hard_limit:
                words = self.hard_limiter(
                    words, self.low_hard_limit, self.high_hard_limit
                )

            # TODO: Per word English normalizer?
            if self.remove_en_stopwords:
                words = [w for w in words if w not in self.en_stopwords]
            if self.do_en_stemmer:
                words = [self.en_stemmer(w) for w in words]

            if self.do_en_lemmatizer:
                # Lemmatize
                if self.en_lemmatize_method == "all":
                    words = [self.en_lemmatizer(w) for w in words]

                # Lemmatize with part-of-speech (POS) tags
                # 'v' for verbs
                if self.en_lemmatize_method == "verbs":
                    words = [self.en_lemmatizer(word, pos="v") for word in words]

                # Lemmatize with Wordnet
                if self.en_lemmatize_method == "known":
                    words = [
                        self.en_lemmatizer(w, get_wordnet_pos(w))
                        for w in words
                        if self.en_lemmatizer(w, get_wordnet_pos(w)) != ""
                    ]

            # words = [w for w in words if " " not in w]

            sent = " ".join(words)
            procd_sentences.append(sent)

        s = "\n".join(procd_sentences)

        # Remove extra spaces
        if keep_lf:
            s = re.sub("[\t\r\n\v\f]", " ", s)
            s = re.sub("[\t\r\n\v\f][\t\r\n\v\f]+", " ", s)
        else:
            # s = re.sub("\s", " ", s)
            # s = re.sub("\s\s+", " ", s)
            s = " ".join(s.split())

        # Removed chars leave spaces
        s = s.strip()

        return s


class WordTokenizer:
    def __init__(self, fa_tokenizer=False, en_tokenizer=False):
        if fa_tokenizer ^ en_tokenizer:
            # TODO: Can also use both tokenizers
            self.word_tokenizer = lambda s: s.split(" ")
        elif fa_tokenizer:
            # Splits based on family (en, fa, numeric):
            self.word_tokenizer = hazm.word_tokenize
        else:
            self.word_tokenizer = nltk.word_tokenize

    def __call__(self, s: str) -> str:
        return self.word_tokenizer(s)


class SentTokenizer:
    def __init__(self, fa_sentence_tokenizer=False, en_sentence_tokenizer=False):
        if fa_sentence_tokenizer ^ en_sentence_tokenizer:
            # TODO: Can also use both tokenizers
            self.sent_tokenizer = lambda s: s.split("\n")
        elif fa_sentence_tokenizer:
            self.sent_tokenizer = hazm.sent_tokenize
        else:
            self.sent_tokenizer = nltk.sent_tokenize

    def __call__(self, s: str) -> str:
        return self.sent_tokenizer(s)


class MySelector(Selector):
    def get(self):
        """
        My serialize and return the matched nodes in a single unicode string.
        Percent encoded content is unquoted.
        """
        try:
            t = etree.tostring(
                self.root,
                method=self._tostring_method,
                encoding="unicode",
                with_tail=False,
            )
            t = urllib.parse.unquote(t)
            t = html.unescape(t)
            t = unicodedata.normalize("NFKD", t)
            return t
        except (AttributeError, TypeError):
            if self.root is True:
                return "1"
            elif self.root is False:
                return "0"
            else:
                t = six.text_type(self.root)
                t = urllib.parse.unquote(t)
                t = html.unescape(t)
                t = unicodedata.normalize("NFKD", t)
                return t


class MyResponse:
    def __init__(self, url, body) -> None:
        self.url = url
        self.body = body
        self.css = lambda sel: MySelector(body.decode("utf-8")).css(sel)
        self.xpath = lambda sel: MySelector(body.decode("utf-8")).xpath(sel)


def html_to_text(t: str):
    """Html or text to text.
    Parse does not fail if not html."""

    # if isinstance(t, int):
    #     return t
    # css method fails if input is all numbers with this weird error:
    # ValueError: Cannot use css on a Selector of type 'json'
    # if t.numeric():
    # return t
    if t == "null":
        return t
    try:
        float(t)
        return t
    except ValueError:
        pass

    selector = MySelector(t)
    parsed = selector.css("::text").getall()
    parsed = " ".join(parsed)
    return parsed


class UnicodeRanges:
    """
    References:
        https://en.wikipedia.org/wiki/Arabic_script_in_Unicode
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
    """

    english_letters = r"\u0041-\u005A\u0061-\u007A"
    persian_letters = r"\u0600-\u06FF\u0750-\u077F"
    numerical_digits = r"\u0030-\u0039"
    special_chars = "!@#$%^&*()_+{{}}\[\]:;\"'<>,.?/|~`×\u200c=.-]"

    # A-Z and a-z
    english_letters_ranges = [(0x0041, 0x005B), (0x0061, 0x007B)]
    # Arabic and Persian ranges
    persian_letters_ranges = [(0x0600, 0x0700), (0x0750, 0x0780)]
    # 0-9:;<=>?
    numerical_digits_ranges = [(0x0030, 0x0040)]
    specials = [(0x0032, 0x0047), (0x0058, 0x0064), (0x0091, 0x0096), (0x0123, 0x0126)]

    cjk_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        # (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
        # (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
        # (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
        # (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
        # (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
        # (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        # (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    ]

    @classmethod
    def regex_from_range(self, ranges: list[tuple[int, int]], negate=False):
        hex_to_unicode = lambda x: rf"\u{format(x, '04X')}"
        ranges = [(hex_to_unicode(s), hex_to_unicode(e)) for s, e in ranges]
        ranges = [f"{s}-{e}" for s, e in ranges]
        ranges = "".join(ranges)
        if negate == False:
            pattern = rf"[{ranges}]"
        else:
            pattern = rf"[^ {ranges}]"  # Added space
        return re.compile(pattern)

    @classmethod
    def unknown_chars(self):
        return rf"[^{self.english_letters}{self.persian_letters}{self.numerical_digits}{self.special_chars}"

    @classmethod
    def regex_cjk_chars(self):
        return self.regex_from_range(self.cjk_ranges)

    @classmethod
    def print_unknown_chars(self, words: list):
        regex = re.compile(self.unknown_chars())

        for word in tqdm(words):
            if bool(regex.search(word)):
                print(word)

    @classmethod
    def print_unicode_range(self, ranges: list[tuple[int, int]]):
        for start, end in ranges:
            for codepoint in range(start, end):
                print(chr(codepoint), end="")
            print()  # New line after each range

    @classmethod
    def print_known_ranges(self):
        print("English Letters:")
        self.print_unicode_range(self.english_letters_ranges)

        print("\nPersian Letters:")
        self.print_unicode_range(self.persian_letters_ranges)

        print("\nNumerical Digits:")
        self.print_unicode_range(self.numerical_digits_ranges)

        print("\nSpecial chars:")
        self.print_unicode_range(self.specials)


class Unicode:
    # https://www.fileformat.info/info/unicode/category/index.htm
    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string

    def strip_accents(text: str):
        """Strip accents from text (e.g., é -> e)."""

        return "".join(c for c in unicodedata.normalize("NFD", text))

    def sanitize_cmp(text: str):
        """Strip CMP categories from text:
        Other (Control, Format, Not Assigned, Private Use, Surrogate),
        Mark (Spacing Combining, Enclosing, Nonspacing),
        Punctuation (Connector, Dash, Close, Final quote, Initial quote, Other, Open).

        We keep LNSZ (Letter, Number, Symbol, Separator). Separator is handled with regex \s.
        """

        # Other (C) removes LF
        # Punctuation (P) removes dot

        return "".join(
            (
                c
                if unicodedata.category(c)[0] not in ["C", "M", "P"] or c in ["\n", "."]
                else " "
            )
            for c in text
        )

    # Remove trash character sets
    def sanitize_char_sets(text: str):
        return "".join(
            c if unicodeblock.blocks.of(c) not in ["HANGUL_JAMO", "CYRILLIC"] else " "
            for c in text
        )


class Chars:
    # Equivalent EXTENDED ARABIC-INDIC DIGIT
    mappings_extended_arin_en_digit = [
        ("۰", "0"),
        ("۱", "1"),
        ("۲", "2"),
        ("۳", "3"),
        ("۴", "4"),
        ("۵", "5"),
        ("۶", "6"),
        ("۷", "7"),
        ("۸", "8"),
        ("۹", "9"),
    ]

    # Equivalent ARABIC-INDIC DIGIT
    mappings_arin_en_digit = [
        ("٠", "0"),
        ("١", "1"),
        ("٢", "2"),
        ("٣", "3"),
        ("٤", "4"),
        ("٥", "5"),
        ("٦", "6"),
        ("٧", "7"),
        ("٨", "8"),
        ("٩", "9"),
    ]

    # Equivalent letters
    mappings_ar_fa_letters = [
        ("ە", "ه"),  # ARABIC LETTER AE, ARABIC LETTER HEH
        ("ہ", "ه"),  # ARABIC LETTER HEH GOAL, ARABIC LETTER HEH
        ("ٸ", "ی"),  # ARABIC LETTER HIGH HAMZA YEH, ARABIC LETTER FARSI YEH
        ("ھ", "ه"),  # ARABIC LETTER HEH DOACHASHMEE, ARABIC LETTER HEH
        ("ى", "ی"),  # ARABIC LETTER ALEF MAKSURA, ARABIC LETTER FARSI YEH
        ("ں", "ن"),  # ARABIC LETTER NOON GHUNNA, ARABIC LETTER NOON
        ("ے", "ی"),  # ARABIC LETTER YEH BARREE, ARABIC LETTER FARSI YEH
        ("ﯼ", "ی"),  # ARABIC LETTER FARSI YEH ISOLATED FORM # NEW!
        ("آ", "ا"),  # A kolah-dar
        ("ي", "ی"),  # ARABIC LETTER YEH
    ]

    # Equivalent special characters
    mappings_ar_ = [
        ("؛", ";"),
        ("٪", "%"),
        ("²", "2"),
        ("³", "3"),
    ]

    burn_chars_a = [
        "\u200e",  # RLM
        "\u200f",  # LRM
        "\u200c",  # ZWNJ
        "\u00ad",  # SOFT HYPHEN [SHY]
        "\u2026",  # HORIZONTAL ELLIPSIS (re.sub doesn't pick up … so used hex value)
        "®",
        "©",
        "™",
        "(",
        ")",
        "/",
        ",",
        ":",
        "[",
        "]",
        "«",
        "»",
        "<",
        ">",
        "'",
        '"',
        "#",
        "*",
        "،",  # ARABIC COMMA
        "-",
        "|",
    ]

    def get_mappings():
        classvars = list(filter(lambda a: a.startswith("mappings_"), vars(Chars)))
        mappings = []
        for cv in classvars:
            mappings.append(getattr(Chars, cv))
        return mappings

    def get_burn_chars():
        classvars = list(filter(lambda a: a.startswith("burn_"), vars(Chars)))
        burn_chars = []
        for cv in classvars:
            burn_chars += getattr(Chars, cv)
        burn_chars = "".join(re.escape(c) for c in burn_chars)
        return f"[{burn_chars}]"

    def get_cjk():
        return re.compile("[\u4e00-\u9fff]")
