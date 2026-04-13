"""
HeceTokenizer: A Syllable-Based Tokenizer for Turkish
Author: Senol Gulgonul
GitHub: https://github.com/senolgulgonul/hecetokenizer

Standalone script — no dependencies beyond Python standard library.
"""


SESLI = set("aeıioöuü")


def _s(c):
    """Returns True if character is a consonant."""
    return c not in SESLI


def _v(c):
    """Returns True if character is a vowel."""
    return c in SESLI


def hecele(kelime):
    """
    Syllabify a single Turkish word using right-to-left greedy pattern matching.

    Six canonical syllable patterns (V=vowel, C=consonant):
        CVCC, VCC, CVC, VC, CV, V

    Isolated consonants (e.g. from loanword clusters) are handled in step g.

    Args:
        kelime (str): A single word (will be lowercased internally).

    Returns:
        list[str]: List of syllables.

    Examples:
        >>> hecele("türkiye")
        ['tür', 'ki', 'ye']
        >>> hecele("kardeş")
        ['kar', 'deş']
        >>> hecele("trabzon")
        ['t', 'rab', 'zon']
    """
    kelime = kelime.lower()
    kelime = ''.join(c for c in kelime if c.isalpha())
    if not kelime:
        return []

    heceler = []
    i = len(kelime) - 1

    while i >= 0:
        kalan = i + 1

        # a. CVCC
        if kalan >= 4 and _s(kelime[i-3]) and _v(kelime[i-2]) and _s(kelime[i-1]) and _s(kelime[i]):
            heceler.insert(0, kelime[i-3:i+1])
            i -= 4

        # b. VCC
        elif kalan >= 3 and _v(kelime[i-2]) and _s(kelime[i-1]) and _s(kelime[i]):
            heceler.insert(0, kelime[i-2:i+1])
            i -= 3

        # c. CVC
        elif kalan >= 3 and _s(kelime[i-2]) and _v(kelime[i-1]) and _s(kelime[i]):
            heceler.insert(0, kelime[i-2:i+1])
            i -= 3

        # d. VC
        elif kalan >= 2 and _v(kelime[i-1]) and _s(kelime[i]):
            heceler.insert(0, kelime[i-1:i+1])
            i -= 2

        # e. CV
        elif kalan >= 2 and _s(kelime[i-1]) and _v(kelime[i]):
            heceler.insert(0, kelime[i-1:i+1])
            i -= 2

        # f. V
        elif _v(kelime[i]):
            heceler.insert(0, kelime[i])
            i -= 1

        # g. isolated consonant (loanword clusters)
        else:
            heceler.insert(0, kelime[i])
            i -= 1

    return heceler


def metni_hecele(metin):
    """
    Syllabify a full text string.

    Args:
        metin (str): Input text.

    Returns:
        str: Syllabified text where each syllable is whitespace-delimited.

    Examples:
        >>> metni_hecele("Türkiye büyük bir ülkedir")
        'tür ki ye bü yük bir ül ke dir'
    """
    return " ".join(
        h for kelime in metin.lower().split()
        for h in hecele(kelime)
    )


def chunk_passage(passage, chunk_size=4):
    """
    Split a passage into overlapping word-based chunks with stride 1.

    Args:
        passage (str): Input passage.
        chunk_size (int): Number of words per chunk (default: 4, optimal for retrieval).

    Returns:
        list[str]: List of text chunks.
    """
    words = passage.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(len(words))]


if __name__ == "__main__":
    test_words = [
        "türkiye", "kardeş", "trabzon",
        "matematikçiler", "atasözleri", "geçmişten", "günümüze",
    ]

    print("HeceTokenizer — Syllabification Demo")
    print("=" * 40)
    for word in test_words:
        syllables = hecele(word)
        print(f"{word:20s} -> {' + '.join(syllables)}")

    print()
    text = "Atasözleri geçmişten günümüze kadar ulaşan sözlerdir"
    print(f"Input : {text}")
    print(f"Output: {metni_hecele(text)}")
