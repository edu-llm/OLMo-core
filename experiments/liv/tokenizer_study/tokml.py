import json, unicodedata, collections, sys
p="/scratch/users/ericrcwu/liv/tok/lfm2_tokenizer.json"
d=json.load(open(p)); m=d["model"]; vocab=m["vocab"]
at={t["id"] for t in d.get("added_tokens",[])}
atc={t["content"] for t in d.get("added_tokens",[])}

# byte-level (GPT-2 style) unicode<->byte map
def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("\u00a1"),ord("\u00ac")+1))+list(range(ord("\u00ae"),ord("\u00ff")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n+=1
    return dict(zip(bs,[chr(c) for c in cs]))
b2u=bytes_to_unicode(); u2b={v:k for k,v in b2u.items()}

def decode(tok):
    try:
        return bytes([u2b[c] for c in tok]).decode("utf-8", errors="strict")
    except KeyError:
        return None   # not pure byte-level
    except UnicodeDecodeError:
        return "\x00PARTIAL"

def script(ch):
    o=ord(ch)
    if o<128: return "ASCII"
    try: n=unicodedata.name(ch)
    except ValueError: return "UNNAMED"
    for pre,lab in [("CJK","CJK"),("HIRAGANA","Japanese"),("KATAKANA","Japanese"),
                    ("HANGUL","Korean"),("CYRILLIC","Cyrillic"),("ARABIC","Arabic"),
                    ("HEBREW","Hebrew"),("DEVANAGARI","Devanagari"),("GREEK","Greek"),
                    ("THAI","Thai"),("BENGALI","Bengali"),("TAMIL","Tamil"),
                    ("TELUGU","Telugu"),("ARMENIAN","Armenian"),("GEORGIAN","Georgian"),
                    ("ETHIOPIC","Ethiopic"),("KHMER","Khmer"),("MYANMAR","Myanmar"),
                    ("LAO","Lao"),("SINHALA","Sinhala"),("GUJARATI","Gujarati"),
                    ("GURMUKHI","Gurmukhi"),("KANNADA","Kannada"),("MALAYALAM","Malayalam"),
                    ("ORIYA","Oriya"),("TIBETAN","Tibetan"),("MONGOLIAN","Mongolian"),
                    ("SYRIAC","Syriac"),("THAANA","Thaana"),("CHEROKEE","Cherokee")]:
        if n.startswith(pre): return lab
    if n.startswith("LATIN"): return "Latin-accented"
    cat=unicodedata.category(ch)
    if cat.startswith("S"): return "Symbol/Emoji"
    if cat.startswith("P"): return "Punct"
    if cat.startswith("Z") or cat.startswith("C"): return "Space/Ctrl"
    return "Other-"+n.split()[0]

n_total=len(vocab)
n_ascii=0; n_nonascii=0; n_partial=0; n_notbytelevel=0; n_special=0
scr=collections.Counter(); examples=collections.defaultdict(list)
alnum_ascii_word=0
for tok,i in vocab.items():
    if tok in atc:
        n_special+=1; continue
    s=decode(tok)
    if s is None: n_notbytelevel+=1; continue
    if s=="\x00PARTIAL": n_partial+=1; continue
    labs={script(c) for c in s}
    non={l for l in labs if l not in ("ASCII",)}
    if not non:
        n_ascii+=1
    else:
        n_nonascii+=1
        # dominant non-ascii script
        cnt=collections.Counter(script(c) for c in s if ord(c)>=128)
        lab=cnt.most_common(1)[0][0]
        scr[lab]+=1
        if len(examples[lab])<6: examples[lab].append(s)

print("=== TOTALS (model.vocab entries) ===")
print("len(model.vocab)          =", n_total)
print("  special/added (in vocab)=", n_special)
print("  pure-ASCII text tokens  =", n_ascii)
print("  non-ASCII text tokens   =", n_nonascii)
print("  partial-UTF8 byte frags =", n_partial)
print("  not byte-level decodable=", n_notbytelevel)
denom = n_total - n_special
print()
print("NON-ASCII fraction of NON-SPECIAL vocab = %d/%d = %.2f%%" % (n_nonascii, denom, 100*n_nonascii/denom))
print("NON-ASCII fraction of nominal 65536     = %.2f%%" % (100*n_nonascii/65536))
print("(partial byte fragments counted separately: %d = %.2f%% of non-special)" % (n_partial, 100*n_partial/denom))
print()
print("=== script breakdown (non-ASCII tokens) ===")
for k,v in scr.most_common(40):
    print("  %-16s %6d  %5.2f%% of vocab   ex: %s" % (k, v, 100*v/denom, examples[k][:5]))
