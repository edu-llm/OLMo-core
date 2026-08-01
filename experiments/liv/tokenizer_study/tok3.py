import json
from tokenizers import Tokenizer
p="/scratch/users/ericrcwu/liv/tok/lfm2_tokenizer.json"
tk=Tokenizer.from_file(p)
print("get_vocab_size(with_added_tokens=True) =", tk.get_vocab_size(True))
print("get_vocab_size(with_added_tokens=False)=", tk.get_vocab_size(False))
V=tk.get_vocab(True)
print("max id =", max(V.values()), " distinct ids =", len(set(V.values())))
print("config vocab_size = 65536 -> UNUSED EMBEDDING ROWS =", 65536-tk.get_vocab_size(True))
print()
print("=== DIGIT BEHAVIOR (decisive for passkey/phonebook) ===")
for s in ["1234567890","The passkey is 84721.","Call 415-555-0132","9","99","999","9999","99999",
          " 1234", "0.0001", "3.14159265358979", "1000000"]:
    e=tk.encode(s, add_special_tokens=False)
    print("  %-24r -> %2d toks %s" % (s, len(e.ids), [tk.decode([i]) if i>500 else tk.id_to_token(i) for i in e.ids]))
print()
print("=== FERTILITY on English prose (bytes/token, chars/token) ===")
samples={
 "wiki-ish":"The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy.",
 "code-py":"def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint([fibonacci(i) for i in range(20)])\n",
 "math":"Let $f(x) = \\int_0^x e^{-t^2} dt$. Then $f'(x) = e^{-x^2}$ and $\\lim_{x\\to\\infty} f(x) = \\frac{\\sqrt{\\pi}}{2}$.",
 "plain":"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
}
for k,s in samples.items():
    e=tk.encode(s, add_special_tokens=False)
    b=len(s.encode()); n=len(e.ids)
    print("  %-9s chars=%4d bytes=%4d tokens=%4d  bytes/tok=%.3f  chars/tok=%.3f" % (k,len(s),b,n,b/n,len(s)/n))
print()
print("=== add_special_tokens behavior ===")
e=tk.encode("hello world")
print("  encode('hello world') ids:", e.ids, "tokens:", e.tokens)
