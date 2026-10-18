import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

tr_text = "Kim ne derse desin aşk için önce hoş sonra boş gelir Her seferinde canım yanar Aşk bana yalan gelir"
en_text = "No matter what anyone says, love starts off pleasant, then ends up feeling empty. Every single time, I end up getting hurt; to me, love feels like a lie."

for name, t in [("tr", tr_text), ("en", en_text)]:
    n_tokens = len(enc.encode(t))
    n_words = len(t.split())
    print(name, n_words, n_tokens, n_tokens / n_words)