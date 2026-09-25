"""Train-batch reciprocal weak-negative pairing; no labels or extra RNG."""
import re
import torch

def reciprocal_pairs(rows, device):
    n = len(rows); pairs = list(range(n)); valid = [False] * n
    texts = []; words = []
    for row in rows:
        text = str((row or {}).get("query", "")).lower()
        token = re.findall(r"[a-z0-9]+", text)
        texts.append(" ".join(token)); words.append(set(token))
    rejected = dict(missing=0, identity=0, duplicate=0, near_duplicate=0)
    for i in range(n):
        if valid[i]: continue
        for j in range(i+1, n):
            if valid[j]: continue
            a,b=rows[i] or {},rows[j] or {}
            if (not texts[i] or not texts[j] or a.get("vid") in (None, "") or b.get("vid") in (None, "")
                    or a.get("qid") is None or b.get("qid") is None):
                rejected["missing"] += 1; continue
            if str(a.get("qid")) == str(b.get("qid")) or a["vid"] == b["vid"]:
                rejected["identity"] += 1; continue
            if texts[i] == texts[j]:
                rejected["duplicate"] += 1; continue
            similarity=len(words[i]&words[j])/max(1,len(words[i]|words[j]))
            if similarity >= .5:
                rejected["near_duplicate"] += 1; continue
            pairs[i],pairs[j]=j,i;valid[i]=valid[j]=True;break
    return torch.tensor(pairs,device=device),torch.tensor(valid,device=device),rejected
