"""Emulate _block_partials/_block_merge index formulas verbatim (float64) against reference attention."""
import math, torch
torch.manual_seed(0)
def emulate(query, key, value, position, scale, splits, block_n, chain):
    B, T, Hq, D = query.shape; Hkv, C = key.shape[1:3]; G = Hq // Hkv; M = T * G
    chunk = -(-C // splits)
    qf, kf, vf = query.reshape(-1), key.reshape(-1), value.reshape(-1)
    partial = torch.zeros(B * Hkv, splits, M, D, dtype=torch.float64); stats = torch.zeros(B * Hkv, splits, M, 2, dtype=torch.float64)
    pf, sf = partial.reshape(-1), stats.reshape(-1)
    for group in range(B * Hkv):
        row, kv_head = group // Hkv, group % Hkv
        for split in range(splits):
            first = int(position[row]) + 1
            begin = split * chunk; end = min(min(begin + chunk, C), first + (T - 1))
            for member in range(M):
                token = member // G
                q_head = kv_head * G + member % G
                q_offset = ((row * T + token) * Hq + q_head) * D
                q = qf[q_offset:q_offset + D]
                chained = token < chain
                valid = first + (token if chained else 0)
                own = -1 if chained else first - 1 + token
                maximum, denom, acc = -math.inf, 0.0, torch.zeros(D, dtype=torch.float64)
                for start in range(begin, end, block_n):
                    toks = [t for t in range(start, start + block_n)]
                    scores = []
                    for t in toks:
                        if t < end and (t < valid or t == own):
                            k = kf[group * C * D + t * D: group * C * D + t * D + D]
                            scores.append(float(q @ k) * scale)
                        else: scores.append(-math.inf)
                    nxt = max(maximum, max(scores)); pivot = 0.0 if nxt == -math.inf else nxt
                    probs = [math.exp(sc - pivot) if sc > -math.inf else 0.0 for sc in scores]
                    corr = math.exp(maximum - pivot) if maximum > -math.inf else 0.0
                    denom = denom * corr + sum(probs); acc = acc * corr
                    for t, pr in zip(toks, probs):
                        if pr: acc = acc + pr * vf[group * C * D + t * D: group * C * D + t * D + D]
                    maximum = nxt
                slot = (group * splits + split) * M + member
                pf[slot * D: slot * D + D] = acc; sf[slot * 2] = maximum; sf[slot * 2 + 1] = denom
    out = torch.zeros(B * T * Hq * D, dtype=torch.float64)
    for index in range(B * T * Hq):
        head = index % Hq; token = (index // Hq) % T; row = index // (Hq * T)
        group = row * Hkv + head // G; member = token * G + head % G
        maxima, denoms, parts = [], [], []
        for sp in range(splits):
            slot = (group * splits + sp) * M + member
            maxima.append(float(sf[slot * 2])); denoms.append(float(sf[slot * 2 + 1])); parts.append(pf[slot * D: slot * D + D])
        mx = max(maxima); corr = [math.exp(m - mx) if m > -math.inf else 0.0 for m in maxima]
        out[index * D:(index + 1) * D] = sum(p * c for p, c in zip(parts, corr)) / sum(d * c for d, c in zip(denoms, corr))
    return out.reshape(B, T, Hq, D)

def reference(query, key, value, position, scale, chain):
    B, T, Hq, D = query.shape; Hkv = key.shape[1]; G = Hq // Hkv
    out = torch.zeros_like(query)
    for b in range(B):
        for t in range(T):
            p = int(position[b])
            slots = list(range(p + t + 1)) if t < chain else list(range(p + 1)) + [p + t]
            for h in range(Hq):
                k, v = key[b, h // G, slots], value[b, h // G, slots]
                out[b, t, h] = torch.softmax((k @ query[b, t, h]) * scale, 0) @ v
    return out

for B, T, Hq, Hkv, D, C, splits, block_n in ((1, 5, 8, 2, 4, 23, 3, 4), (3, 4, 8, 4, 4, 19, 4, 4), (2, 3, 4, 4, 8, 12, 5, 2), (2, 5, 8, 2, 4, 40, 7, 8)):
    q = torch.randn(B, T, Hq, D, dtype=torch.float64); k = torch.randn(B, Hkv, C, D, dtype=torch.float64); v = torch.randn_like(k)
    k[:, :, :, :] += 0  # unused tail slots hold huge junk: they must never be read
    position = torch.randint(0, C - T, (B,))
    for b in range(B): k[b, :, int(position[b]) + T:] = 1e6; v[b, :, int(position[b]) + T:] = 1e6
    for chain in (T, max(1, T - 2), 2, 1):
        got, want = emulate(q, k, v, position, D ** -0.5, splits, block_n, chain), reference(q, k, v, position, D ** -0.5, chain)
        assert torch.allclose(got, want, atol=1e-9), (B, T, chain, float((got - want).abs().max()))
print("block attention index formulas (chain and tree masks) match the reference, incl. position 0, empty splits, junk tails")
